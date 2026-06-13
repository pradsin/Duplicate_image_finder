# vif.py
# Run:
# python vif.py --use-cache -i "D:\input.jpg" -f "[D:\Folder1]:[E:\Folder2]" -s 90

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sqlite3
import sys
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
from tqdm import tqdm


SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"
}

HASH_SIZE = 32
PHASH_BITS = 64
HIST_BINS = (32, 32)
DEFAULT_SIMILARITY = 80.0


class ErrorCode(str, Enum):
    INVALID_ARGUMENT = "INVALID_ARGUMENT"
    INPUT_FILE_NOT_FOUND = "INPUT_FILE_NOT_FOUND"
    INPUT_FILE_NOT_READABLE = "INPUT_FILE_NOT_READABLE"
    INPUT_FILE_NOT_IMAGE = "INPUT_FILE_NOT_IMAGE"
    FOLDER_NOT_FOUND = "FOLDER_NOT_FOUND"
    FOLDER_NOT_READABLE = "FOLDER_NOT_READABLE"
    DATABASE_ERROR = "DATABASE_ERROR"
    IMAGE_PROCESSING_ERROR = "IMAGE_PROCESSING_ERROR"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    CANCELLED = "CANCELLED"
    UNKNOWN_ERROR = "UNKNOWN_ERROR"


class DifError(Exception):
    def __init__(self, code: ErrorCode, message: str, path: str | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.path = path

    def to_json(self) -> dict:
        return {
            "success": False,
            "error": {
                "code": self.code.value,
                "message": self.message,
                "path": self.path,
            },
        }


@dataclass(frozen=True)
class ImageFeature:
    path: str
    file_size: int
    modified_time: float
    phash: int
    histogram: bytes


def get_default_db_path() -> Path:
    appdata = os.getenv("APPDATA") or str(Path.home() / "AppData" / "Roaming")
    db_dir = Path(appdata) / "DIF"
    db_dir.mkdir(parents=True, exist_ok=True)
    return db_dir / "dif_cache.sqlite"


def parse_folders(raw_value: str) -> list[Path]:
    folders = re.findall(r"\[(.*?)]", raw_value)
    if not folders:
        raise DifError(
            ErrorCode.INVALID_ARGUMENT,
            'Invalid folder format. Use: -f "[D:\\Photos]:[E:\\Images]"',
        )
    return [Path(folder).expanduser().resolve() for folder in folders]


def iter_image_files(folders: Sequence[Path]) -> Iterable[Path]:
    for folder in folders:
        try:
            if not folder.exists():
                logging.warning("Folder does not exist: %s", folder)
                continue

            if not folder.is_dir():
                logging.warning("Path is not a folder: %s", folder)
                continue

            for path in folder.rglob("*"):
                try:
                    if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                        yield path.resolve()
                except PermissionError:
                    logging.warning("Permission denied: %s", path)
                except OSError as error:
                    logging.warning("Cannot access path %s: %s", path, error)

        except PermissionError:
            logging.warning("Permission denied for folder: %s", folder)
        except OSError as error:
            logging.warning("Cannot scan folder %s: %s", folder, error)


def read_image(path: Path, *, is_input: bool = False) -> np.ndarray:
    try:
        path = path.expanduser().resolve()

        if not path.exists():
            raise DifError(
                ErrorCode.INPUT_FILE_NOT_FOUND if is_input else ErrorCode.IMAGE_PROCESSING_ERROR,
                f"File does not exist: {path}",
                str(path),
            )

        if not path.is_file():
            raise DifError(
                ErrorCode.INVALID_ARGUMENT,
                f"Path is not a file: {path}",
                str(path),
            )

        with path.open("rb"):
            pass

        image = cv2.imread(str(path), cv2.IMREAD_COLOR)

        if image is None:
            raise DifError(
                ErrorCode.INPUT_FILE_NOT_IMAGE if is_input else ErrorCode.IMAGE_PROCESSING_ERROR,
                f"File cannot be decoded as a supported image: {path}",
                str(path),
            )

        return image

    except PermissionError:
        raise DifError(
            ErrorCode.PERMISSION_DENIED,
            f"Permission denied while reading file: {path}",
            str(path),
        )
    except OSError as error:
        raise DifError(
            ErrorCode.INPUT_FILE_NOT_READABLE if is_input else ErrorCode.IMAGE_PROCESSING_ERROR,
            f"Unable to read file: {path}. Reason: {error}",
            str(path),
        )


def compute_phash(image: np.ndarray) -> int:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (HASH_SIZE, HASH_SIZE), interpolation=cv2.INTER_AREA)

    dct = cv2.dct(np.float32(resized))
    dct_low = dct[:8, :8]

    median = np.median(dct_low[1:])
    bits = dct_low > median

    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bit)

    return value


def compute_histogram(image: np.ndarray) -> bytes:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    hist = cv2.calcHist(
        [hsv],
        [0, 1],
        None,
        list(HIST_BINS),
        [0, 180, 0, 256],
    )

    cv2.normalize(hist, hist)
    return hist.astype(np.float32).tobytes()


def compute_feature(path: Path, *, is_input: bool = False) -> ImageFeature:
    path = path.expanduser().resolve()
    image = read_image(path, is_input=is_input)
    stat = path.stat()

    return ImageFeature(
        path=str(path),
        file_size=stat.st_size,
        modified_time=stat.st_mtime,
        phash=compute_phash(image),
        histogram=compute_histogram(image),
    )


def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def phash_similarity(a: int, b: int) -> float:
    return ((PHASH_BITS - hamming_distance(a, b)) / PHASH_BITS) * 100.0


def histogram_similarity(hist_a: bytes, hist_b: bytes) -> float:
    a = np.frombuffer(hist_a, dtype=np.float32)
    b = np.frombuffer(hist_b, dtype=np.float32)

    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator == 0:
        return 0.0

    return float(np.dot(a, b) / denominator) * 100.0


def connect_db(db_path: Path) -> sqlite3.Connection:
    db_path = db_path.expanduser().resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA mmap_size=30000000000")

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS images (
            path TEXT PRIMARY KEY,
            file_size INTEGER NOT NULL,
            modified_time REAL NOT NULL,
            phash TEXT NOT NULL,
            histogram BLOB NOT NULL,
            indexed_at REAL NOT NULL
        )
        """
    )

    columns = {
        row[1]: row[2]
        for row in conn.execute("PRAGMA table_info(images)").fetchall()
    }

    if columns.get("phash", "").upper() == "INTEGER":
        conn.execute("ALTER TABLE images RENAME TO images_old")

        conn.execute(
            """
            CREATE TABLE images (
                path TEXT PRIMARY KEY,
                file_size INTEGER NOT NULL,
                modified_time REAL NOT NULL,
                phash TEXT NOT NULL,
                histogram BLOB NOT NULL,
                indexed_at REAL NOT NULL
            )
            """
        )

        conn.execute(
            """
            INSERT OR REPLACE INTO images
            SELECT
                path,
                file_size,
                modified_time,
                CAST(phash AS TEXT),
                histogram,
                indexed_at
            FROM images_old
            """
        )

        conn.commit()

    return conn


def get_cached_metadata(conn: sqlite3.Connection) -> dict[str, tuple[int, float]]:
    rows = conn.execute(
        "SELECT path, file_size, modified_time FROM images"
    ).fetchall()

    return {row[0]: (row[1], row[2]) for row in rows}


def update_cache(conn: sqlite3.Connection, folders: Sequence[Path]) -> None:
    print("Scanning folders...", file=sys.stderr)

    files = list(iter_image_files(folders))
    cached = get_cached_metadata(conn)

    changed_files: list[Path] = []

    for path in files:
        try:
            resolved = str(path.resolve())
            stat = path.stat()
            old = cached.get(resolved)

            if old != (stat.st_size, stat.st_mtime):
                changed_files.append(path)

        except PermissionError:
            logging.warning("Permission denied: %s", path)
        except OSError as error:
            logging.warning("Cannot stat file %s: %s", path, error)

    print(f"Images found: {len(files)}", file=sys.stderr)
    print(f"Images to index/update: {len(changed_files)}", file=sys.stderr)

    rows = []

    for path in tqdm(changed_files, desc="Indexing", unit="image"):
        try:
            feature = compute_feature(path)

            rows.append(
                (
                    feature.path,
                    feature.file_size,
                    feature.modified_time,
                    str(feature.phash),
                    feature.histogram,
                    time.time(),
                )
            )

            if len(rows) >= 500:
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO images
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )
                conn.commit()
                rows.clear()

        except DifError as error:
            logging.warning("%s", error.message)
        except sqlite3.Error:
            raise
        except Exception as error:
            logging.warning("Failed to index %s: %s", path, error)

    if rows:
        conn.executemany(
            """
            INSERT OR REPLACE INTO images
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()


def compute_similarity(
    input_phash: int,
    input_histogram: bytes,
    candidate_phash: int,
    candidate_histogram: bytes,
) -> tuple[float, float, float]:
    hash_score = phash_similarity(input_phash, candidate_phash)

    if hash_score < 50:
        return hash_score, 0.0, 0.0

    hist_score = histogram_similarity(input_histogram, candidate_histogram)

    final_score = round((hash_score * 0.65) + (hist_score * 0.35), 2)
    return final_score, round(hash_score, 2), round(hist_score, 2)


def search_cache(
    conn: sqlite3.Connection,
    input_image: Path,
    minimum_similarity: float,
) -> list[dict]:
    input_feature = compute_feature(input_image, is_input=True)

    rows = conn.execute(
        "SELECT path, phash, histogram FROM images"
    ).fetchall()

    results = []

    for path, cached_phash, cached_histogram in tqdm(
        rows,
        desc="Searching cache",
        unit="image",
    ):
        try:
            if path == input_feature.path:
                continue

            final_score, hash_score, hist_score = compute_similarity(
                input_feature.phash,
                input_feature.histogram,
                int(cached_phash),
                cached_histogram,
            )

            if final_score >= minimum_similarity:
                results.append(
                    {
                        "path": path,
                        "similarity": final_score,
                        "phash_similarity": hash_score,
                        "histogram_similarity": hist_score,
                    }
                )

        except Exception as error:
            logging.warning("Skipping cached row %s: %s", path, error)

    return sorted(results, key=lambda item: item["similarity"], reverse=True)


def search_without_cache(
    input_image: Path,
    folders: Sequence[Path],
    minimum_similarity: float,
) -> list[dict]:
    input_feature = compute_feature(input_image, is_input=True)
    files = list(iter_image_files(folders))

    results = []

    for path in tqdm(files, desc="Searching", unit="image"):
        try:
            feature = compute_feature(path)

            if feature.path == input_feature.path:
                continue

            final_score, hash_score, hist_score = compute_similarity(
                input_feature.phash,
                input_feature.histogram,
                feature.phash,
                feature.histogram,
            )

            if final_score >= minimum_similarity:
                results.append(
                    {
                        "path": feature.path,
                        "similarity": final_score,
                        "phash_similarity": hash_score,
                        "histogram_similarity": hist_score,
                    }
                )

        except DifError as error:
            logging.warning("%s", error.message)
        except Exception as error:
            logging.warning("Skipping %s: %s", path, error)

    return sorted(results, key=lambda item: item["similarity"], reverse=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dif",
        description="Find visually similar images recursively.",
    )

    parser.add_argument("-i", "--input", required=True, type=Path)
    parser.add_argument("-f", "--folders", required=True)

    parser.add_argument(
        "-s",
        "--similarity",
        type=float,
        default=DEFAULT_SIMILARITY,
        help="Similarity percentage from 50 to 100.",
    )

    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Build and use local cache database.",
    )

    parser.add_argument(
        "--db",
        type=Path,
        default=get_default_db_path(),
        help="SQLite DB path. Default: %%APPDATA%%\\DIF\\dif_cache.sqlite",
    )

    return parser


def run(argv: Sequence[str]) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.similarity < 50 or args.similarity > 100:
        raise DifError(
            ErrorCode.INVALID_ARGUMENT,
            "Similarity must be between 50 and 100.",
        )

    input_image = args.input.expanduser().resolve()

    if not input_image.exists():
        raise DifError(
            ErrorCode.INPUT_FILE_NOT_FOUND,
            f"Input image does not exist: {input_image}",
            str(input_image),
        )

    folders = parse_folders(args.folders)

    if args.use_cache:
        db_path = args.db.expanduser().resolve()
        print(f"Using cache DB: {db_path}", file=sys.stderr)

        conn = connect_db(db_path)

        try:
            update_cache(conn, folders)
            matches = search_cache(conn, input_image, args.similarity)
        finally:
            conn.close()

    else:
        matches = search_without_cache(input_image, folders, args.similarity)

    print(
        json.dumps(
            {
                "success": True,
                "input": str(input_image),
                "use_cache": args.use_cache,
                "db": str(args.db.expanduser().resolve()) if args.use_cache else None,
                "minimum_similarity": args.similarity,
                "matches_found": len(matches),
                "matches": matches,
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    return 0


def main(argv: Sequence[str]) -> int:
    try:
        return run(argv)

    except DifError as error:
        print(json.dumps(error.to_json(), indent=2), file=sys.stderr)
        return 2

    except sqlite3.Error as error:
        payload = {
            "success": False,
            "error": {
                "code": ErrorCode.DATABASE_ERROR.value,
                "message": str(error),
                "path": None,
            },
        }
        print(json.dumps(payload, indent=2), file=sys.stderr)
        return 3

    except KeyboardInterrupt:
        payload = {
            "success": False,
            "error": {
                "code": ErrorCode.CANCELLED.value,
                "message": "Operation cancelled by user.",
                "path": None,
            },
        }
        print(json.dumps(payload, indent=2), file=sys.stderr)
        return 130

    except Exception as error:
        payload = {
            "success": False,
            "error": {
                "code": ErrorCode.UNKNOWN_ERROR.value,
                "message": str(error),
                "path": None,
            },
        }
        print(json.dumps(payload, indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s: %(message)s",
    )
    try:
        cv2.setLogLevel(0)
    except AttributeError:
        try:
            cv2.utils.logging.setLogLevel(
                cv2.utils.logging.LOG_LEVEL_ERROR
            )
        except Exception:
            pass
    raise SystemExit(main(sys.argv[1:]))