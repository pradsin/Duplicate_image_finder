# dif.py — Closest image finder using perceptual hashing + ORB feature matching
#
# pip install opencv-python Pillow imagehash numpy
#
# Usage:
#   python dif.py -i path/to/target.jpg -if path/to/search_folder

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import cv2
import imagehash
import numpy as np
from PIL import Image, UnidentifiedImageError

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Weight given to each scoring component when blending the final score.
# pHash captures global perceptual similarity; ORB captures local structural
# keypoint matches that survive watermarks/overlays well.
PHASH_WEIGHT = 0.4
ORB_WEIGHT = 0.6

# Maximum Hamming distance for pHash (0 = identical, 64 = completely different).
# imagehash.phash() produces a 64-bit hash by default.
MAX_PHASH_DISTANCE = 64.0

# ORB detector settings
ORB_N_FEATURES = 1000    # Max keypoints to detect per image
ORB_LOWE_RATIO = 0.75    # Ratio test threshold for match filtering

# Two scores within this tolerance of the best score are treated as equal
# matches and all returned together.
SCORE_TIE_TOLERANCE = 0.01

# Reusable BFMatcher instance — created once, not per comparison.
# crossCheck=False is required for knnMatch (k=2) to work correctly.
_BF_MATCHER = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)


# ---------------------------------------------------------------------------
# Image collection
# ---------------------------------------------------------------------------

def collect_images(folder: Path) -> list[Path]:
    """Recursively find all supported image files under *folder*."""
    images = []
    for root, _dirs, files in os.walk(folder):
        for fname in files:
            if Path(fname).suffix.lower() in SUPPORTED_EXTENSIONS:
                images.append(Path(root) / fname)
    return images


# ---------------------------------------------------------------------------
# Perceptual hash helpers
# ---------------------------------------------------------------------------

def compute_phash(image_path: Path) -> Optional[imagehash.ImageHash]:
    """
    Return the pHash for *image_path*, or None on failure.

    Pillow is used here because it handles colour mode conversion
    (CMYK, palette, etc.) that cv2.imread can misread or skip.
    img.convert("RGB") normalises the mode before hashing so the
    hash is consistent regardless of source colour space.
    """
    try:
        with Image.open(image_path) as img:
            # FIX 1: convert to RGB before hashing.
            # Without this, RGBA/palette/CMYK images produce hashes that are
            # incompatible with each other, inflating Hamming distances falsely.
            return imagehash.phash(img.convert("RGB"))
    except (UnidentifiedImageError, OSError):
        return None
    except Exception:
        # Catch-all for unexpected decode errors (e.g. truncated files)
        return None


def phash_similarity(hash_a: imagehash.ImageHash,
                     hash_b: imagehash.ImageHash) -> float:
    """
    Convert Hamming distance between two pHashes into a [0, 1] similarity
    score where 1.0 means identical and 0.0 means maximally different.
    """
    distance = hash_a - hash_b   # imagehash overloads __sub__ for Hamming distance
    return 1.0 - (distance / MAX_PHASH_DISTANCE)


# ---------------------------------------------------------------------------
# ORB feature-matching helpers
# ---------------------------------------------------------------------------

def load_gray(image_path: Path) -> Optional[np.ndarray]:
    """
    Load an image as a grayscale NumPy array for OpenCV, or None on failure.

    FIX 2: Use Pillow for loading and convert to grayscale via numpy.
    cv2.imread uses the OS file path directly and silently fails on:
      - Non-ASCII / Unicode characters in the path (Windows especially)
      - Files whose extension doesn't match their actual format
    Pillow handles both cases correctly.
    """
    try:
        with Image.open(image_path) as img:
            gray = np.array(img.convert("L"))  # "L" = 8-bit grayscale
        if gray.size == 0:
            return None
        return gray
    except (UnidentifiedImageError, OSError):
        return None
    except Exception:
        return None


def compute_orb_descriptors(
    gray: np.ndarray,
    orb: cv2.ORB,
) -> tuple[list, Optional[np.ndarray]]:
    """Detect keypoints and compute ORB descriptors for a grayscale image."""
    # FIX 3: detectAndCompute can return None for descriptors on blank/uniform
    # images with zero keypoints. Callers already handle None, but being
    # explicit here avoids a confusing AttributeError downstream.
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    return keypoints, descriptors  # descriptors may legitimately be None


def orb_similarity(
    desc_a: Optional[np.ndarray],
    desc_b: Optional[np.ndarray],
) -> float:
    """
    Match two ORB descriptor sets using a BF Hamming matcher + Lowe ratio test
    and return a normalised [0, 1] similarity score.

    Falls back to 0.0 if either descriptor set is None or too small.
    """
    if desc_a is None or desc_b is None:
        return 0.0
    if len(desc_a) < 2 or len(desc_b) < 2:
        return 0.0

    try:
        # knnMatch returns the 2 nearest neighbours for each descriptor in desc_a.
        # FIX 4: knnMatch can return singleton tuples (only 1 neighbour found)
        # when desc_b has very few descriptors. Guard against unpacking errors.
        raw_matches = _BF_MATCHER.knnMatch(desc_a, desc_b, k=2)
    except cv2.error:
        return 0.0

    # Lowe's ratio test: keep only unambiguous matches (clear nearest neighbour)
    good_matches = [
        m for pair in raw_matches
        if len(pair) == 2                          # guard: skip singleton tuples
        for m, n in (pair,)
        if m.distance < ORB_LOWE_RATIO * n.distance
    ]

    # Normalise by the smaller descriptor count so score stays in [0, 1]
    max_possible = min(len(desc_a), len(desc_b))
    if max_possible == 0:
        return 0.0

    return len(good_matches) / max_possible


# ---------------------------------------------------------------------------
# Combined scoring
# ---------------------------------------------------------------------------

def combined_score(phash_sim: float, orb_sim: float) -> float:
    """
    Blend pHash and ORB similarity into a single score in [0, 1].
    Higher means more similar to the target.
    """
    return PHASH_WEIGHT * phash_sim + ORB_WEIGHT * orb_sim


# ---------------------------------------------------------------------------
# Core search logic
# ---------------------------------------------------------------------------

def find_closest(target_path: Path, search_folder: Path) -> list[Path]:
    """
    Search *search_folder* recursively and return all images whose similarity
    score is within SCORE_TIE_TOLERANCE of the single best score found.

    Returns a sorted list of Paths (empty if no candidates could be scored).

    Strategy
    --------
    1. Compute the target's pHash and ORB descriptors once up front.
    2. Score every candidate and accumulate (score, path) pairs.
    3. Find the best score, then collect all candidates within the tolerance
       band — these are all "equally close" matches.
    """
    # --- Compute target features once ----------------------------------
    target_phash = compute_phash(target_path)
    if target_phash is None:
        print(f"Error: Cannot open target image '{target_path}'.", file=sys.stderr)
        return []

    target_gray = load_gray(target_path)

    # OPT 1: Create a single ORB instance and reuse it across all candidates.
    # cv2.ORB_create() allocates internal buffers; creating it per-image
    # wastes time and memory for large folders.
    orb = cv2.ORB_create(nfeatures=ORB_N_FEATURES)

    if target_gray is not None:
        _target_kp, target_desc = compute_orb_descriptors(target_gray, orb)
    else:
        target_desc = None

    # --- Collect candidates --------------------------------------------
    candidates = collect_images(search_folder)
    if not candidates:
        print(f"Error: No supported images found under '{search_folder}'.",
              file=sys.stderr)
        return []

    # OPT 2: Resolve the target path once outside the loop instead of
    # re-resolving it on every iteration.
    resolved_target = target_path.resolve()

    # Store (score, path) for every successfully scored candidate
    scored: list[tuple[float, Path]] = []

    for candidate_path in candidates:
        # Skip if the candidate IS the target (same resolved path)
        if candidate_path.resolve() == resolved_target:
            continue

        # --- pHash component -------------------------------------------
        cand_phash = compute_phash(candidate_path)
        if cand_phash is None:
            # Unreadable by Pillow — skip silently
            continue
        ph_sim = phash_similarity(target_phash, cand_phash)

        # OPT 3: Early-skip ORB (the expensive step) when pHash already
        # signals the images are perceptually very different. If the pHash
        # similarity is so low that even a perfect ORB score (1.0) couldn't
        # push the combined score above the current best, skip ORB entirely.
        # This is a safe optimisation because it only skips clearly poor matches.
        if scored:
            current_best = max(s for s, _ in scored)
            max_achievable = combined_score(ph_sim, 1.0)
            if max_achievable < current_best - SCORE_TIE_TOLERANCE:
                # Can't possibly beat the current best — skip ORB
                score = combined_score(ph_sim, 0.0)
                scored.append((score, candidate_path))
                continue

        # --- ORB component ---------------------------------------------
        cand_gray = load_gray(candidate_path)
        if cand_gray is not None:
            _cand_kp, cand_desc = compute_orb_descriptors(cand_gray, orb)
            orb_sim = orb_similarity(target_desc, cand_desc)
        else:
            orb_sim = 0.0

        score = combined_score(ph_sim, orb_sim)
        scored.append((score, candidate_path))

    if not scored:
        return []

    # --- Collect all matches within tolerance of the best score --------
    best_score = max(s for s, _ in scored)

    matches = [
        path for score, path in scored
        if score >= best_score - SCORE_TIE_TOLERANCE
    ]

    # Sort for deterministic, readable output
    return sorted(matches)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find the closest matching image(s) in a folder to a target image."
    )
    parser.add_argument(
        "-i",
        required=True,
        metavar="INPUT_IMAGE",
        help="Path to the target input image.",
    )
    parser.add_argument(
        "-if",
        dest="input_folder",
        required=True,
        metavar="INPUT_FOLDER",
        help="Path to the folder (searched recursively) containing candidate images.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    target_path = Path(args.i).expanduser().resolve()
    search_folder = Path(args.input_folder).expanduser().resolve()

    # --- Basic path validation -----------------------------------------
    if not target_path.is_file():
        print(f"Error: Target image not found: '{target_path}'", file=sys.stderr)
        sys.exit(1)

    if not search_folder.is_dir():
        print(f"Error: Search folder not found: '{search_folder}'", file=sys.stderr)
        sys.exit(1)

    # FIX 5: Validate that the target file is actually a supported image
    # format before running any processing. Catches cases like passing a
    # .txt or .pdf file accidentally via the -i argument.
    if target_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        print(
            f"Error: Target file '{target_path.name}' is not a supported image format. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
            file=sys.stderr,
        )
        sys.exit(1)

    # --- Run search ----------------------------------------------------
    results = find_closest(target_path, search_folder)

    if not results:
        print("No matching image could be found.", file=sys.stderr)
        sys.exit(1)

    # Print one absolute path per line — single result or multiple ties
    for match in results:
        print(match.resolve())


if __name__ == "__main__":
    main()