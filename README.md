# dif.py — Closest Image Finder

A production-ready Python 3 command-line tool that finds the closest matching image(s) from a folder tree compared to a target input image. Designed to be resilient to watermarks, compression artifacts, and colour-mode differences.

---

## How It Works

The script uses a **dual-method scoring approach** that blends two independent similarity signals:

| Method | Weight | What it captures |
|---|---|---|
| **Perceptual Hashing (pHash)** | 40% | Global visual structure — overall composition, brightness, layout |
| **ORB Feature Matching** | 60% | Local structural keypoints — corners, edges, textures that survive overlays |

Combining both makes the tool robust against watermarks, logos, and compression — a watermark disrupts the global hash slightly but barely affects the hundreds of background keypoints that ORB matches.

A **tie tolerance band** (`SCORE_TIE_TOLERANCE = 0.01`) ensures that multiple images scoring within 1% of the best score are all returned, rather than arbitrarily picking one.

---

## Requirements

- Python 3.10+
- The following libraries:

```
opencv-python>=4.8.0
Pillow>=10.0.0
imagehash>=4.3.1
numpy>=1.24.0
```

Install all dependencies at once:

```bash
pip install -r requirements.txt
```

---

## Installation

```bash
# 1. Clone or download the project
git clone https://github.com/your-username/dif.git
cd dif

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Usage

```bash
python dif.py -i <target_image> -if <search_folder>
```

### Arguments

| Argument | Required | Description |
|---|---|---|
| `-i` | Yes | Path to the target input image to match against |
| `-if` | Yes | Path to the folder to search (all subfolders are included) |

### Supported Image Formats

`.jpg` · `.jpeg` · `.png` · `.bmp` · `.webp`

---

## Examples

**Basic usage:**
```bash
python dif.py -i photo.jpg -if C:\Pictures
```

**With subdirectories (always recursive):**
```bash
python dif.py -i ~/Desktop/target.png -if ~/Documents/ImageArchive
```

**Finding duplicates across a painting collection:**
```bash
python dif.py -i original.jpg -if "C:\ABCD\Paintings"
```

### Output

The script prints one absolute path per line for every image that matches within the tie tolerance:

```
# Single match
C:\ABCD\Paintings\portrait.jpg

# Multiple equally-close matches
C:\ABCD\Paintings\portrait.jpg
C:\ABCD\Paintings\Copies\portrait_hd.png
C:\Archive\portrait_watermarked.webp
```

If no match is found, a message is printed to `stderr` and the script exits with code `1`.

---

## Configuration

All tunable constants are at the top of `dif.py`:

| Constant | Default | Description |
|---|---|---|
| `PHASH_WEIGHT` | `0.4` | Weight of perceptual hash in the final score |
| `ORB_WEIGHT` | `0.6` | Weight of ORB feature matching in the final score |
| `ORB_N_FEATURES` | `1000` | Max keypoints ORB detects per image |
| `ORB_LOWE_RATIO` | `0.75` | Lowe's ratio test threshold — lower = stricter matching |
| `SCORE_TIE_TOLERANCE` | `0.01` | Score band within which images are treated as equal matches |
| `MAX_PHASH_DISTANCE` | `64.0` | Max possible Hamming distance for a 64-bit pHash |

**Tuning tips:**
- Increase `ORB_WEIGHT` if your images frequently have large watermarks covering central content.
- Lower `SCORE_TIE_TOLERANCE` (e.g. `0.005`) to return fewer but stricter matches.
- Increase `ORB_N_FEATURES` for highly detailed images; decrease it for faster runs on large folders.

---

## Error Handling

| Situation | Behaviour |
|---|---|
| Target image not found | Prints error to `stderr`, exits with code `1` |
| Target is not a supported format | Prints error to `stderr`, exits with code `1` |
| Search folder not found | Prints error to `stderr`, exits with code `1` |
| Search folder has no images | Prints error to `stderr`, exits with code `1` |
| Corrupt / unreadable candidate image | Silently skipped, search continues |
| Candidate with Unicode path (Windows) | Handled correctly via Pillow |
| Candidate is the same file as target | Automatically excluded from results |

---

## Project Structure

```
dif/
├── dif.py            # Main script
├── requirements.txt  # Python dependencies
└── README.md         # This file
```

---

## Dependencies & Licences

| Library | Licence | Purpose |
|---|---|---|
| [opencv-python](https://github.com/opencv/opencv-python) | Apache 2.0 | ORB keypoint detection and descriptor matching |
| [Pillow](https://python-pillow.org/) | HPND | Image loading, colour mode normalisation |
| [imagehash](https://github.com/JohannesBuchner/imagehash) | BSD 2-Clause | Perceptual hashing (pHash) |
| [numpy](https://numpy.org/) | BSD 3-Clause | Array operations for grayscale conversion |
