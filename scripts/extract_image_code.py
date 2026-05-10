"""
Extract product codes from catalog images and rename the existing image files.

Usage:
    python scripts/extract_image_code.py image.jpg
    python scripts/extract_image_code.py img1.jpg img2.jpg img3.jpg
    python scripts/extract_image_code.py images/
    python scripts/extract_image_code.py images/*.jpg
    python scripts/extract_image_code.py images/ --paddleocr --gpu
    python scripts/extract_image_code.py images/ --easyocr

This script renames the existing file in place. It does not copy the image.
"""

import glob
import os
import re
import sys
from pathlib import Path

import pytesseract
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
UPSCALE_FACTOR = 6
CONTRAST_BOOST = 5.0
TESSERACT_CONFIGS = (
    "--psm 7",
    "--psm 13",
    "--psm 6",
    "--psm 7 -c tessedit_char_whitelist=0123456789",
    "--psm 13 -c tessedit_char_whitelist=0123456789",
)
_EASYOCR_READER = None
_PADDLEOCR_READER = None

os.environ.setdefault("PADDLEOCR_HOME", str(Path.cwd() / ".paddleocr"))
os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(Path.cwd() / ".paddlex"))


def _find_caption_row(img):
    """
    Find the bright caption text area near the bottom of the image.
    Returns (y_top, y_bottom). Falls back to a fixed bottom band.
    """
    if not NUMPY_AVAILABLE:
        h = img.height
        return int(h * 0.955), int(h * 0.99)

    arr = np.array(img.convert("L"))
    h, w = arr.shape
    start_row = int(h * 0.90)

    bright_rows = []
    for y in range(start_row, h):
        bright = int(np.sum(arr[y, w // 2 :] > 190))
        if bright > 10:
            bright_rows.append(y)

    if bright_rows:
        pad = max(4, int(h * 0.004))
        return max(0, bright_rows[0] - pad), min(h, bright_rows[-1] + pad)

    return int(h * 0.955), int(h * 0.99)


def extract_code_candidates_from_text(text):
    """
    Return possible 8-digit product codes from OCR text.
    If OCR merges prefix numbers with the code, include the last 8 digits.
    """
    if not text:
        return []

    candidates = []

    groups = re.findall(r"\d+", text)
    for index, group in enumerate(groups):
        if len(group) == 8:
            candidates.append((group, "exact", index))
        elif len(group) > 8:
            candidates.append((group[-8:], "suffix", index))

    digits = re.sub(r"\D", "", text)
    if len(digits) >= 8:
        candidates.append((digits[-8:], "all_digits", len(groups)))

    return candidates


def extract_code_from_text(text):
    candidates = extract_code_candidates_from_text(text)
    if not candidates:
        return None
    return candidates[-1][0]


def _score_candidate(candidate, source, crop_index):
    code, match_type, group_index = candidate
    score = 0

    if code.startswith("100"):
        score += 50
    elif code.startswith("005"):
        score += 40

    if match_type == "exact":
        score += 12
    elif match_type == "suffix":
        score += 8
    else:
        score += 3

    if source == "tesseract_plain":
        score += 4
    elif source == "paddleocr":
        score += 6
    elif source == "easyocr":
        score += 3

    score += max(0, 8 - crop_index)
    score += min(group_index, 5)

    return score


def _preprocess_crop(crop, aggressive=False):
    """Create several OCR-friendly versions of a caption crop."""
    big = crop.resize(
        (max(1, crop.width * UPSCALE_FACTOR), max(1, crop.height * UPSCALE_FACTOR)),
        Image.LANCZOS,
    )
    gray = big.convert("L")
    enhanced = ImageEnhance.Contrast(gray).enhance(CONTRAST_BOOST)
    sharp = enhanced.filter(ImageFilter.SHARPEN)

    variants = [enhanced]

    if not aggressive:
        return variants

    variants.append(sharp)

    if NUMPY_AVAILABLE:
        arr = np.array(sharp)
        thresholds = (150, 190)
        for threshold in thresholds:
            binary = np.where(arr > threshold, 255, 0).astype("uint8")
            img = Image.fromarray(binary)
            variants.append(img)
            variants.append(ImageOps.invert(img))

    return variants


def _ocr_crop_candidates(crop, crop_index, aggressive=False):
    """Return scored candidate codes from one crop."""
    candidates = []

    for image in _preprocess_crop(crop, aggressive=aggressive):
        for config in TESSERACT_CONFIGS:
            text = pytesseract.image_to_string(image, config=config)
            source = "tesseract_whitelist" if "whitelist" in config else "tesseract_plain"
            for candidate in extract_code_candidates_from_text(text):
                candidates.append(
                    (candidate[0], _score_candidate(candidate, source, crop_index))
                )

    return candidates


def _get_easyocr_reader():
    global _EASYOCR_READER
    if _EASYOCR_READER is None:
        import easyocr

        _EASYOCR_READER = easyocr.Reader(["en"], gpu=False)
    return _EASYOCR_READER


def _get_paddleocr_reader(use_gpu=True):
    global _PADDLEOCR_READER
    if _PADDLEOCR_READER is None:
        from paddleocr import PaddleOCR

        try:
            _PADDLEOCR_READER = PaddleOCR(
                use_angle_cls=True,
                lang="en",
                use_gpu=use_gpu,
                show_log=False,
            )
        except TypeError:
            _PADDLEOCR_READER = PaddleOCR(use_angle_cls=True, lang="en")

    return _PADDLEOCR_READER


def _extract_paddle_texts(result):
    texts = []

    if not result:
        return texts

    # PaddleOCR v2 usually returns: [[box, (text, score)], ...] nested by image.
    pages = result if isinstance(result, list) else [result]
    for page in pages:
        if not page:
            continue
        for line in page:
            try:
                if isinstance(line, (list, tuple)) and len(line) >= 2:
                    text_info = line[1]
                    if isinstance(text_info, (list, tuple)) and text_info:
                        texts.append(str(text_info[0]))
                    elif isinstance(text_info, str):
                        texts.append(text_info)
            except Exception:
                continue

    return texts


def _paddleocr_crop_candidates(crop, crop_index, use_gpu=True):
    """GPU-friendly OCR fallback using PaddleOCR."""
    candidates = []

    try:
        reader = _get_paddleocr_reader(use_gpu=use_gpu)
        for image in _preprocess_crop(crop, aggressive=True):
            if not NUMPY_AVAILABLE:
                continue
            result = reader.ocr(np.array(image.convert("RGB")), cls=True)
            text = " ".join(_extract_paddle_texts(result))
            for candidate in extract_code_candidates_from_text(text):
                candidates.append(
                    (candidate[0], _score_candidate(candidate, "paddleocr", crop_index))
                )
    except Exception as exc:
        print(f"[PaddleOCR Error] {exc}")
        return candidates

    return candidates


def _easyocr_crop_candidates(crop, crop_index):
    """Fallback OCR for crops Tesseract cannot read."""
    candidates = []

    try:
        reader = _get_easyocr_reader()
        for image in _preprocess_crop(crop, aggressive=True):
            if not NUMPY_AVAILABLE:
                continue
            results = reader.readtext(np.array(image), detail=0, allowlist="0123456789")
            for candidate in extract_code_candidates_from_text(" ".join(results)):
                candidates.append(
                    (candidate[0], _score_candidate(candidate, "easyocr", crop_index))
                )
    except Exception:
        return candidates

    return candidates


def _caption_crops(img):
    """Generate likely caption regions, ordered from most to least specific."""
    w, h = img.size
    y_top, y_bot = _find_caption_row(img)

    crops = [
        img.crop((0, y_top, w, y_bot)),
        img.crop((0, int(h * 0.92), w, h)),
        img.crop((0, int(h * 0.88), w, h)),
        img.crop((w // 2, int(h * 0.90), w, h)),
    ]

    return [crop for crop in crops if crop.width > 0 and crop.height > 0]


def extract_product_code(
    image_path,
    use_easyocr=False,
    use_paddleocr=False,
    paddle_gpu=True,
):
    """
    Return the 8-digit product code found in the image caption, or None.
    """
    img = Image.open(image_path)

    crops = _caption_crops(img)

    scores = {}

    for crop_index, crop in enumerate(crops):
        for code, score in _ocr_crop_candidates(crop, crop_index, aggressive=False):
            scores[code] = scores.get(code, 0) + score

    if any(code.startswith(("100", "005")) for code in scores):
        return max(scores.items(), key=lambda item: item[1])[0]

    for crop_index, crop in enumerate(crops):
        for code, score in _ocr_crop_candidates(crop, crop_index, aggressive=True):
            scores[code] = scores.get(code, 0) + score

    if any(code.startswith(("100", "005")) for code in scores):
        return max(scores.items(), key=lambda item: item[1])[0]

    if use_paddleocr:
        for crop_index, crop in enumerate(crops):
            for code, score in _paddleocr_crop_candidates(
                crop,
                crop_index,
                use_gpu=paddle_gpu,
            ):
                scores[code] = scores.get(code, 0) + score

    if any(code.startswith(("100", "005")) for code in scores):
        return max(scores.items(), key=lambda item: item[1])[0]

    # EasyOCR is slow on CPU, so run it only when explicitly requested.
    if use_easyocr:
        for crop_index, crop in enumerate(crops):
            for code, score in _easyocr_crop_candidates(crop, crop_index):
                scores[code] = scores.get(code, 0) + score

    if scores:
        return max(scores.items(), key=lambda item: item[1])[0]

    return None


def collect_image_paths(inputs):
    paths = {}

    for item in inputs:
        path = Path(item)

        if path.is_dir():
            candidates = path.iterdir()
        elif path.exists():
            candidates = [path]
        else:
            candidates = (Path(match) for match in glob.glob(item))

        for candidate in candidates:
            if candidate.is_file() and candidate.suffix.lower() in IMAGE_EXTENSIONS:
                # resolve() prevents duplicate processing on case-insensitive Windows paths.
                paths[str(candidate.resolve()).lower()] = candidate

    return sorted(paths.values(), key=lambda item: str(item).lower())


def rename_image_with_code(path, code):
    """
    Rename the existing image file to <code><same extension>.
    It never overwrites another file.
    """
    target = path.with_name(f"{code}{path.suffix.lower()}")

    if path.resolve() == target.resolve():
        return "already_named", target

    if target.exists():
        return "target_exists", target

    path.rename(target)
    return "renamed", target


def main():
    if len(sys.argv) < 2:
        print(
            "Usage: python scripts/extract_image_code.py <image_or_folder> "
            "[more images ...] [--paddleocr] [--gpu] [--easyocr]"
        )
        sys.exit(1)

    use_easyocr = "--easyocr" in sys.argv
    use_paddleocr = "--paddleocr" in sys.argv
    paddle_gpu = "--cpu" not in sys.argv
    flags = {"--easyocr", "--paddleocr", "--gpu", "--cpu"}
    inputs = [arg for arg in sys.argv[1:] if arg not in flags]

    paths = collect_image_paths(inputs)
    if not paths:
        print("No image files found.")
        sys.exit(1)

    col_w = max(len(path.name) for path in paths) + 2
    print(f"{'File':<{col_w}} Product Code  Status")
    print("-" * (col_w + 35))

    extracted_count = 0
    renamed_count = 0

    for path in paths:
        try:
            if not path.exists():
                print(f"{path.name:<{col_w}} MISSING       file no longer exists")
                continue

            code = extract_product_code(
                str(path),
                use_easyocr=use_easyocr,
                use_paddleocr=use_paddleocr,
                paddle_gpu=paddle_gpu,
            )
            if not code:
                print(f"{path.name:<{col_w}} NOT FOUND     not renamed")
                continue

            extracted_count += 1
            status, new_path = rename_image_with_code(path, code)

            if status == "renamed":
                renamed_count += 1
                print(f"{path.name:<{col_w}} {code}     renamed -> {new_path.name}")
            elif status == "already_named":
                print(f"{path.name:<{col_w}} {code}     already named")
            else:
                print(f"{path.name:<{col_w}} {code}     skipped, {new_path.name} exists")
        except Exception as exc:
            print(f"{path.name:<{col_w}} ERROR: {exc}")

    print("-" * (col_w + 35))
    print(f"Processed {len(paths)} image(s). Extracted {extracted_count} code(s). Renamed {renamed_count} file(s).")


if __name__ == "__main__":
    main()
