"""
preprocess.py
-------------
Face detection, landmark extraction, alignment, and cropping pipeline.

Directory structure expected:
    dataset/raw/<person_name>/<image_file>   →  input
    dataset/processed/<person_name>/<file>   →  output (112×112 aligned faces)

Usage:
    python preprocess.py [--input dataset/raw] [--output dataset/processed]
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Optional: suppress TensorFlow/PyTorch startup noise before importing MTCNN
# ---------------------------------------------------------------------------
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

try:
    from facenet_pytorch import MTCNN
    import torch
    _MTCNN_AVAILABLE = True
except ImportError:
    _MTCNN_AVAILABLE = False

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARGET_SIZE = (112, 112)          # output face crop size
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

# Reference 5-point landmarks for a canonical 112×112 face
# (left-eye, right-eye, nose, left-mouth, right-mouth)
REFERENCE_LANDMARKS = np.array(
    [
        [38.2946,  51.6963],
        [73.5318,  51.5014],
        [56.0252,  71.7366],
        [41.5493,  92.3655],
        [70.7299,  92.2041],
    ],
    dtype=np.float32,
)


# ---------------------------------------------------------------------------
# Detector initialisation
# ---------------------------------------------------------------------------

def build_detector(device: Optional[str] = None) -> "MTCNN":
    """
    Initialise MTCNN from facenet-pytorch.

    Parameters
    ----------
    device : str | None
        'cuda', 'cpu', or None (auto-detect).

    Returns
    -------
    MTCNN detector instance configured to return landmarks.
    """
    if not _MTCNN_AVAILABLE:
        log.error("facenet-pytorch is not installed. Run: pip install facenet-pytorch")
        sys.exit(1)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    log.info("Initialising MTCNN on device: %s", device)
    detector = MTCNN(
        image_size=160,          # internal size; we crop manually afterwards
        margin=0,
        min_face_size=20,
        thresholds=[0.6, 0.7, 0.7],
        factor=0.709,
        post_process=False,      # return raw pixel tensor
        keep_all=True,           # detect ALL faces in the image
        device=device,
    )
    return detector


# ---------------------------------------------------------------------------
# Core geometry helpers
# ---------------------------------------------------------------------------

def estimate_affine_from_landmarks(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
) -> np.ndarray:
    """
    Compute a 2×3 affine transformation matrix that maps *src_pts* → *dst_pts*
    using OpenCV's ``estimateAffinePartial2D`` (similarity: scale + rotation +
    translation; no shear).

    Parameters
    ----------
    src_pts : (5, 2) float32 array  – detected landmarks
    dst_pts : (5, 2) float32 array  – reference / target landmarks

    Returns
    -------
    M : (2, 3) float32 affine matrix
    """
    M, _ = cv2.estimateAffinePartial2D(
        src_pts.reshape(-1, 1, 2),
        dst_pts.reshape(-1, 1, 2),
        method=cv2.LMEDS,
    )
    if M is None:
        raise ValueError("estimateAffinePartial2D returned None – degenerate points?")
    return M


def align_face(
    image_bgr: np.ndarray,
    landmarks_5pt: np.ndarray,
    target_size: tuple[int, int] = TARGET_SIZE,
    ref_landmarks: np.ndarray = REFERENCE_LANDMARKS,
) -> np.ndarray:
    """
    Align a single face crop via affine warp.

    Parameters
    ----------
    image_bgr      : full input image in BGR (H×W×3 uint8).
    landmarks_5pt  : (5, 2) float32 array of detected facial landmarks.
    target_size    : (width, height) of the output crop.
    ref_landmarks  : (5, 2) reference landmarks for the target_size canvas.

    Returns
    -------
    aligned : (H, W, 3) uint8 BGR aligned face image.
    """
    M = estimate_affine_from_landmarks(landmarks_5pt, ref_landmarks)
    aligned = cv2.warpAffine(
        image_bgr,
        M,
        target_size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    return aligned


# ---------------------------------------------------------------------------
# Per-image processing
# ---------------------------------------------------------------------------

def process_image(
    img_path: Path,
    detector: "MTCNN",
    output_dir: Path,
    person_name: str,
) -> int:
    """
    Detect all faces in *img_path*, align each one, and save to *output_dir*.

    Parameters
    ----------
    img_path    : path to the raw input image.
    detector    : initialised MTCNN detector.
    output_dir  : root of the processed dataset.
    person_name : sub-folder name (typically the identity / class label).

    Returns
    -------
    Number of face crops successfully saved.
    """
    # -- Read image ----------------------------------------------------------
    image_bgr = cv2.imread(str(img_path))
    if image_bgr is None:
        log.warning("  [SKIP] Cannot read image: %s", img_path)
        return 0

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # -- Detect faces --------------------------------------------------------
    try:
        boxes, probs, landmarks = detector.detect(image_rgb, landmarks=True)
    except Exception as exc:
        log.warning("  [SKIP] Detection failed on %s – %s", img_path.name, exc)
        return 0

    if boxes is None or len(boxes) == 0:
        log.warning("  [SKIP] No face detected in: %s", img_path.name)
        return 0

    # -- Prepare output directory -------------------------------------------
    person_out = output_dir / person_name
    person_out.mkdir(parents=True, exist_ok=True)

    saved = 0
    stem = img_path.stem

    for idx, (box, prob, lmk) in enumerate(zip(boxes, probs, landmarks)):
        if prob < 0.90:
            log.debug("  [LOW CONF %.3f] skipping face %d in %s", prob, idx, img_path.name)
            continue

        # landmarks shape: (5, 2) – (x, y) for each of the 5 key-points
        lmk_pts = lmk.astype(np.float32)   # (5, 2)

        try:
            aligned = align_face(image_bgr, lmk_pts)
        except ValueError as exc:
            log.warning("  [SKIP] Alignment failed for face %d in %s – %s", idx, img_path.name, exc)
            continue

        # -- Save ------------------------------------------------------------
        suffix = f"_{idx}" if len(boxes) > 1 else ""
        out_name = f"{stem}{suffix}.jpg"
        out_path = person_out / out_name

        ok = cv2.imwrite(str(out_path), aligned, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if ok:
            log.debug("  Saved %s (conf=%.3f)", out_path, prob)
            saved += 1
        else:
            log.warning("  [ERROR] Could not write: %s", out_path)

    return saved


# ---------------------------------------------------------------------------
# Dataset traversal
# ---------------------------------------------------------------------------

def preprocess_dataset(
    input_dir: Path,
    output_dir: Path,
    detector: "MTCNN",
) -> None:
    """
    Walk *input_dir*, detect + align every face image, and write results to
    *output_dir* preserving the ``<person_name>/<filename>`` structure.

    Parameters
    ----------
    input_dir  : root raw dataset directory.
    output_dir : root processed dataset directory.
    detector   : initialised MTCNN detector.
    """
    if not input_dir.exists():
        log.error("Input directory does not exist: %s", input_dir)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    person_dirs = sorted([p for p in input_dir.iterdir() if p.is_dir()])
    if not person_dirs:
        log.error("No person sub-directories found under: %s", input_dir)
        sys.exit(1)

    total_images = 0
    total_faces = 0
    skipped = 0

    for person_dir in person_dirs:
        person_name = person_dir.name
        image_files = sorted(
            f for f in person_dir.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS
        )

        if not image_files:
            log.warning("[%s] No supported images found – skipping.", person_name)
            continue

        log.info("[%s] Processing %d image(s) …", person_name, len(image_files))
        person_saved = 0

        for img_path in image_files:
            n = process_image(img_path, detector, output_dir, person_name)
            person_saved += n
            if n == 0:
                skipped += 1
            total_images += 1

        log.info("[%s] Saved %d face crop(s).", person_name, person_saved)
        total_faces += person_saved

    log.info(
        "Done. %d image(s) processed | %d face crop(s) saved | %d image(s) skipped.",
        total_images,
        total_faces,
        skipped,
    )


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Face detection, landmark alignment, and cropping pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=Path("dataset/raw"),
        help="Root directory of raw face images (<input>/<person_name>/<img>).",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("dataset/processed"),
        help="Root directory for aligned face crops.",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default=None,
        help="Force CPU or CUDA. Defaults to CUDA if available.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    log.info("Input  directory : %s", args.input.resolve())
    log.info("Output directory : %s", args.output.resolve())

    detector = build_detector(device=args.device)
    preprocess_dataset(args.input, args.output, detector)


if __name__ == "__main__":
    main()

