"""
inference.py
------------
Face Recognition Inference Pipeline for FastAPI Backend (Member 2).
Handles:
  1. Face Detection & 5-point Alignment (MTCNN)
  2. 512-d Embedding Extraction (MobileFaceNet + ArcFace)
  3. Cosine Similarity & Face Verification (1:1 matching)
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Re-use preprocessing & model components
from preprocess import align_face, build_detector
from train_arcface import build_backbone

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger(__name__)

DEFAULT_THRESHOLD = 0.65  # Recommended Cosine Similarity threshold


class FaceRecognitionPipeline:
    """
    Production-ready Face Recognition inference class.
    Can be directly imported into FastAPI endpoints.
    """

    def __init__(
        self,
        model_path: str = "best_arcface_model.pth",
        device: Optional[str] = None,
        threshold: float = DEFAULT_THRESHOLD,
    ):
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.threshold = threshold
        log.info("Initializing FaceRecognitionPipeline on device: %s", self.device)

        # 1. Initialize Face Detector
        self.detector = build_detector(device=str(self.device))

        # 2. Load Model Weights
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path.resolve()}")

        ckpt = torch.load(str(self.model_path), map_location=self.device)
        backbone_name = ckpt.get("backbone", "mobilefacenet")
        embedding_dim = ckpt.get("embedding_dim", 512)

        self.backbone = build_backbone(backbone_name, embedding_dim=embedding_dim)
        self.backbone.load_state_dict(ckpt["backbone_state"])
        self.backbone.to(self.device).eval()
        log.info("Loaded %s model from %s", backbone_name, self.model_path.name)

    def _load_image(self, image_input: Union[str, Path, np.ndarray, bytes, Image.Image]) -> np.ndarray:
        """Helper to convert any input into a BGR uint8 numpy array."""
        if isinstance(image_input, (str, Path)):
            img = cv2.imread(str(image_input))
            if img is None:
                raise ValueError(f"Could not read image from path: {image_input}")
            return img
        elif isinstance(image_input, bytes):
            nparr = np.frombuffer(image_input, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Could not decode image bytes")
            return img
        elif isinstance(image_input, Image.Image):
            return cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
        elif isinstance(image_input, np.ndarray):
            return image_input
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

    def extract_aligned_face(self, image_input) -> Optional[np.ndarray]:
        """Detect and align a single face from an input image to 112x112."""
        image_bgr = self._load_image(image_input)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        boxes, probs, landmarks = self.detector.detect(image_rgb, landmarks=True)
        if boxes is None or len(boxes) == 0:
            log.warning("No face detected in image.")
            return None

        # Choose the face with the highest confidence
        best_idx = int(np.argmax(probs))
        lmk_pts = landmarks[best_idx].astype(np.float32)

        aligned_bgr = align_face(image_bgr, lmk_pts)
        return aligned_bgr

    @torch.no_grad()
    def get_embedding(self, image_input) -> Optional[np.ndarray]:
        """
        Extract a 512-d L2-normalized face embedding.
        Returns:
            np.ndarray of shape (512,) or None if no face is detected.
        """
        aligned_bgr = self.extract_aligned_face(image_input)
        if aligned_bgr is None:
            return None

        aligned_rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)
        # Preprocessing: [0, 255] -> [-1, 1], (H, W, C) -> (1, C, H, W)
        tensor = torch.from_numpy(aligned_rgb).float() / 127.5 - 1.0
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)

        embedding = self.backbone(tensor)
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding.cpu().numpy().flatten()

    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute Cosine Similarity between two 512-d embeddings."""
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))

    def verify(self, image1, image2, threshold: Optional[float] = None) -> Dict:
        """
        1:1 Face Verification between two images.
        """
        th = threshold if threshold is not None else self.threshold
        emb1 = self.get_embedding(image1)
        emb2 = self.get_embedding(image2)

        if emb1 is None or emb2 is None:
            return {
                "verified": False,
                "similarity": 0.0,
                "threshold": th,
                "error": "Face detection failed on one or both images."
            }

        sim = self.compute_similarity(emb1, emb2)
        return {
            "verified": bool(sim >= th),
            "similarity": round(sim, 4),
            "threshold": th,
            "error": None
        }


# ---------------------------------------------------------------------------
# CLI Test Runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Face Recognition Inference.")
    parser.add_argument("--image", type=str, help="Extract 512-d embedding for a single image.")
    parser.add_argument("--compare", nargs=2, help="Compare two images and verify identity.")
    args = parser.parse_args()

    pipeline = FaceRecognitionPipeline()

    if args.image:
        emb = pipeline.get_embedding(args.image)
        if emb is not None:
            print(f"\n Successfully extracted 512-d embedding!")
            print(f"Shape: {emb.shape} | First 5 values: {np.round(emb[:5], 4)}")
        else:
            print("\n Face not detected.")

    elif args.compare:
        result = pipeline.verify(args.compare[0], args.compare[1])
        print("\n--- Verification Result ---")
        print(f"Image 1    : {args.compare[0]}")
        print(f"Image 2    : {args.compare[1]}")
        print(f"Similarity : {result['similarity']}")
        print(f"Threshold  : {result['threshold']}")
        print(f"Match?     : {' YES' if result['verified'] else ' NO'}")
        if result['error']:
            print(f"Error      : {result['error']}")