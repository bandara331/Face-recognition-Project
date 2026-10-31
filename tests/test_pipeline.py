"""
tests/test_pipeline.py
----------------------
Unit and Integration tests for the Face Recognition AI/ML pipeline.
Designed for GitHub Actions CI/CD pipeline automation.
"""

from pathlib import Path
import numpy as np
import pytest
import torch

from inference import FaceRecognitionPipeline


@pytest.fixture(scope="module")
def pipeline():
    """Shared pipeline instance to avoid reloading detector/model per test."""
    return FaceRecognitionPipeline(model_path="best_arcface_model.pth")


@pytest.fixture(scope="module")
def sample_image_path():
    """Path to a verified test image in the local dataset."""
    p = Path("dataset/raw/person_1/image1.jpg")
    if not p.exists():
        pytest.skip(f"Sample test image not found at: {p}")
    return str(p)


def test_model_files_exist():
    """Ensure key model checkpoint and standalone TorchScript model exist."""
    pth_file = Path("best_arcface_model.pth")
    pt_file = Path("best_arcface_model.pt")

    assert pth_file.exists(), "best_arcface_model.pth is missing!"
    assert pt_file.exists(), "best_arcface_model.pt is missing!"
    assert pth_file.stat().st_size > 1_000_000, "Checkpoint file is too small or corrupted"
    assert pt_file.stat().st_size > 1_000_000, "TorchScript model is too small or corrupted"


def test_pipeline_initialization(pipeline):
    """Verify that detector and backbone are correctly initialized."""
    assert pipeline.detector is not None, "MTCNN face detector was not initialized"
    assert pipeline.backbone is not None, "Backbone network was not loaded"
    assert pipeline.threshold == 0.65, "Default verification threshold should be 0.65"


def test_embedding_extraction_shape_and_norm(pipeline, sample_image_path):
    """Check that an aligned face produces a 512-d unit-normalized embedding."""
    emb = pipeline.get_embedding(sample_image_path)
    assert emb is not None, "Face detection failed on the sample image"
    assert isinstance(emb, np.ndarray), "Embedding must be a numpy array"
    assert emb.shape == (512,), f"Expected shape (512,), but got {emb.shape}"

    # L2 Norm of normalized embedding must be approximately 1.0
    norm = np.linalg.norm(emb)
    assert np.isclose(norm, 1.0, atol=1e-3), f"Expected unit norm ~1.0, got {norm}"


def test_cosine_similarity_logic(pipeline):
    """Verify mathematical properties of cosine similarity."""
    v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v2 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v3 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    v4 = np.array([-1.0, 0.0, 0.0], dtype=np.float32)

    # Identical vectors -> similarity = 1.0
    assert np.isclose(pipeline.compute_similarity(v1, v2), 1.0)
    # Orthogonal vectors -> similarity = 0.0
    assert np.isclose(pipeline.compute_similarity(v1, v3), 0.0)
    # Opposite vectors -> similarity = -1.0
    assert np.isclose(pipeline.compute_similarity(v1, v4), -1.0)


def test_self_verification(pipeline, sample_image_path):
    """Comparing an image against itself should yield a match (verified=True)."""
    result = pipeline.verify(sample_image_path, sample_image_path, threshold=0.65)

    assert result["error"] is None, f"Verification returned an error: {result['error']}"
    assert result["verified"] is True, "Self-verification should always be True"
    assert result["similarity"] >= 0.99, f"Self-similarity should be ~1.0, got {result['similarity']}"
    assert result["threshold"] == 0.65
