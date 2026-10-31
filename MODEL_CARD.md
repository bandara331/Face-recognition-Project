# Model Card: ArcFace MobileFaceNet 512-d

## 1. Overview
- **Role:** Member 1 - AI/ML Engineer (Model & Data Setup)
- **Model Architecture:** MobileFaceNet Backbone with ArcFace Loss
- **Output Embedding Size:** 512-dimensional normalized float vector
- **Input Image Size:** 112x112 RGB (Aligned with 5-point facial landmarks)

## 2. Performance Metrics
- **Inference Latency:** 12.47 ms / image (CPU)
- **Throughput:** ~80 images / second
- **Model File Size:** 5.00 MB (`best_arcface_model.pt`)
- **Recommended Cosine Similarity Threshold:** `0.65`

## 3. How to Use for Member 2 (FastAPI Backend)
```python
from inference import FaceRecognitionPipeline

# Initialize once at app startup
pipeline = FaceRecognitionPipeline(model_path="best_arcface_model.pth")

# 1. In /register endpoint:
embedding = pipeline.get_embedding(uploaded_image_bytes)
# Store embedding (512-d float array) in Milvus / Database

# 2. In /verify endpoint:
result = pipeline.verify(image1_bytes, image2_bytes, threshold=0.65)
# Returns: {"verified": True/False, "similarity": float, "threshold": 0.65}
```

## 4. Deliverable Files
- `preprocess.py`: MTCNN 5-point alignment & 112x112 cropping.
- `train_arcface.py`: ArcFace training pipeline.
- `evaluate.py`: TAR@FAR and ROC evaluation script.
- `inference.py`: Production-ready inference helper class.
- `export_model.py`: Standalone TorchScript model exporter.
- `best_arcface_model.pt`: Standalone production model (5 MB).