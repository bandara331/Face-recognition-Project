"""
export_model.py
---------------
Export trained PyTorch ArcFace backbone to production-ready formats:
  1. TorchScript (.pt) - Standalone optimized model (ready for FastAPI / C++)
  2. ONNX (.onnx)       - Cross-platform optimized runtime model

Usage:
  python export_model.py
"""

import argparse
import logging
from pathlib import Path
import time
import torch
import torch.nn.functional as F

from train_arcface import build_backbone

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger(__name__)


def load_trained_model(weights_path: Path, device: torch.device):
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path.resolve()}")

    log.info("Loading weights from: %s", weights_path.name)
    ckpt = torch.load(str(weights_path), map_location=device)
    backbone_name = ckpt.get("backbone", "mobilefacenet")
    embedding_dim = ckpt.get("embedding_dim", 512)

    model = build_backbone(backbone_name, embedding_dim=embedding_dim)
    model.load_state_dict(ckpt["backbone_state"])
    model.to(device).eval()
    return model, backbone_name, embedding_dim


def export_torchscript(model, dummy_input: torch.Tensor, output_path: Path):
    """Export model to TorchScript (.pt)"""
    log.info("Exporting to TorchScript: %s ...", output_path.name)
    with torch.no_grad():
        traced = torch.jit.trace(model, dummy_input)
        traced.save(str(output_path))
    size_mb = output_path.stat().st_size / (1024 * 1024)
    log.info("  Successfully exported TorchScript (%.2f MB) -> %s", size_mb, output_path.name)
    return traced


def export_onnx(model, dummy_input: torch.Tensor, output_path: Path):
    """Export model to ONNX (.onnx) if onnx library is available"""
    try:
        import onnx
        log.info("Exporting to ONNX: %s ...", output_path.name)
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            input_names=["input"],
            output_names=["embedding"],
            dynamic_axes={"input": {0: "batch_size"}, "embedding": {0: "batch_size"}},
            opset_version=14,
            dynamo=False,
        )
        size_mb = output_path.stat().st_size / (1024 * 1024)
        log.info("  Successfully exported ONNX model (%.2f MB) -> %s", size_mb, output_path.name)
    except ImportError:
        log.info("ℹ️ Note: 'onnx' library not installed. Skipped ONNX export (TorchScript is ready).")
    except Exception as e:
        log.warning("Could not export ONNX: %s", e)


def benchmark_speed(model, dummy_input: torch.Tensor, iterations: int = 50):
    """Benchmark inference latency (ms per face)."""
    # Warmup
    for _ in range(5):
        _ = model(dummy_input)

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(dummy_input)
    total_time = time.perf_counter() - start
    avg_latency_ms = (total_time / iterations) * 1000.0
    throughput = iterations / total_time
    return avg_latency_ms, throughput


def main():
    parser = argparse.ArgumentParser(description="Export ArcFace model to production formats.")
    parser.add_argument("--weights", type=Path, default=Path("best_arcface_model.pth"))
    parser.add_argument("--out-dir", type=Path, default=Path("."))
    args = parser.parse_args()

    device = torch.device("cpu")
    model, backbone_name, embedding_dim = load_trained_model(args.weights, device)

    # Standard face input: (batch_size=1, channels=3, height=112, width=112)
    dummy_input = torch.randn(1, 3, 112, 112, device=device)

    # 1. Export TorchScript
    ts_path = args.out_dir / "best_arcface_model.pt"
    ts_model = export_torchscript(model, dummy_input, ts_path)

    # 2. Export ONNX (Optional / if onnx installed)
    onnx_path = args.out_dir / "best_arcface_model.onnx"
    export_onnx(model, dummy_input, onnx_path)

    # 3. Benchmark Inference Speed
    log.info("Benchmarking inference latency on CPU (50 iterations) ...")
    latency_ms, throughput = benchmark_speed(ts_model, dummy_input)
    log.info("  Average Latency : %.2f ms / image", latency_ms)
    log.info("  Throughput      : %.1f images / sec", throughput)
    log.info(" Export complete! Handover files are ready for Member 2 & MLOps.")


if __name__ == "__main__":
    main()