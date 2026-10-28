# -*- coding: utf-8 -*-
"""
evaluate.py
-----------
Evaluate a trained ArcFace face recognition model.

Metrics reported
----------------
* Embedding extraction speed (images/sec, ms/image)
* Cosine-similarity distribution (genuine vs impostor pairs)
* ROC curve       AUC
* TAR @ FAR = {1e-5, 1e-4, 1e-3, 1e-2}
* Equal Error Rate (EER)
* 1:1 Verification accuracy at the EER threshold
* 1:N Identification Rank-1 / Rank-5 accuracy (closed-set)

Usage
-----
    python evaluate.py
    python evaluate.py --model best_arcface_model.pth --data dataset/processed
    python evaluate.py --pairs pairs.csv     # supply explicit pairs file
    python evaluate.py --save-roc roc.png    # save ROC curve image
"""

import argparse
import itertools
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Re-use backbone definitions from train_arcface
try:
    from train_arcface import MobileFaceNet, ResNet100, build_backbone
except ImportError:
    sys.exit(
        "[ERROR] Could not import train_arcface.py. "
        "Make sure it exists in the same directory."
    )

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(stream=__import__("sys").stdout)],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FAR_TARGETS: List[float] = [1e-5, 1e-4, 1e-3, 1e-2]   # TAR reported at each
EMBED_DIM = 512


# ===========================================================================
# 1.  MODEL LOADING
# ===========================================================================

def load_model(
    model_path: Path,
    device: torch.device,
) -> Tuple[torch.nn.Module, Dict[str, int]]:
    """
    Load backbone weights from a saved checkpoint.

    Parameters
    ----------
    model_path : path to ``best_arcface_model.pth``.
    device     : torch device.

    Returns
    -------
    (backbone_module, class_to_idx)
    """
    if not model_path.exists():
        log.error("Model file not found: %s", model_path)
        sys.exit(1)

    log.info("Loading model from: %s", model_path)
    ckpt = torch.load(str(model_path), map_location=device)

    backbone_name  = ckpt.get("backbone",      "mobilefacenet")
    embedding_dim  = ckpt.get("embedding_dim", EMBED_DIM)
    class_to_idx   = ckpt.get("class_to_idx",  {})

    backbone = build_backbone(backbone_name, embedding_dim=embedding_dim)
    backbone.load_state_dict(ckpt["backbone_state"])
    backbone.to(device).eval()

    num_classes = ckpt.get("num_classes", len(class_to_idx))
    log.info(
        "Loaded %s backbone | embedding_dim=%d | trained on %d class(es)",
        backbone_name, embedding_dim, num_classes,
    )
    return backbone, class_to_idx


# ===========================================================================
# 2.  EMBEDDING EXTRACTION
# ===========================================================================

def _val_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def extract_embeddings(
    data_dir: Path,
    backbone: torch.nn.Module,
    device: torch.device,
    batch_size: int = 64,
    num_workers: int = 4,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Forward-pass all images in *data_dir* through the backbone.

    Returns
    -------
    embeddings : (N, D) float32 L2-normalised embeddings.
    labels     : (N,)   int64  class indices.
    class_names: list of class name strings (index     name).
    """
    dataset = datasets.ImageFolder(str(data_dir), transform=_val_transform())
    if len(dataset) == 0:
        log.error("No images found in: %s", data_dir)
        sys.exit(1)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    log.info(
        "Extracting embeddings: %d images / %d classes    ",
        len(dataset), len(dataset.classes),
    )

    all_embeddings: List[np.ndarray] = []
    all_labels:     List[np.ndarray] = []

    t_start = time.perf_counter()

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            embs = backbone(images)                         # already L2-normed
            all_embeddings.append(embs.cpu().numpy())
            all_labels.append(labels.numpy())

    t_end = time.perf_counter()

    embeddings  = np.concatenate(all_embeddings, axis=0).astype(np.float32)
    labels      = np.concatenate(all_labels,     axis=0).astype(np.int64)
    class_names = dataset.classes

    elapsed      = t_end - t_start
    imgs_per_sec = len(dataset) / elapsed
    ms_per_img   = elapsed / len(dataset) * 1000.0

    log.info(
        "Extraction complete: %.1f img/s  |  %.2f ms/img  |  %.2fs total",
        imgs_per_sec, ms_per_img, elapsed,
    )

    return embeddings, labels, class_names, {
        "imgs_per_sec": imgs_per_sec,
        "ms_per_img":   ms_per_img,
        "total_sec":    elapsed,
        "n_images":     len(dataset),
    }


# ===========================================================================
# 3.  PAIR GENERATION
# ===========================================================================

def _build_pairs_from_embeddings(
    embeddings: np.ndarray,
    labels: np.ndarray,
    max_pairs_per_class: int = 500,
    max_impostor_pairs:  int = 100_000,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Automatically generate genuine (same-identity) and impostor (different-identity)
    pairs from the extracted embeddings.

    Returns
    -------
    sims   : (P,)   cosine similarity for each pair (embeddings already L2-normed,
                    so dot product = cosine similarity).
    is_genuine : (P,) bool   True      same identity.
    pair_labels : (P, 2) int  index of each element in the pair.
    """
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)

    genuine_idx:  List[Tuple[int, int]] = []
    impostor_idx: List[Tuple[int, int]] = []

    # --- Genuine pairs (intra-class) --------------------------------------
    for cls in classes:
        idxs = np.where(labels == cls)[0]
        if len(idxs) < 2:
            continue
        pairs = list(itertools.combinations(idxs.tolist(), 2))
        if len(pairs) > max_pairs_per_class:
            chosen = rng.choice(len(pairs), max_pairs_per_class, replace=False)
            pairs = [pairs[i] for i in chosen]
        genuine_idx.extend(pairs)

    # --- Impostor pairs (inter-class) -------------------------------------
    all_combos = list(itertools.combinations(classes.tolist(), 2))
    rng.shuffle(all_combos)

    per_combo = max(1, max_impostor_pairs // max(len(all_combos), 1))

    for cls_a, cls_b in all_combos:
        idxs_a = np.where(labels == cls_a)[0]
        idxs_b = np.where(labels == cls_b)[0]
        na = min(len(idxs_a), per_combo)
        nb = min(len(idxs_b), per_combo)
        sel_a = rng.choice(idxs_a, na, replace=False)
        sel_b = rng.choice(idxs_b, nb, replace=False)
        impostor_idx.extend(zip(sel_a.tolist(), sel_b.tolist()))
        if len(impostor_idx) >= max_impostor_pairs:
            impostor_idx = impostor_idx[:max_impostor_pairs]
            break

    if not genuine_idx:
        log.error(
            "No genuine pairs could be formed. Each class needs    2 images.\n"
            "  Tip: add more images per identity or use --pairs to supply an explicit file."
        )
        sys.exit(1)

    log.info(
        "Pairs: %d genuine  |  %d impostor",
        len(genuine_idx), len(impostor_idx),
    )

    all_pairs  = np.array(genuine_idx + impostor_idx, dtype=np.int64)
    is_genuine = np.array(
        [True] * len(genuine_idx) + [False] * len(impostor_idx), dtype=bool
    )

    # Cosine similarity (dot product of L2-normed vectors)
    e1 = embeddings[all_pairs[:, 0]]
    e2 = embeddings[all_pairs[:, 1]]
    sims = (e1 * e2).sum(axis=1)

    return sims, is_genuine, all_pairs


def _load_pairs_csv(
    csv_path: Path,
    embeddings: np.ndarray,
    img_paths: List[Path],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load pairs from a CSV with columns: path1, path2, label (1=genuine, 0=impostor).
    Returns (sims, is_genuine).
    """
    import csv

    path_to_idx = {str(p): i for i, p in enumerate(img_paths)}
    sims, labels_out = [], []

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            i1 = path_to_idx.get(row["path1"])
            i2 = path_to_idx.get(row["path2"])
            if i1 is None or i2 is None:
                continue
            sim = float((embeddings[i1] * embeddings[i2]).sum())
            sims.append(sim)
            labels_out.append(int(row["label"]))

    return np.array(sims, dtype=np.float32), np.array(labels_out, dtype=bool)


# ===========================================================================
# 4.  METRICS
# ===========================================================================

def compute_roc(
    sims: np.ndarray,
    is_genuine: np.ndarray,
    n_thresholds: int = 10_000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute ROC curve (FAR vs TAR) by sweeping similarity thresholds.

    Returns
    -------
    thresholds : (T,)  cosine-similarity thresholds (descending).
    far        : (T,)  False Acceptance Rate at each threshold.
    tar        : (T,)  True  Acceptance Rate at each threshold.
    """
    thresholds = np.linspace(sims.max(), sims.min(), n_thresholds)

    n_genuine  = is_genuine.sum()
    n_impostor = (~is_genuine).sum()

    tar_list, far_list = [], []

    for thr in thresholds:
        accepted   = sims >= thr
        ta = (accepted  &  is_genuine).sum()
        fa = (accepted  & ~is_genuine).sum()
        tar_list.append(ta / max(n_genuine,  1))
        far_list.append(fa / max(n_impostor, 1))

    return thresholds, np.array(far_list), np.array(tar_list)


def tar_at_far(
    far: np.ndarray,
    tar: np.ndarray,
    target_far: float,
) -> Tuple[float, float]:
    """
    Interpolate TAR at a specific FAR target.

    Returns
    -------
    (tar_value, actual_far_achieved)
    """
    if target_far < far.min():
        return 0.0, far.min()
    idx = np.searchsorted(far, target_far, side="left")
    idx = np.clip(idx, 0, len(far) - 1)
    return float(tar[idx]), float(far[idx])


def compute_eer(
    far: np.ndarray,
    tar: np.ndarray,
    thresholds: np.ndarray,
) -> Tuple[float, float]:
    """
    Equal Error Rate: the operating point where FAR     FRR = 1     TAR.

    Returns
    -------
    (eer, eer_threshold)
    """
    frr = 1.0 - tar
    diff = np.abs(far - frr)
    idx  = diff.argmin()
    eer  = float((far[idx] + frr[idx]) / 2.0)
    return eer, float(thresholds[idx])


def compute_auc(far: np.ndarray, tar: np.ndarray) -> float:
    """Trapezoidal AUC of the ROC curve."""
    # Sort by ascending FAR
    order = np.argsort(far)
    trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
    return float(trapz_fn(tar[order], far[order]))


def rank_n_accuracy(
    embeddings: np.ndarray,
    labels: np.ndarray,
    ranks: Tuple[int, ...] = (1, 5),
    batch_size: int = 256,
) -> Dict[int, float]:
    """
    Closed-set 1:N identification accuracy at specified ranks.
    For each probe, the gallery contains all other images.

    Parameters
    ----------
    embeddings : (N, D) L2-normalised.
    labels     : (N,)   class indices.
    ranks      : which rank positions to evaluate (e.g. 1, 5).
    batch_size : number of probes processed per matrix multiply.

    Returns
    -------
    {rank: accuracy}
    """
    n = len(embeddings)
    rank_correct = {r: 0 for r in ranks}

    for start in range(0, n, batch_size):
        end   = min(start + batch_size, n)
        probe = embeddings[start:end]                   # (B, D)

        # Cosine similarity to all gallery items
        sim_matrix = probe @ embeddings.T               # (B, N)

        # Exclude self-similarity (probe vs itself)
        for local_i, global_i in enumerate(range(start, end)):
            sim_matrix[local_i, global_i] = -2.0

        # Sort descending
        sorted_idxs = np.argsort(-sim_matrix, axis=1)  # (B, N)
        probe_labels = labels[start:end]

        for r in ranks:
            top_r = sorted_idxs[:, :r]                 # (B, r)
            top_r_labels = labels[top_r]               # (B, r)
            match = (top_r_labels == probe_labels[:, None]).any(axis=1)
            rank_correct[r] += match.sum()

    return {r: rank_correct[r] / n for r in ranks}


def verification_accuracy(
    sims: np.ndarray,
    is_genuine: np.ndarray,
    threshold: float,
) -> float:
    """Binary verification accuracy at a fixed cosine-similarity threshold."""
    predictions = sims >= threshold
    return float((predictions == is_genuine).mean())


# ===========================================================================
# 5.  REPORT
# ===========================================================================

def print_report(
    speed_stats:    Dict,
    auc:            float,
    eer:            float,
    eer_threshold:  float,
    tar_far_pairs:  Dict[float, Tuple[float, float]],
    verif_acc:      float,
    rank_accs:      Dict[int, float],
    n_genuine:      int,
    n_impostor:     int,
) -> None:
    """Print a structured evaluation summary to stdout."""
    w = 60
    sep = "   " * w

    print()
    print("   " * w)
    print("  ArcFace Model Evaluation Report")
    print("   " * w)

    print("\n[*]  Dataset Statistics")
    print(sep)
    print(f"  Genuine  pairs  : {n_genuine:>10,}")
    print(f"  Impostor pairs  : {n_impostor:>10,}")

    print("\n[!]  Inference Speed")
    print(sep)
    print(f"  Images processed  : {speed_stats['n_images']:>10,}")
    print(f"  Throughput        : {speed_stats['imgs_per_sec']:>10.1f}  img/s")
    print(f"  Latency           : {speed_stats['ms_per_img']:>10.2f}  ms/img")
    print(f"  Total time        : {speed_stats['total_sec']:>10.2f}  s")

    print("\n[~]  ROC / Verification Metrics")
    print(sep)
    print(f"  AUC               : {auc:>10.6f}")
    print(f"  EER               : {eer * 100:>9.4f} %  (threshold = {eer_threshold:.4f})")
    print(f"  Verif. Acc (EER)  : {verif_acc * 100:>9.4f} %")

    print("\n[>]  TAR @ FAR Targets")
    print(sep)
    for far_target, (tar_val, far_achieved) in sorted(tar_far_pairs.items()):
        print(
            f"  TAR @ FAR     1e{int(round(np.log10(far_target))):+d}  :  "
            f"{tar_val * 100:7.4f} %   (actual FAR = {far_achieved:.2e})"
        )

    print("\n[#]  Closed-Set Identification (1:N)")
    print(sep)
    for rank, acc in sorted(rank_accs.items()):
        print(f"  Rank-{rank:<2} Accuracy  : {acc * 100:>9.4f} %")

    print("\n" + "   " * w + "\n")


# ===========================================================================
# 6.  ROC PLOT  (optional, requires matplotlib)
# ===========================================================================

def save_roc_plot(
    far: np.ndarray,
    tar: np.ndarray,
    auc: float,
    eer: float,
    output_path: Path,
    tar_far_pairs: Optional[Dict] = None,
) -> None:
    """Save a ROC curve PNG (requires matplotlib)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not installed     skipping ROC plot. pip install matplotlib")
        return

    order = np.argsort(far)
    x, y  = far[order], tar[order]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.semilogx(x, y, lw=2, color="#2563eb", label=f"ROC  (AUC = {auc:.6f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4)

    # Mark EER
    ax.scatter([eer], [1 - eer], marker="x", color="crimson", zorder=5,
               s=80, label=f"EER = {eer * 100:.3f}%")

    # Mark TAR @ FAR targets
    if tar_far_pairs:
        colors = ["#d97706", "#059669", "#7c3aed", "#db2777"]
        for (far_t, (tar_v, _)), col in zip(
            sorted(tar_far_pairs.items()), colors
        ):
            ax.scatter(
                [far_t], [tar_v],
                marker="o", color=col, zorder=5, s=60,
                label=f"TAR={tar_v * 100:.2f}% @ FAR=1e{int(round(np.log10(far_t))):+d}",
            )

    ax.set_xlabel("False Acceptance Rate (FAR)", fontsize=12)
    ax.set_ylabel("True Acceptance Rate (TAR)", fontsize=12)
    ax.set_title("ROC Curve     ArcFace Face Recognition", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.set_xlim([1e-6, 1.0])
    ax.set_ylim([0.0,  1.01])
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150)
    plt.close()
    log.info("ROC plot saved     %s", output_path.resolve())


# ===========================================================================
# 7.  MAIN
# ===========================================================================

def evaluate(args: argparse.Namespace) -> None:

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    if args.device == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA not available     falling back to CPU.")
        args.device = "cpu"
    device = torch.device(args.device)
    log.info("Device: %s", device)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    backbone, _ = load_model(args.model, device)

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------
    embeddings, labels, class_names, speed_stats = extract_embeddings(
        data_dir=args.data,
        backbone=backbone,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # ------------------------------------------------------------------
    # Pairs
    # ------------------------------------------------------------------
    if args.pairs and Path(args.pairs).exists():
        log.info("Loading pairs from CSV: %s", args.pairs)
        # Reconstruct image paths list from the dataset
        dataset = datasets.ImageFolder(str(args.data), transform=None)
        img_paths = [Path(s[0]) for s in dataset.samples]
        sims, is_genuine = _load_pairs_csv(Path(args.pairs), embeddings, img_paths)
    else:
        sims, is_genuine, _ = _build_pairs_from_embeddings(
            embeddings, labels,
            max_pairs_per_class=args.max_genuine,
            max_impostor_pairs=args.max_impostor,
        )

    n_genuine  = int(is_genuine.sum())
    n_impostor = int((~is_genuine).sum())

    # ------------------------------------------------------------------
    # ROC
    # ------------------------------------------------------------------
    log.info("Computing ROC curve (threshold sweep)    ")
    thresholds, far, tar = compute_roc(sims, is_genuine, n_thresholds=20_000)

    auc = compute_auc(far, tar)
    eer, eer_thr = compute_eer(far, tar, thresholds)

    # ------------------------------------------------------------------
    # TAR @ FAR targets
    # ------------------------------------------------------------------
    tar_far_pairs: Dict[float, Tuple[float, float]] = {}
    for far_target in FAR_TARGETS:
        tar_far_pairs[far_target] = tar_at_far(far, tar, far_target)

    # ------------------------------------------------------------------
    # Verification accuracy at EER threshold
    # ------------------------------------------------------------------
    verif_acc = verification_accuracy(sims, is_genuine, threshold=eer_thr)

    # ------------------------------------------------------------------
    # Rank-1 / Rank-5 identification
    # ------------------------------------------------------------------
    log.info("Computing closed-set identification accuracy (Rank-1 / Rank-5)    ")
    rank_accs = rank_n_accuracy(embeddings, labels, ranks=(1, 5))

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    print_report(
        speed_stats=speed_stats,
        auc=auc,
        eer=eer,
        eer_threshold=eer_thr,
        tar_far_pairs=tar_far_pairs,
        verif_acc=verif_acc,
        rank_accs=rank_accs,
        n_genuine=n_genuine,
        n_impostor=n_impostor,
    )

    # ------------------------------------------------------------------
    # Optional: save ROC plot
    # ------------------------------------------------------------------
    if args.save_roc:
        save_roc_plot(
            far, tar, auc, eer,
            output_path=Path(args.save_roc),
            tar_far_pairs=tar_far_pairs,
        )

    # ------------------------------------------------------------------
    # Optional: save similarity scores to CSV for further analysis
    # ------------------------------------------------------------------
    if args.save_scores:
        import csv
        out_csv = Path(args.save_scores)
        with open(out_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["similarity", "is_genuine"])
            for s, g in zip(sims.tolist(), is_genuine.tolist()):
                writer.writerow([f"{s:.6f}", int(g)])
        log.info("Similarity scores saved     %s", out_csv.resolve())


# ===========================================================================
# 8.  CLI
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate a trained ArcFace face recognition model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Paths
    p.add_argument("--model",  type=Path, default=Path("best_arcface_model.pth"),
                   help="Path to trained model weights.")
    p.add_argument("--data",   type=Path, default=Path("dataset/processed"),
                   help="Root of the face image dataset (ImageFolder structure).")
    p.add_argument("--pairs",  type=str,  default=None,
                   help="Optional CSV file with (path1, path2, label) columns.")

    # Pair generation
    p.add_argument("--max-genuine",  type=int, default=500,
                   help="Max genuine pairs per identity class.")
    p.add_argument("--max-impostor", type=int, default=100_000,
                   help="Max total impostor pairs.")

    # Outputs
    p.add_argument("--save-roc",    type=str, default=None,
                   help="If set, save the ROC curve as a PNG at this path.")
    p.add_argument("--save-scores", type=str, default=None,
                   help="If set, save similarity scores to this CSV path.")

    # Hardware
    p.add_argument("--device",      choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--batch-size",  type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)

    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())


