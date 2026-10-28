"""
train_arcface.py
----------------
Train a face recognition backbone (MobileFaceNet or ResNet-100) with the
ArcFace (Additive Angular Margin Loss) objective.

Directory structure expected:
    dataset/processed/<person_name>/<aligned_face.jpg>

Outputs:
    best_arcface_model.pth   – best backbone weights (lowest val loss)
    arcface_checkpoint.pth   – latest checkpoint (resume training)

Usage:
    python train_arcface.py [options]
    python train_arcface.py --backbone resnet100 --epochs 50 --batch-size 64
    python train_arcface.py --resume arcface_checkpoint.pth
"""

import argparse
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.models import resnet50

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ===========================================================================
# 1.  DATA
# ===========================================================================

def build_transforms(augment: bool = True) -> transforms.Compose:
    """
    Return a torchvision transform pipeline for 112×112 face images.

    Parameters
    ----------
    augment : Apply data augmentation (train split).
    """
    normalize = transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
    )
    if augment:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            transforms.ToTensor(),
            normalize,
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])


def build_dataloaders(
    data_dir: Path,
    val_split: float = 0.1,
    batch_size: int = 64,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, int]:
    """
    Build train and validation DataLoaders from an ImageFolder dataset.

    Returns
    -------
    (train_loader, val_loader, num_classes)
    """
    if not data_dir.exists():
        log.error("Processed dataset directory not found: %s", data_dir)
        sys.exit(1)

    # Full dataset without augmentation to allow manual split
    full_dataset = datasets.ImageFolder(str(data_dir), transform=None)
    num_classes = len(full_dataset.classes)
    log.info("Dataset: %d images across %d identities.", len(full_dataset), num_classes)

    if num_classes < 2:
        log.error("Need at least 2 identity classes to train; found %d.", num_classes)
        sys.exit(1)

    # Train / val split
    n_val = max(1, int(len(full_dataset) * val_split))
    n_train = len(full_dataset) - n_val
    train_subset, val_subset = random_split(
        full_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    # Attach per-split transforms
    class _TransformWrapper(torch.utils.data.Dataset):
        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform

        def __len__(self):
            return len(self.subset)

        def __getitem__(self, idx):
            img, label = self.subset[idx]
            if self.transform:
                img = self.transform(img)
            return img, label

    train_ds = _TransformWrapper(train_subset, build_transforms(augment=True))
    val_ds   = _TransformWrapper(val_subset,   build_transforms(augment=False))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,         # keeps batch sizes uniform for ArcFace
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    log.info(
        "Split → train: %d samples | val: %d samples | batch_size: %d",
        n_train, n_val, batch_size,
    )
    return train_loader, val_loader, num_classes


# ===========================================================================
# 2.  BACKBONE NETWORKS
# ===========================================================================

# ---------------------------------------------------------------------------
# 2a.  MobileFaceNet  (lightweight backbone)
# ---------------------------------------------------------------------------

class _DepthwiseSepConv(nn.Module):
    """Depthwise-separable convolution block."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.PReLU(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.pw(self.dw(x))))


class _InvertedResidual(nn.Module):
    """MobileNetV2-style inverted residual block."""

    def __init__(self, in_ch: int, out_ch: int, stride: int, expand: int):
        super().__init__()
        mid = in_ch * expand
        layers = []
        if expand != 1:
            layers += [
                nn.Conv2d(in_ch, mid, 1, bias=False),
                nn.BatchNorm2d(mid),
                nn.PReLU(mid),
            ]
        layers += [
            nn.Conv2d(mid, mid, 3, stride=stride, padding=1, groups=mid, bias=False),
            nn.BatchNorm2d(mid),
            nn.PReLU(mid),
            nn.Conv2d(mid, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        ]
        self.block = nn.Sequential(*layers)
        self.use_skip = (stride == 1 and in_ch == out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        return out + x if self.use_skip else out


class MobileFaceNet(nn.Module):
    """
    MobileFaceNet – a lightweight backbone designed for on-device face recognition.
    Produces a 512-d L2-normalised embedding from a 112×112 input.

    Reference: Chen et al., "MobileFaceNets", CCBR 2018.
    """

    def __init__(self, embedding_dim: int = 512):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Entry flow
        self.entry = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64),
            _DepthwiseSepConv(64, 64),
        )

        # Bottleneck layers: (expand, out_ch, repeats, stride)
        cfg = [
            (2,  64, 5, 2),
            (4, 128, 1, 2),
            (2, 128, 6, 1),
            (4, 128, 1, 2),
            (2, 128, 2, 1),
        ]
        layers = []
        in_ch = 64
        for expand, out_ch, repeats, stride in cfg:
            for i in range(repeats):
                layers.append(_InvertedResidual(in_ch, out_ch, stride if i == 0 else 1, expand))
                in_ch = out_ch
        self.bottlenecks = nn.Sequential(*layers)

        # Final conv → GDC → embedding
        self.conv_final = nn.Sequential(
            nn.Conv2d(128, 512, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.PReLU(512),
        )
        # Global Depthwise Convolution: kernel = spatial size after stride chain
        # 112 → 56 → 28 → 14 → 7
        self.gdc = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=7, groups=512, bias=False),
            nn.BatchNorm2d(512),
            nn.Flatten(),
        )
        self.fc = nn.Linear(512, embedding_dim, bias=False)
        self.bn_out = nn.BatchNorm1d(embedding_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return L2-normalised 512-d embedding."""
        x = self.entry(x)
        x = self.bottlenecks(x)
        x = self.conv_final(x)
        x = self.gdc(x)
        x = self.bn_out(self.fc(x))
        return F.normalize(x, p=2, dim=1)


# ---------------------------------------------------------------------------
# 2b.  ResNet-100 backbone  (high-accuracy backbone)
# ---------------------------------------------------------------------------

class _IResBlock(nn.Module):
    """Identity residual block used in iResNet (IR-SE variant without SE)."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.bn0  = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.prelu = nn.PReLU(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)

        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x) if self.downsample else x
        out = self.bn0(x)
        out = self.prelu(self.bn1(self.conv1(out)))
        out = self.bn2(self.conv2(out))
        return out + identity


class ResNet100(nn.Module):
    """
    iResNet-100 backbone for face recognition.
    Produces a 512-d L2-normalised embedding from a 112×112 input.

    Reference: Deng et al., "ArcFace", CVPR 2019.
    """

    def __init__(self, embedding_dim: int = 512):
        super().__init__()
        self.embedding_dim = embedding_dim

        self.input_layer = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64),
        )

        # (out_ch, num_blocks, stride)
        layers_cfg = [
            ( 64,  3, 2),
            (128, 13, 2),
            (256, 30, 2),
            (512,  3, 2),
        ]
        blocks = []
        in_ch = 64
        for out_ch, n_blocks, stride in layers_cfg:
            blocks.append(_IResBlock(in_ch, out_ch, stride))
            for _ in range(n_blocks - 1):
                blocks.append(_IResBlock(out_ch, out_ch, 1))
            in_ch = out_ch
        self.body = nn.Sequential(*blocks)

        self.output_layer = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, embedding_dim, bias=False),
            nn.BatchNorm1d(embedding_dim),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return L2-normalised 512-d embedding."""
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_layer(x)
        return F.normalize(x, p=2, dim=1)


def build_backbone(name: str, embedding_dim: int = 512) -> nn.Module:
    """
    Factory: return the requested backbone.

    Parameters
    ----------
    name          : 'mobilefacenet' or 'resnet100'
    embedding_dim : dimensionality of the output embedding.
    """
    name = name.lower().strip()
    if name == "mobilefacenet":
        log.info("Backbone: MobileFaceNet (embedding_dim=%d)", embedding_dim)
        return MobileFaceNet(embedding_dim=embedding_dim)
    if name in ("resnet100", "iresnet100"):
        log.info("Backbone: ResNet-100 / iResNet-100 (embedding_dim=%d)", embedding_dim)
        return ResNet100(embedding_dim=embedding_dim)
    log.error("Unknown backbone '%s'. Choose 'mobilefacenet' or 'resnet100'.", name)
    sys.exit(1)


# ===========================================================================
# 3.  ARCFACE LOSS
# ===========================================================================

class ArcFaceLoss(nn.Module):
    """
    ArcFace: Additive Angular Margin Loss.

    Formulation
    -----------
        logit_i = cos(θ_yi + m)  for the ground-truth class
        logit_j = cos(θ_j)       for all other classes
        loss     = CrossEntropy(s · logits, labels)

    Parameters
    ----------
    embedding_dim : dimensionality of the input feature vector.
    num_classes   : number of identity classes.
    scale (s)     : feature scale (default 64).
    margin (m)    : angular margin in radians (default 0.5 ≈ 28.6°).
    easy_margin   : use cos(m) threshold instead of cos(π - m).

    Reference: Deng et al., "ArcFace: Additive Angular Margin Loss for Deep
               Face Recognition", CVPR 2019.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        scale: float = 64.0,
        margin: float = 0.5,
        easy_margin: bool = False,
    ):
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.easy_margin = easy_margin

        # Learnable class-centre weight matrix  (C × D), stored on the same
        # device as the rest of the model.
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

        # Precompute angular margin constants
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.threshold = math.cos(math.pi - margin)   # for standard margin
        self.sinmm = math.sin(math.pi - margin) * margin  # sin(π-m)·m

        self.ce = nn.CrossEntropyLoss()

    def forward(
        self,
        embeddings: torch.Tensor,   # (B, D) – L2-normalised
        labels: torch.Tensor,        # (B,)  – class indices
    ) -> torch.Tensor:
        # Normalise class centres
        W = F.normalize(self.weight, p=2, dim=1)          # (C, D)

        # Cosine similarity:  (B, C)
        cos_theta = F.linear(embeddings, W)
        cos_theta = cos_theta.clamp(-1.0, 1.0)

        # sin(θ) = sqrt(1 - cos²(θ))
        sin_theta = torch.sqrt(1.0 - cos_theta.pow(2)).clamp(1e-7, 1.0)

        # cos(θ + m) = cos θ cos m − sin θ sin m
        cos_theta_m = cos_theta * self.cos_m - sin_theta * self.sin_m

        if self.easy_margin:
            # Only add margin when cos θ > 0 (θ < 90°)
            cos_theta_m = torch.where(cos_theta > 0, cos_theta_m, cos_theta)
        else:
            # Standard ArcFace: fall back to cos θ − sin(π−m)·m when θ > π−m
            cos_theta_m = torch.where(
                cos_theta > self.threshold,
                cos_theta_m,
                cos_theta - self.sinmm,
            )

        # Build the final logit matrix: replace the ground-truth column with
        # the margin-adjusted value.
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        logits = one_hot * cos_theta_m + (1.0 - one_hot) * cos_theta
        logits *= self.scale

        return self.ce(logits, labels)


# ===========================================================================
# 4.  TRAINING LOOP
# ===========================================================================

def accuracy_top1(embeddings: torch.Tensor, labels: torch.Tensor, weight: torch.Tensor) -> float:
    """Compute top-1 accuracy against the ArcFace class centres (no margin)."""
    with torch.no_grad():
        W = F.normalize(weight, p=2, dim=1)
        cos_sim = F.linear(embeddings, W)       # (B, C)
        preds = cos_sim.argmax(dim=1)
        return (preds == labels).float().mean().item()


def save_checkpoint(
    path: Path,
    backbone: nn.Module,
    arcface: ArcFaceLoss,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    best_val_loss: float,
    scaler: Optional[GradScaler],
    class_to_idx: dict,
) -> None:
    """Persist training state to disk."""
    torch.save(
        {
            "epoch": epoch,
            "backbone_state": backbone.state_dict(),
            "arcface_state": arcface.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler else None,
            "scaler_state": scaler.state_dict() if scaler else None,
            "best_val_loss": best_val_loss,
            "class_to_idx": class_to_idx,
        },
        str(path),
    )


def load_checkpoint(
    path: Path,
    backbone: nn.Module,
    arcface: ArcFaceLoss,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: Optional[GradScaler],
    device: torch.device,
) -> Tuple[int, float]:
    """
    Restore training state from a checkpoint file.

    Returns
    -------
    (start_epoch, best_val_loss)
    """
    log.info("Resuming from checkpoint: %s", path)
    ckpt = torch.load(str(path), map_location=device)
    backbone.load_state_dict(ckpt["backbone_state"])
    arcface.load_state_dict(ckpt["arcface_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler and ckpt.get("scheduler_state"):
        scheduler.load_state_dict(ckpt["scheduler_state"])
    if scaler and ckpt.get("scaler_state"):
        scaler.load_state_dict(ckpt["scaler_state"])
    return ckpt["epoch"] + 1, ckpt["best_val_loss"]


def run_epoch(
    phase: str,
    loader: DataLoader,
    backbone: nn.Module,
    arcface: ArcFaceLoss,
    optimizer: Optional[torch.optim.Optimizer],
    scaler: Optional[GradScaler],
    device: torch.device,
    use_amp: bool,
) -> Tuple[float, float]:
    """
    Run one epoch of training or validation.

    Returns
    -------
    (avg_loss, avg_accuracy)
    """
    is_train = (phase == "train")
    backbone.train(is_train)
    arcface.train(is_train)

    total_loss = 0.0
    total_acc  = 0.0
    n_batches  = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()

    with ctx:
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast(enabled=use_amp):
                embeddings = backbone(images)                    # (B, D)
                loss = arcface(embeddings, labels)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(
                        list(backbone.parameters()) + list(arcface.parameters()),
                        max_norm=5.0,
                    )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        list(backbone.parameters()) + list(arcface.parameters()),
                        max_norm=5.0,
                    )
                    optimizer.step()

            acc = accuracy_top1(embeddings.detach(), labels, arcface.weight.detach())
            total_loss += loss.item()
            total_acc  += acc
            n_batches  += 1

    return total_loss / max(n_batches, 1), total_acc / max(n_batches, 1)


def train(args: argparse.Namespace) -> None:
    """Main training routine."""

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    if args.device == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA requested but not available – falling back to CPU.")
        args.device = "cpu"
    device = torch.device(args.device)
    use_amp = (device.type == "cuda") and args.amp
    log.info("Device: %s | Mixed precision (AMP): %s", device, use_amp)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    train_loader, val_loader, num_classes = build_dataloaders(
        data_dir=args.data,
        val_split=args.val_split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    class_to_idx = train_loader.dataset.subset.dataset.class_to_idx

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    backbone = build_backbone(args.backbone, embedding_dim=args.embedding_dim).to(device)
    arcface  = ArcFaceLoss(
        embedding_dim=args.embedding_dim,
        num_classes=num_classes,
        scale=args.scale,
        margin=args.margin,
        easy_margin=args.easy_margin,
    ).to(device)

    total_params = sum(p.numel() for p in backbone.parameters()) / 1e6
    log.info("Backbone parameters: %.2f M", total_params)

    # ------------------------------------------------------------------
    # Optimiser & Scheduler
    # ------------------------------------------------------------------
    all_params = list(backbone.parameters()) + list(arcface.parameters())

    if args.optimizer == "sgd":
        optimizer = SGD(all_params, lr=args.lr, momentum=0.9, weight_decay=args.wd)
    else:
        optimizer = AdamW(all_params, lr=args.lr, weight_decay=args.wd)

    if args.scheduler == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    else:  # onecycle
        scheduler = OneCycleLR(
            optimizer,
            max_lr=args.lr,
            steps_per_epoch=len(train_loader),
            epochs=args.epochs,
        )

    scaler = GradScaler() if use_amp else None

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------
    start_epoch = 0
    best_val_loss = float("inf")
    checkpoint_path = Path("arcface_checkpoint.pth")
    best_model_path = Path("best_arcface_model.pth")

    if args.resume and Path(args.resume).exists():
        start_epoch, best_val_loss = load_checkpoint(
            Path(args.resume), backbone, arcface, optimizer, scheduler, scaler, device
        )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    log.info(
        "Training for %d epoch(s) starting from epoch %d …",
        args.epochs,
        start_epoch + 1,
    )
    log.info("%-6s  %-12s %-12s %-12s %-12s %-10s", "Epoch", "Train Loss", "Train Acc",
             "Val Loss", "Val Acc", "LR")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        train_loss, train_acc = run_epoch(
            "train", train_loader, backbone, arcface, optimizer, scaler, device, use_amp
        )

        val_loss, val_acc = run_epoch(
            "val", val_loader, backbone, arcface, None, None, device, use_amp
        )

        # Step scheduler
        if args.scheduler == "cosine":
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        log.info(
            "%-6d  %-12.4f %-12.4f %-12.4f %-12.4f %-10.2e  [%.1fs]",
            epoch + 1, train_loss, train_acc, val_loss, val_acc, current_lr, elapsed,
        )

        # ---- Save best model ------------------------------------------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "backbone_state": backbone.state_dict(),
                    "embedding_dim": args.embedding_dim,
                    "backbone": args.backbone,
                    "num_classes": num_classes,
                    "class_to_idx": class_to_idx,
                    "val_loss": best_val_loss,
                    "val_acc": val_acc,
                },
                str(best_model_path),
            )
            log.info(
                "  ↳ Best model saved (val_loss=%.4f, val_acc=%.4f) → %s",
                best_val_loss, val_acc, best_model_path,
            )

        # ---- Rolling checkpoint ----------------------------------------
        save_checkpoint(
            checkpoint_path, backbone, arcface, optimizer, scheduler,
            epoch, best_val_loss, scaler, class_to_idx,
        )

    log.info("Training complete. Best val loss: %.4f", best_val_loss)
    log.info("Best model  → %s", best_model_path.resolve())
    log.info("Checkpoint  → %s", checkpoint_path.resolve())


# ===========================================================================
# 5.  CLI
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a face recognition backbone with ArcFace loss.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Paths
    p.add_argument("--data", type=Path, default=Path("dataset/processed"),
                   help="Root of processed face dataset.")
    p.add_argument("--resume", type=str, default=None,
                   help="Path to a checkpoint .pth file to resume from.")

    # Model
    p.add_argument("--backbone", choices=["mobilefacenet", "resnet100"],
                   default="mobilefacenet",
                   help="Backbone architecture.")
    p.add_argument("--embedding-dim", type=int, default=512,
                   help="Embedding dimensionality.")

    # ArcFace hyper-parameters
    p.add_argument("--scale",  type=float, default=64.0,  help="ArcFace feature scale s.")
    p.add_argument("--margin", type=float, default=0.5,   help="ArcFace angular margin m (rad).")
    p.add_argument("--easy-margin", action="store_true",  help="Use easy-margin variant.")

    # Training
    p.add_argument("--epochs",     type=int,   default=50)
    p.add_argument("--batch-size", type=int,   default=64)
    p.add_argument("--lr",         type=float, default=0.1)
    p.add_argument("--wd",         type=float, default=5e-4, help="Weight decay.")
    p.add_argument("--optimizer",  choices=["sgd", "adamw"], default="sgd")
    p.add_argument("--scheduler",  choices=["cosine", "onecycle"], default="cosine")
    p.add_argument("--val-split",  type=float, default=0.1,
                   help="Fraction of data used for validation.")

    # Hardware
    p.add_argument("--device",      choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--amp",         action="store_true", default=True,
                   help="Enable mixed-precision (FP16) training on CUDA.")
    p.add_argument("--num-workers", type=int, default=4,
                   help="DataLoader worker processes.")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)

