"""
FINAL OPTIMIZED Training Script - Maximum IoU Performance
✅ EMA (Exponential Moving Average)
✅ CIoU Loss + L1 Loss Combination
✅ Gradient Accumulation
✅ Cosine Annealing LR with Warmup
✅ Comprehensive Validation
✅ Mixed Precision Training
"""

import argparse
import os
import logging
import json
import time
from typing import Dict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from preprocess import create_dataloader, seed_everything, Vocabulary
from model import create_model

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EMA:
    """
    Exponential Moving Average for model parameters.
    Improves model stability and generalization.
    """

    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        """Register model parameters for EMA tracking."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """Update EMA shadow weights after each training step."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        """Apply EMA weights for validation."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        """Restore original weights after validation."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


def to_xyxy(box: torch.Tensor) -> torch.Tensor:
    """
    Convert (cx, cy, w, h) to (x1, y1, x2, y2) format.

    Args:
        box: [B, 4] in normalized (cx, cy, w, h)

    Returns:
        [B, 4] in (x1, y1, x2, y2)
    """
    cx, cy, w, h = box.unbind(-1)
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return torch.stack([x1, y1, x2, y2], dim=-1)


def compute_iou(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute IoU between predicted and ground truth bboxes.

    Args:
        pred: [B, 4] in normalized (cx, cy, w, h)
        target: [B, 4] in normalized (cx, cy, w, h)

    Returns:
        IoU: [B]
    """
    pred_xyxy = to_xyxy(pred)
    tgt_xyxy = to_xyxy(target)

    # Intersection
    x1 = torch.max(pred_xyxy[..., 0], tgt_xyxy[..., 0])
    y1 = torch.max(pred_xyxy[..., 1], tgt_xyxy[..., 1])
    x2 = torch.min(pred_xyxy[..., 2], tgt_xyxy[..., 2])
    y2 = torch.min(pred_xyxy[..., 3], tgt_xyxy[..., 3])

    inter_w = (x2 - x1).clamp(min=0)
    inter_h = (y2 - y1).clamp(min=0)
    inter = inter_w * inter_h

    # Union
    area_p = (pred_xyxy[..., 2] - pred_xyxy[..., 0]) * (pred_xyxy[..., 3] - pred_xyxy[..., 1])
    area_t = (tgt_xyxy[..., 2] - tgt_xyxy[..., 0]) * (tgt_xyxy[..., 3] - tgt_xyxy[..., 1])
    area_p = area_p.clamp(min=0)
    area_t = area_t.clamp(min=0)
    union = area_p + area_t - inter + 1e-6

    iou = inter / union
    return iou


def ciou_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Complete-IoU Loss - Best for bbox regression.

    CIoU = DIoU - α*v
    where v is the aspect ratio consistency and α is the weight.

    Args:
        pred: [B, 4] in normalized (cx, cy, w, h)
        target: [B, 4] in normalized (cx, cy, w, h)

    Returns:
        CIoU Loss: scalar
    """
    pred_xyxy = to_xyxy(pred)
    tgt_xyxy = to_xyxy(target)

    # Intersection
    x1_inter = torch.max(pred_xyxy[..., 0], tgt_xyxy[..., 0])
    y1_inter = torch.max(pred_xyxy[..., 1], tgt_xyxy[..., 1])
    x2_inter = torch.min(pred_xyxy[..., 2], tgt_xyxy[..., 2])
    y2_inter = torch.min(pred_xyxy[..., 3], tgt_xyxy[..., 3])

    inter_w = (x2_inter - x1_inter).clamp(min=0)
    inter_h = (y2_inter - y1_inter).clamp(min=0)
    inter = inter_w * inter_h

    # Union
    area_p = (pred_xyxy[..., 2] - pred_xyxy[..., 0]) * (pred_xyxy[..., 3] - pred_xyxy[..., 1])
    area_t = (tgt_xyxy[..., 2] - tgt_xyxy[..., 0]) * (tgt_xyxy[..., 3] - tgt_xyxy[..., 1])
    area_p = area_p.clamp(min=0)
    area_t = area_t.clamp(min=0)
    union = area_p + area_t - inter + 1e-7

    # IoU
    iou = inter / union

    # Center distance
    pred_cx = (pred_xyxy[..., 0] + pred_xyxy[..., 2]) / 2
    pred_cy = (pred_xyxy[..., 1] + pred_xyxy[..., 3]) / 2
    tgt_cx = (tgt_xyxy[..., 0] + tgt_xyxy[..., 2]) / 2
    tgt_cy = (tgt_xyxy[..., 1] + tgt_xyxy[..., 3]) / 2

    center_distance = (pred_cx - tgt_cx) ** 2 + (pred_cy - tgt_cy) ** 2

    # Diagonal of enclosing box
    x1_enclose = torch.min(pred_xyxy[..., 0], tgt_xyxy[..., 0])
    y1_enclose = torch.min(pred_xyxy[..., 1], tgt_xyxy[..., 1])
    x2_enclose = torch.max(pred_xyxy[..., 2], tgt_xyxy[..., 2])
    y2_enclose = torch.max(pred_xyxy[..., 3], tgt_xyxy[..., 3])

    enclose_w = (x2_enclose - x1_enclose).clamp(min=0)
    enclose_h = (y2_enclose - y1_enclose).clamp(min=0)
    enclose_diagonal = enclose_w ** 2 + enclose_h ** 2 + 1e-7

    # Aspect ratio consistency
    pred_w = (pred_xyxy[..., 2] - pred_xyxy[..., 0]).clamp(min=1e-7)
    pred_h = (pred_xyxy[..., 3] - pred_xyxy[..., 1]).clamp(min=1e-7)
    tgt_w = (tgt_xyxy[..., 2] - tgt_xyxy[..., 0]).clamp(min=1e-7)
    tgt_h = (tgt_xyxy[..., 3] - tgt_xyxy[..., 1]).clamp(min=1e-7)

    v = (4 / (torch.pi ** 2)) * torch.pow(torch.atan(tgt_w / tgt_h) - torch.atan(pred_w / pred_h), 2)

    with torch.no_grad():
        alpha = v / (1 - iou + v + 1e-7)

    # CIoU
    ciou = iou - (center_distance / enclose_diagonal) - alpha * v

    # Loss = 1 - CIoU
    loss = 1 - ciou
    return loss.mean()


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    scaler,
    device,
    epoch,
    writer,
    global_step,
    args,
    ema=None,
) -> tuple:
    """
    Train for one epoch with gradient accumulation and mixed precision.

    Args:
        model: Model to train
        dataloader: Training data loader
        optimizer: Optimizer
        scaler: GradScaler for mixed precision
        device: Device (cuda/cpu)
        epoch: Current epoch number
        writer: TensorBoard writer
        global_step: Global training step
        args: Training arguments
        ema: EMA instance (optional)

    Returns:
        metrics: Dict with loss and IoU
        global_step: Updated global step
    """
    model.train()

    total_loss = 0.0
    total_iou = 0.0
    n_samples = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")

    for step, batch in enumerate(pbar):
        images = batch["images"].to(device)
        text_ids = batch["text_ids"].to(device)
        text_lens = batch["text_lens"].to(device)
        targets = batch["targets"].to(device)

        B = images.size(0)
        n_samples += B

        # Forward with mixed precision
        with autocast(enabled=args.use_amp):
            preds = model(images, text_ids, text_lens)
            preds = torch.clamp(preds, 0.0, 1.0)

            # Combined loss: CIoU + L1
            ciou = ciou_loss(preds, targets)
            l1 = F.smooth_l1_loss(preds, targets)
            loss = ciou * args.ciou_weight + l1

            # Gradient accumulation
            loss = loss / args.accumulation_steps

        # Backward
        scaler.scale(loss).backward()

        # Optimizer step with gradient accumulation
        if (step + 1) % args.accumulation_steps == 0 or (step + 1) == len(dataloader):
            # Gradient clipping
            if args.max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            # Update EMA
            if ema is not None:
                ema.update()

        # Metrics
        with torch.no_grad():
            iou = compute_iou(preds.detach(), targets)
            total_loss += loss.item() * B * args.accumulation_steps
            total_iou += iou.sum().item()

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item() * args.accumulation_steps:.4f}',
            'iou': f'{iou.mean().item():.4f}'
        })

        # TensorBoard logging
        if step % 10 == 0:
            writer.add_scalar('train/loss_step', loss.item() * args.accumulation_steps, global_step)
            writer.add_scalar('train/iou_step', iou.mean().item(), global_step)

        global_step += 1

    avg_loss = total_loss / n_samples
    avg_iou = total_iou / n_samples

    return {"loss": avg_loss, "iou": avg_iou}, global_step


@torch.no_grad()
def evaluate(model, dataloader, device, args) -> Dict:
    """
    Evaluate model on validation set.

    Args:
        model: Model to evaluate
        dataloader: Validation data loader
        device: Device (cuda/cpu)
        args: Training arguments

    Returns:
        metrics: Dict with loss, IoU, and threshold metrics
    """
    model.eval()

    total_loss = 0.0
    total_iou = 0.0
    iou_50_count = 0
    iou_75_count = 0
    n_samples = 0

    for batch in tqdm(dataloader, desc="Validation"):
        images = batch["images"].to(device)
        text_ids = batch["text_ids"].to(device)
        text_lens = batch["text_lens"].to(device)
        targets = batch["targets"].to(device)

        B = images.size(0)
        n_samples += B

        preds = model(images, text_ids, text_lens)
        preds = torch.clamp(preds, 0.0, 1.0)

        # Compute loss
        ciou = ciou_loss(preds, targets)
        l1 = F.smooth_l1_loss(preds, targets)
        loss = ciou * args.ciou_weight + l1

        iou = compute_iou(preds, targets)

        total_loss += loss.item() * B
        total_iou += iou.sum().item()
        iou_50_count += (iou > 0.5).sum().item()
        iou_75_count += (iou > 0.75).sum().item()

    avg_loss = total_loss / n_samples if n_samples > 0 else 0.0
    avg_iou = total_iou / n_samples if n_samples > 0 else 0.0
    iou_50 = iou_50_count / n_samples if n_samples > 0 else 0.0
    iou_75 = iou_75_count / n_samples if n_samples > 0 else 0.0

    return {
        "loss": avg_loss,
        "iou": avg_iou,
        "iou@0.5": iou_50,
        "iou@0.75": iou_75,
    }


def main(args):
    """Main training function."""
    # Seed for reproducibility
    seed_everything(args.seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("="*70)
    print("🚀 FINAL OPTIMIZED TRAINING - Maximum IoU Performance")
    print("="*70)
    print(f"Device: {device}")
    print(f"Image Size: {args.img_size}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.lr}")
    print(f"Epochs: {args.epochs}")
    print(f"Loss: CIoU + L1 (weight={args.ciou_weight})")
    print(f"EMA: {args.use_ema} (decay={args.ema_decay})")
    print(f"Mixed Precision: {args.use_amp}")
    print(f"Gradient Accumulation: {args.accumulation_steps} steps")
    print("="*70)

    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # TensorBoard
    writer = SummaryWriter(log_dir=args.log_dir)

    # Create dataloaders
    logger.info("\n📂 Loading training data...")

    # Determine if using single or multiple directories
    if args.train_json_dirs and args.train_img_roots:
        # Multiple directories
        train_samples, vocab, train_ds, train_loader = create_dataloader(
            json_dirs=args.train_json_dirs,
            img_roots=args.train_img_roots,
            vocab=None,
            build_vocab=True,
            batch_size=args.batch_size,
            img_size=args.img_size,
            is_train=True,
            use_albumentations=args.use_albumentations,
            num_workers=args.num_workers,
        )
    elif args.train_json_dir and args.train_img_root:
        # Single directory (backward compatibility)
        train_samples, vocab, train_ds, train_loader = create_dataloader(
            json_dir=args.train_json_dir,
            img_root=args.train_img_root,
            vocab=None,
            build_vocab=True,
            batch_size=args.batch_size,
            img_size=args.img_size,
            is_train=True,
            use_albumentations=args.use_albumentations,
            num_workers=args.num_workers,
        )
    else:
        raise ValueError("Must provide either (--train_json_dir, --train_img_root) or (--train_json_dirs, --train_img_roots)")

    logger.info(f"✅ Vocabulary size: {len(vocab)}")
    logger.info(f"✅ Training samples: {len(train_samples)}")

    # Validation dataloader
    val_loader = None
    if (args.val_json_dirs and args.val_img_roots) or (args.val_json_dir and args.val_img_root):
        logger.info("\n📂 Loading validation data...")

        if args.val_json_dirs and args.val_img_roots:
            # Multiple validation directories
            val_samples, _, val_ds, val_loader = create_dataloader(
                json_dirs=args.val_json_dirs,
                img_roots=args.val_img_roots,
                vocab=vocab,
                build_vocab=False,
                batch_size=args.batch_size,
                img_size=args.img_size,
                is_train=False,
                use_albumentations=args.use_albumentations,
                num_workers=args.num_workers,
            )
        else:
            # Single validation directory
            val_samples, _, val_ds, val_loader = create_dataloader(
                json_dir=args.val_json_dir,
                img_root=args.val_img_root,
                vocab=vocab,
            build_vocab=False,
            batch_size=args.batch_size,
            img_size=args.img_size,
            is_train=False,
            use_albumentations=False,
            num_workers=args.num_workers,
        )
        logger.info(f"✅ Validation samples: {len(val_samples)}")

    # Create model
    logger.info(f"\n🧠 Creating model with backbone={args.backbone}...")
    model = create_model(
        vocab_size=len(vocab),
        embed_dim=args.embed_dim,
        backbone=args.backbone,
        pretrained=args.pretrained,
        num_heads=args.num_heads,
        dropout=args.dropout,
    )
    model = model.to(device)

    # Log parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"✅ Total params: {total_params:,}")
    logger.info(f"✅ Trainable params: {trainable_params:,}")

    # EMA
    ema = None
    if args.use_ema:
        ema = EMA(model, decay=args.ema_decay)
        logger.info(f"✅ Using EMA with decay={args.ema_decay}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Scheduler with warmup
    if args.scheduler == "cosine":
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs - args.warmup_epochs, eta_min=args.lr * 0.01
        )
        if args.warmup_epochs > 0:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.1, total_iters=args.warmup_epochs
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[args.warmup_epochs]
            )
        else:
            scheduler = main_scheduler
    else:
        scheduler = None

    # Mixed precision scaler
    scaler = GradScaler(enabled=args.use_amp)

    # Training loop
    best_iou = 0.0
    best_epoch = 0
    patience_counter = 0
    global_step = 0
    training_log = []

    logger.info("\n" + "="*70)
    logger.info("🎯 Starting training...")
    logger.info("="*70)

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        logger.info(f"\n{'='*70}")
        logger.info(f"Epoch {epoch}/{args.epochs}")
        logger.info(f"{'='*70}")

        # Train
        train_stats, global_step = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            epoch=epoch,
            writer=writer,
            global_step=global_step,
            args=args,
            ema=ema,
        )

        # Log epoch stats
        logger.info(
            f"[Epoch {epoch}] Train - Loss: {train_stats['loss']:.4f}, "
            f"IoU: {train_stats['iou']:.4f}"
        )
        writer.add_scalar('train/loss_epoch', train_stats['loss'], epoch)
        writer.add_scalar('train/iou_epoch', train_stats['iou'], epoch)

        # Validate
        if val_loader is not None:
            # Apply EMA for validation
            if ema is not None:
                ema.apply_shadow()

            val_stats = evaluate(model, val_loader, device, args)

            # Restore original weights
            if ema is not None:
                ema.restore()

            logger.info(
                f"[Epoch {epoch}] Val   - Loss: {val_stats['loss']:.4f}, "
                f"IoU: {val_stats['iou']:.4f}, "
                f"IoU@0.5: {val_stats['iou@0.5']:.4f}, "
                f"IoU@0.75: {val_stats['iou@0.75']:.4f}"
            )

            writer.add_scalar('val/loss', val_stats['loss'], epoch)
            writer.add_scalar('val/iou', val_stats['iou'], epoch)
            writer.add_scalar('val/iou@0.5', val_stats['iou@0.5'], epoch)
            writer.add_scalar('val/iou@0.75', val_stats['iou@0.75'], epoch)

            # Save best model
            if val_stats["iou"] > best_iou:
                best_iou = val_stats["iou"]
                best_epoch = epoch
                patience_counter = 0

                ckpt_path = os.path.join(args.save_dir, "best_model.pt")
                checkpoint = {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "vocab": vocab.vocab,
                    "idx2word": vocab.idx2word,
                    "best_iou": best_iou,
                    "args": vars(args),
                }
                if scheduler is not None:
                    checkpoint["scheduler_state"] = scheduler.state_dict()
                if ema is not None:
                    checkpoint["ema_shadow"] = ema.shadow

                torch.save(checkpoint, ckpt_path)
                logger.info(f"✅ Saved best model (IoU: {best_iou:.4f}) to {ckpt_path}")
            else:
                patience_counter += 1
                logger.info(f"⏳ No improvement for {patience_counter} epoch(s)")

            # Early stopping
            if args.patience > 0 and patience_counter >= args.patience:
                logger.info(f"\n{'='*70}")
                logger.info(f"⛔ Early stopping at epoch {epoch}")
                logger.info(f"🏆 Best IoU: {best_iou:.4f} at epoch {best_epoch}")
                logger.info(f"{'='*70}")
                break

        # LR scheduler step
        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            logger.info(f"📊 LR: {current_lr:.2e}")
            writer.add_scalar('train/lr', current_lr, epoch)

        # Log
        epoch_time = time.time() - start_time
        log_entry = {
            'epoch': epoch,
            'train_loss': train_stats['loss'],
            'train_iou': train_stats['iou'],
            'val_loss': val_stats['loss'] if val_loader else 0,
            'val_iou': val_stats['iou'] if val_loader else 0,
            'lr': optimizer.param_groups[0]['lr'],
            'time': epoch_time
        }
        training_log.append(log_entry)

    # Save training log
    log_path = os.path.join(args.save_dir, "training_log.json")
    with open(log_path, 'w') as f:
        json.dump(training_log, f, indent=2)

    logger.info("\n" + "="*70)
    logger.info("🎉 Training completed!")
    if val_loader:
        logger.info(f"🏆 Best IoU: {best_iou:.4f} at epoch {best_epoch}")
    logger.info("="*70)

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FINAL Training Script for Maximum IoU")

    # Data (supports both single and multiple directories)
    parser.add_argument("--train_json_dir", type=str, default=None, help="Training JSON directory (single)")
    parser.add_argument("--train_img_root", type=str, default=None, help="Training images directory (single)")
    parser.add_argument("--train_json_dirs", type=str, nargs='+', default=None, help="Training JSON directories (multiple)")
    parser.add_argument("--train_img_roots", type=str, nargs='+', default=None, help="Training images directories (multiple)")

    parser.add_argument("--val_json_dir", type=str, default=None, help="Validation JSON directory (single)")
    parser.add_argument("--val_img_root", type=str, default=None, help="Validation images directory (single)")
    parser.add_argument("--val_json_dirs", type=str, nargs='+', default=None, help="Validation JSON directories (multiple)")
    parser.add_argument("--val_img_roots", type=str, nargs='+', default=None, help="Validation images directories (multiple)")

    # Model
    parser.add_argument("--backbone", type=str, default="resnet50",
                       choices=["resnet18", "resnet34", "resnet50"], help="Image backbone")
    parser.add_argument("--pretrained", action="store_true", default=True, help="Use pretrained weights")
    parser.add_argument("--embed_dim", type=int, default=256, help="Embedding dimension")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")

    # Training
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")

    # Loss
    parser.add_argument("--ciou_weight", type=float, default=2.0, help="CIoU weight in combined loss")

    # Scheduler
    parser.add_argument("--scheduler", type=str, default="cosine",
                       choices=["cosine", "none"], help="Learning rate scheduler")
    parser.add_argument("--warmup_epochs", type=int, default=3, help="Warmup epochs")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for clipping")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--use_amp", action="store_true", default=True, help="Use mixed precision")

    # EMA
    parser.add_argument("--use_ema", action="store_true", default=True, help="Use EMA")
    parser.add_argument("--ema_decay", type=float, default=0.9999, help="EMA decay rate")

    # Data processing
    parser.add_argument("--img_size", type=int, default=512, help="Input image size")
    parser.add_argument("--use_albumentations", action="store_true", default=True, help="Use Albumentations")
    parser.add_argument("--num_workers", type=int, default=4, help="Data loading workers")

    # Other
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--save_dir", type=str, default="./checkpoints_final", help="Checkpoint directory")
    parser.add_argument("--log_dir", type=str, default="./logs_final", help="TensorBoard log directory")

    args = parser.parse_args()
    main(args)
