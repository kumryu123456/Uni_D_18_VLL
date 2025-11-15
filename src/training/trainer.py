"""Training loop and checkpoint management."""
import os
import logging
from typing import Dict, List, Optional, Callable
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.training.metrics import iou_xywh_pixel, compute_miou
from src.utils.io import denormalize_bbox

logger = logging.getLogger(__name__)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable,
    device: str = "cuda",
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> Dict[str, float]:
    """Train model for one epoch."""
    model.train()
    running_loss = 0.0
    total_samples = 0

    pbar = tqdm(dataloader, desc="Training")
    for images, query_ids, lengths, targets, meta in pbar:
        images = images.to(device)
        query_ids = query_ids.to(device)
        lengths = lengths.to(device)

        # Filter only samples with targets
        valid_idx = [i for i, t in enumerate(targets) if t is not None]
        if len(valid_idx) == 0:
            continue

        images = images[valid_idx]
        query_ids = query_ids[valid_idx]
        lengths = lengths[valid_idx]
        targets_t = torch.stack([targets[i] for i in valid_idx]).to(device)

        optimizer.zero_grad(set_to_none=True)

        # Mixed precision training
        if scaler is not None:
            with torch.cuda.amp.autocast():
                pred = model(images, query_ids, lengths)
                loss = loss_fn(pred, targets_t)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(images, query_ids, lengths)
            loss = loss_fn(pred, targets_t)
            loss.backward()
            optimizer.step()

        batch_loss = float(loss.item())
        running_loss += batch_loss * len(valid_idx)
        total_samples += len(valid_idx)

        pbar.set_postfix({"loss": f"{batch_loss:.4f}"})

    avg_loss = running_loss / total_samples if total_samples > 0 else 0.0
    return {"loss": avg_loss}


def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: Callable,
    device: str = "cuda",
) -> Dict[str, float]:
    """Validate model on validation set."""
    model.eval()
    running_loss = 0.0
    ious = []
    total_samples = 0

    with torch.no_grad():
        for images, query_ids, lengths, targets, meta in tqdm(
            dataloader, desc="Validation"
        ):
            images = images.to(device)
            query_ids = query_ids.to(device)
            lengths = lengths.to(device)

            pred = model(images, query_ids, lengths)

            # Compute loss and IoU for samples with targets
            for i in range(len(pred)):
                if targets[i] is not None:
                    target_t = targets[i].unsqueeze(0).to(device)
                    loss = loss_fn(pred[i : i + 1], target_t)
                    running_loss += float(loss.item())
                    total_samples += 1

                    # Compute IoU in pixel coordinates
                    W, H = meta[i]["orig_size"]
                    pred_bbox = denormalize_bbox(
                        tuple(pred[i].cpu().numpy().tolist()), W, H
                    )
                    gt_bbox = denormalize_bbox(
                        tuple(targets[i].numpy().tolist()), W, H
                    )
                    iou = iou_xywh_pixel(pred_bbox, gt_bbox)
                    ious.append(iou)

    avg_loss = running_loss / total_samples if total_samples > 0 else 0.0
    miou = compute_miou(ious)

    return {"loss": avg_loss, "miou": miou}


def save_checkpoint(
    model: nn.Module,
    vocab: object,
    save_path: str,
    config: Dict,
    epoch: int = 0,
    val_miou: float = 0.0,
) -> None:
    """Save training checkpoint."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    checkpoint = {
        "model_state": model.state_dict(),
        "vocab_itos": vocab.itos,
        "dim": config.get("dim", 256),
        "no_pretrain": config.get("no_pretrain", False),
        "img_size": config.get("img_size", 512),
        "epoch": epoch,
        "best_val_miou": val_miou,
    }

    torch.save(checkpoint, save_path)
    logger.info(f"Checkpoint saved: {save_path}")


def load_checkpoint(checkpoint_path: str, device: str = "cpu") -> Dict:
    """Load training checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    logger.info(f"Checkpoint loaded: {checkpoint_path}")
    return checkpoint
