"""
Main training script for vision-language model.

This file is a required deliverable for the Dacon competition.
It handles the complete training pipeline including data loading,
model training, validation, and checkpoint management.
"""

import os
import argparse
import logging
from typing import Optional

import torch
from torch.utils.data import DataLoader

from src.utils.seed import seed_everything
from src.utils.config import CFG
from src.utils.io import read_json_annotation
from src.data.vocab import Vocab
from src.data.dataset import UniDSet
from model import create_model, save_model
from src.training.loss import smooth_l1_loss
from src.training.trainer import train_epoch, validate_epoch

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train vision-language model for document element localization"
    )

    # Data paths
    parser.add_argument(
        "--train_json", type=str, required=True, help="Path to training JSON file"
    )
    parser.add_argument(
        "--train_jpg", type=str, required=True, help="Path to training images directory"
    )
    parser.add_argument(
        "--val_json", type=str, default=None, help="Path to validation JSON file"
    )
    parser.add_argument(
        "--val_jpg", type=str, default=None, help="Path to validation images directory"
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs", help="Output directory for checkpoints"
    )

    # Training hyperparameters
    parser.add_argument(
        "--epochs", type=int, default=CFG.EPOCHS, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=CFG.BATCH_SIZE, help="Batch size"
    )
    parser.add_argument(
        "--lr", type=float, default=CFG.LEARNING_RATE, help="Learning rate"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-4, help="Weight decay for optimizer"
    )

    # Model configuration
    parser.add_argument(
        "--img_size", type=int, default=CFG.IMG_SIZE, help="Input image size (square)"
    )
    parser.add_argument(
        "--dim", type=int, default=CFG.DIM, help="Model embedding dimension"
    )
    parser.add_argument(
        "--no_pretrain",
        action="store_true",
        help="Don't use pretrained ImageNet weights for backbone",
    )

    # Training options
    parser.add_argument(
        "--seed", type=int, default=CFG.SEED, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda/cpu)",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use automatic mixed precision (AMP) training",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=CFG.NUM_WORKERS,
        help="Number of data loading workers",
    )

    return parser.parse_args()


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def build_vocab_from_json(json_path: str) -> Vocab:
    """
    Build vocabulary from training data annotations.

    Args:
        json_path: Path to training JSON file

    Returns:
        Vocab instance with built vocabulary
    """
    vocab = Vocab()

    # Read all annotations
    annotations = read_json_annotation(json_path)

    # Extract all queries from visual category annotations
    all_queries = []
    for item in annotations:
        for ann in item.get("annotations", []):
            # Only process visual category (category_id == 1)
            if ann.get("category_id") == 1:
                query = ann.get("query", "")
                if query:
                    all_queries.append(query)

    # Build vocabulary
    vocab.build(all_queries)
    logger.info(f"Built vocabulary from {len(all_queries)} queries")

    return vocab


def main():
    """Main training function."""
    args = parse_args()
    setup_logging()

    logger.info("="*60)
    logger.info("Vision-Language Model Training")
    logger.info("="*60)
    logger.info(f"Device: {args.device}")
    logger.info(f"Mixed Precision: {args.amp}")
    logger.info(f"Image Size: {args.img_size}")
    logger.info(f"Batch Size: {args.batch_size}")
    logger.info(f"Learning Rate: {args.lr}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info("="*60)

    # Set random seed for reproducibility
    seed_everything(args.seed)
    logger.info(f"Random seed set to {args.seed}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Build vocabulary from training data
    logger.info("\nBuilding vocabulary from training data...")
    vocab = build_vocab_from_json(args.train_json)
    logger.info(f"Vocabulary size: {len(vocab)} tokens")

    # Create training dataset
    logger.info("\nCreating training dataset...")
    train_dataset = UniDSet(
        json_files=[args.train_json],
        jpg_dir=args.train_jpg,
        vocab=vocab,
        build_vocab=False,  # Vocabulary already built
        resize_to=args.img_size,
    )
    logger.info(f"Training samples: {len(train_dataset)}")

    # Create validation dataset if provided
    val_dataset = None
    if args.val_json and args.val_jpg:
        logger.info("Creating validation dataset...")
        val_dataset = UniDSet(
            json_files=[args.val_json],
            jpg_dir=args.val_jpg,
            vocab=vocab,
            build_vocab=False,
            resize_to=args.img_size,
        )
        logger.info(f"Validation samples: {len(val_dataset)}")

    # Create data loaders
    from src.data.dataset import collate_fn

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
        )

    # Create model
    logger.info("\nCreating model...")
    model = create_model(
        vocab_size=len(vocab),
        dim=args.dim,
        pretrained=not args.no_pretrain,
        img_size=args.img_size,
    )
    model = model.to(args.device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # Setup mixed precision training
    scaler = None
    if args.amp and args.device == "cuda":
        scaler = torch.cuda.amp.GradScaler()
        logger.info("Mixed precision training (AMP) enabled")

    # Training loop
    best_miou = 0.0
    best_checkpoint_path = os.path.join(args.output_dir, "best_model.pth")
    last_checkpoint_path = os.path.join(args.output_dir, "last_model.pth")

    logger.info("\n" + "="*60)
    logger.info("Starting training...")
    logger.info("="*60 + "\n")

    for epoch in range(args.epochs):
        logger.info(f"\nEpoch {epoch + 1}/{args.epochs}")
        logger.info("-" * 60)

        # Train for one epoch
        train_metrics = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            loss_fn=smooth_l1_loss,
            device=args.device,
            scaler=scaler,
        )
        logger.info(f"Train Loss: {train_metrics['loss']:.4f}")

        # Validate if validation set provided
        if val_loader:
            val_metrics = validate_epoch(
                model=model,
                dataloader=val_loader,
                loss_fn=smooth_l1_loss,
                device=args.device,
            )
            logger.info(
                f"Val Loss: {val_metrics['loss']:.4f}, Val mIoU: {val_metrics['miou']:.4f}"
            )

            # Save best checkpoint based on validation mIoU
            if val_metrics["miou"] > best_miou:
                best_miou = val_metrics["miou"]

                config = {
                    "dim": args.dim,
                    "no_pretrain": args.no_pretrain,
                    "img_size": args.img_size,
                }

                metadata = {
                    "epoch": epoch + 1,
                    "best_val_miou": best_miou,
                    "train_loss": train_metrics["loss"],
                    "val_loss": val_metrics["loss"],
                }

                save_model(
                    model=model,
                    vocab=vocab,
                    save_path=best_checkpoint_path,
                    config=config,
                    metadata=metadata,
                )
                logger.info(f"✓ Best model saved! mIoU: {best_miou:.4f}")

        # Save last checkpoint
        config = {
            "dim": args.dim,
            "no_pretrain": args.no_pretrain,
            "img_size": args.img_size,
        }

        metadata = {
            "epoch": epoch + 1,
            "train_loss": train_metrics["loss"],
        }

        if val_loader:
            metadata["val_loss"] = val_metrics["loss"]
            metadata["val_miou"] = val_metrics["miou"]

        save_model(
            model=model,
            vocab=vocab,
            save_path=last_checkpoint_path,
            config=config,
            metadata=metadata,
        )

    # Training complete
    logger.info("\n" + "="*60)
    logger.info("Training Complete!")
    logger.info("="*60)

    if val_loader:
        logger.info(f"Best Validation mIoU: {best_miou:.4f}")
        logger.info(f"Best checkpoint: {best_checkpoint_path}")

    logger.info(f"Last checkpoint: {last_checkpoint_path}")


if __name__ == "__main__":
    main()
