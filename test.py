"""
Inference script for generating predictions on test data.

This file is a required deliverable for the Dacon competition.
It loads a trained model checkpoint and generates predictions
for the test dataset in the required CSV format.
"""

import os
import argparse
import logging
from typing import List, Dict, Tuple

import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.seed import seed_everything
from src.utils.config import CFG
from src.utils.io import denormalize_bbox
from src.data.vocab import Vocab
from src.data.dataset import UniDSet
from model import load_model

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate predictions for test data"
    )

    # Model and data paths
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model checkpoint (.pth file)",
    )
    parser.add_argument(
        "--test_json",
        type=str,
        required=True,
        help="Path to test JSON file",
    )
    parser.add_argument(
        "--test_jpg",
        type=str,
        required=True,
        help="Path to test images directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="submission.csv",
        help="Output CSV file path",
    )

    # Inference options
    parser.add_argument(
        "--batch_size",
        type=int,
        default=CFG.BATCH_SIZE,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda/cpu)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=CFG.NUM_WORKERS,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=CFG.SEED,
        help="Random seed for reproducibility",
    )

    return parser.parse_args()


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def generate_predictions(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str,
) -> List[Dict]:
    """
    Generate predictions for all samples in dataloader.

    Args:
        model: Trained VLM model
        dataloader: Test data loader
        device: Device to run inference on

    Returns:
        List of prediction dictionaries with keys:
            - ID: annotation ID
            - x, y, w, h: bounding box in pixel coordinates
    """
    model.eval()
    predictions = []

    with torch.no_grad():
        for images, query_ids, lengths, targets, meta in tqdm(
            dataloader, desc="Generating predictions"
        ):
            images = images.to(device)
            query_ids = query_ids.to(device)
            lengths = lengths.to(device)

            # Forward pass
            pred_norm = model(images, query_ids, lengths)

            # Convert normalized predictions to pixel coordinates
            for i in range(len(pred_norm)):
                # Get original image dimensions
                W, H = meta[i]["orig_size"]

                # Get annotation ID and query text
                query_id = meta[i]["query_id"]
                query_text = meta[i]["query_text"]

                # Convert from normalized [cx, cy, w, h] to pixel [x, y, w, h]
                pred_bbox = denormalize_bbox(
                    tuple(pred_norm[i].cpu().numpy().tolist()), W, H
                )

                x, y, w, h = pred_bbox

                predictions.append({
                    "query_id": query_id,
                    "query_text": query_text,
                    "pred_x": x,
                    "pred_y": y,
                    "pred_w": w,
                    "pred_h": h,
                })

    return predictions


def save_predictions_csv(predictions: List[Dict], output_path: str) -> None:
    """
    Save predictions to CSV file in competition format.

    Args:
        predictions: List of prediction dictionaries
        output_path: Path to output CSV file
    """
    df = pd.DataFrame(predictions)

    # Ensure correct column order (원본 코드 형식)
    df = df[["query_id", "query_text", "pred_x", "pred_y", "pred_w", "pred_h"]]

    # Save to CSV with utf-8-sig encoding (원본 코드와 동일)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info(f"Predictions saved to {output_path}")
    logger.info(f"Total predictions: {len(predictions)}")


def main():
    """Main inference function."""
    args = parse_args()
    setup_logging()

    logger.info("="*60)
    logger.info("Vision-Language Model Inference")
    logger.info("="*60)
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Test JSON: {args.test_json}")
    logger.info(f"Test Images: {args.test_jpg}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Device: {args.device}")
    logger.info("="*60)

    # Set random seed
    seed_everything(args.seed)

    # Load model from checkpoint
    logger.info("\nLoading model from checkpoint...")
    model, vocab_dict = load_model(args.checkpoint, device=args.device)
    logger.info("Model loaded successfully")

    # Reconstruct vocabulary object
    vocab = Vocab()
    vocab.itos = vocab_dict["itos"]
    vocab.stoi = vocab_dict["stoi"]
    logger.info(f"Vocabulary size: {len(vocab)}")

    # Get image size from checkpoint
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    img_size = checkpoint.get("img_size", 512)
    logger.info(f"Image size: {img_size}")

    # Create test dataset
    logger.info("\nCreating test dataset...")
    test_dataset = UniDSet(
        json_files=[args.test_json],
        jpg_dir=args.test_jpg,
        vocab=vocab,
        build_vocab=False,  # Use loaded vocabulary
        resize_to=img_size,
    )
    logger.info(f"Test samples: {len(test_dataset)}")

    # Create data loader
    from src.data.dataset import collate_fn

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,  # Don't shuffle test data
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    # Generate predictions
    logger.info("\nGenerating predictions...")
    predictions = generate_predictions(
        model=model,
        dataloader=test_loader,
        device=args.device,
    )

    # Save predictions to CSV
    logger.info("\nSaving predictions...")
    save_predictions_csv(predictions, args.output)

    logger.info("\n" + "="*60)
    logger.info("Inference Complete!")
    logger.info("="*60)


if __name__ == "__main__":
    main()
