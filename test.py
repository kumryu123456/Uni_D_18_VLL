"""
FINAL OPTIMIZED Test & Inference Script - Maximum IoU Performance
✅ Test Time Augmentation (TTA)
✅ Model Ensemble Support
✅ EMA Weight Loading
✅ Robust Error Handling
✅ High-quality Predictions
"""

import os
import json
import argparse
from glob import glob
from typing import Dict, Any, List

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageOps
from tqdm import tqdm

from preprocess import (
    Vocabulary,
    is_visual_element,
    validate_json_file,
    seed_everything,
)

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==================== Helper Functions ====================
def read_json(json_path: str) -> Dict[str, Any]:
    """Read and parse JSON file"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def normalize_bbox(bbox: list, width: int, height: int) -> list:
    """
    Normalize bbox from [x, y, w, h] (pixels) to [cx, cy, w, h] (0-1).

    Args:
        bbox: [x, y, w, h] in pixels
        width: Image width
        height: Image height

    Returns:
        [cx, cy, w, h] normalized to [0, 1]
    """
    x, y, w, h = bbox
    cx = (x + w / 2.0) / width
    cy = (y + h / 2.0) / height
    nw = w / width
    nh = h / height
    return [cx, cy, nw, nh]


def denormalize_bbox(bbox_norm: list, width: int, height: int) -> list:
    """
    Denormalize bbox from [cx, cy, w, h] (0-1) to [x, y, w, h] (pixels).

    Args:
        bbox_norm: [cx, cy, w, h] normalized
        width: Image width
        height: Image height

    Returns:
        [x, y, w, h] in pixels
    """
    cx, cy, nw, nh = bbox_norm
    w = nw * width
    h = nh * height
    x = (cx - nw / 2.0) * width
    y = (cy - nh / 2.0) * height
    return [x, y, w, h]


def get_image_path(json_path: str, data: Dict[str, Any], jpg_root: str) -> str:
    """
    Find corresponding JPG image for JSON annotation file.

    Args:
        json_path: Path to JSON file
        data: Parsed JSON data
        jpg_root: Root directory for images

    Returns:
        Path to corresponding image

    Raises:
        FileNotFoundError: If image cannot be found
    """
    src = data.get("source_data_info", {})
    jpg_name = src.get("source_data_name_jpg", None)

    if jpg_name:
        cand = os.path.join(jpg_root, jpg_name)
        if os.path.exists(cand):
            return cand

        maybe = json_path.replace(os.sep + "json" + os.sep, os.sep + "jpg" + os.sep)
        maybe = os.path.join(os.path.dirname(maybe), jpg_name)
        if os.path.exists(maybe):
            return maybe

        base = os.path.basename(json_path)
        jpg_base = base.replace("MI3", "MI2").rsplit(".", 1)[0] + ".jpg"
        sibling = os.path.join(jpg_root, jpg_base)
        if os.path.exists(sibling):
            return sibling

    base = os.path.basename(json_path)
    stem = os.path.splitext(base)[0]
    cand1 = os.path.join(jpg_root, stem + ".jpg")
    cand2 = os.path.join(jpg_root, stem.replace("MI3", "MI2") + ".jpg")

    if os.path.exists(cand1):
        return cand1
    if os.path.exists(cand2):
        return cand2

    raise FileNotFoundError(f"JPG not found for {json_path}")


class TestDataset(Dataset):
    """Test dataset for inference."""

    def __init__(self, test_dir: str, vocab: Vocabulary, img_size: int = 512):
        """
        Args:
            test_dir: Test data directory
            vocab: Vocabulary instance
            img_size: Image size for resizing
        """
        json_dir = os.path.join(test_dir, "query")
        jpg_root = os.path.join(test_dir, "images")

        json_files = sorted(glob(os.path.join(json_dir, "*.json")))
        self.samples: List[Dict[str, Any]] = []
        self.vocab = vocab
        self.img_size = img_size

        if not json_files:
            print(f"⚠️ No JSON files found in {json_dir}")

        for jf in json_files:
            if not validate_json_file(jf):
                continue

            try:
                data = read_json(jf)
            except Exception:
                continue

            try:
                img_path = get_image_path(jf, data, jpg_root)
            except FileNotFoundError:
                continue

            ann_list = data.get("learning_data_info", {}).get("annotation", [])

            for ann in ann_list:
                if not is_visual_element(ann):
                    continue

                instance_id = str(ann.get("instance_id", "") or "").strip()
                qtext = str(ann.get("visual_instruction", "") or "").strip()

                if not instance_id or not qtext:
                    continue

                self.samples.append({
                    "query_id": instance_id,
                    "query_text": qtext,
                    "image_path": img_path,
                })

        print(f"✅ Test dataset loaded: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = Image.open(s["image_path"]).convert("RGB")
        orig_w, orig_h = img.size

        # Resize
        img_resized = img.resize((self.img_size, self.img_size), Image.BILINEAR)

        # Tokenize
        tokens = self.vocab.tokenize(s["query_text"])
        token_ids = self.vocab.encode(tokens)
        text_len = len(token_ids)

        # To tensor
        img_tensor = torch.from_numpy(np.array(img_resized)).permute(2, 0, 1).float() / 255.0

        meta = {
            "orig_size": (orig_w, orig_h),
            "image_path": s["image_path"],
            "query_text": s["query_text"],
        }

        return s["query_id"], img_tensor, token_ids, text_len, img, meta


def collate_fn_test(batch):
    """
    Collate function for test data loader.

    Args:
        batch: List of samples

    Returns:
        Batched tensors and metadata
    """
    query_ids, img_tensors, token_lists, text_lens, orig_images, metas = zip(*batch)

    # Stack images
    images = torch.stack(img_tensors, dim=0)

    # Pad token sequences
    max_len = max(text_lens)
    batch_size = len(query_ids)

    text_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
    for i, tokens in enumerate(token_lists):
        text_ids[i, :len(tokens)] = torch.tensor(tokens, dtype=torch.long)

    text_lens_tensor = torch.tensor(text_lens, dtype=torch.long)

    return list(query_ids), images, text_ids, text_lens_tensor, list(orig_images), list(metas)


def load_model(checkpoint_path: str, device):
    """
    Load trained model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on

    Returns:
        model: Loaded model
        vocab: Vocabulary instance
        args: Training arguments
    """
    from model import create_model

    print(f"📦 Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    # Restore vocabulary
    vocab = Vocabulary()
    vocab.vocab = checkpoint["vocab"]
    vocab.idx2word = checkpoint["idx2word"]

    # Get model config
    train_args = checkpoint.get("args", {})
    vocab_size = len(vocab)

    # Create model
    model = create_model(
        vocab_size=vocab_size,
        embed_dim=train_args.get("embed_dim", 256),
        backbone=train_args.get("backbone", "resnet50"),
        pretrained=False,  # Don't need pretrained for inference
        num_heads=train_args.get("num_heads", 8),
        dropout=train_args.get("dropout", 0.1),
    )

    # Load weights
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device)
    model.eval()

    # Load EMA weights if available
    if "ema_shadow" in checkpoint:
        print("✅ Loading EMA weights")
        for name, param in model.named_parameters():
            if name in checkpoint["ema_shadow"]:
                param.data = checkpoint["ema_shadow"][name].to(device)

    best_iou = checkpoint.get("best_iou", 0.0)
    epoch = checkpoint.get("epoch", 0)

    print(f"✅ Model loaded successfully")
    print(f"   Epoch: {epoch}, Best IoU: {best_iou:.4f}")
    print(f"   Vocab size: {vocab_size}")

    return model, vocab, train_args


def load_ensemble_models(checkpoint_paths: List[str], device):
    """
    Load multiple models for ensemble.

    Args:
        checkpoint_paths: List of checkpoint paths
        device: Device to load models on

    Returns:
        models: List of loaded models
        vocab: Vocabulary instance (from first model)
    """
    models = []
    vocab = None

    for i, ckpt_path in enumerate(checkpoint_paths):
        if os.path.exists(ckpt_path):
            print(f"\n📦 Loading ensemble model {i+1}/{len(checkpoint_paths)}: {ckpt_path}")
            model, v, _ = load_model(ckpt_path, device)
            models.append(model)

            if vocab is None:
                vocab = v

    if not models:
        raise ValueError("No valid models found for ensemble")

    print(f"\n✅ Loaded {len(models)} models for ensemble")
    return models, vocab


def apply_tta(model, images, text_ids, text_lens, device):
    """
    Apply Test Time Augmentation (TTA).

    Args:
        model: Model to use
        images: [B, 3, H, W] image tensor
        text_ids: [B, L] text token IDs
        text_lens: [B] text lengths
        device: Device

    Returns:
        predictions: List of [original_pred, flipped_pred]
    """
    predictions = []

    # 1. Original prediction
    with torch.no_grad():
        pred = model(images, text_ids, text_lens)
        pred = torch.clamp(pred, 0.0, 1.0)
        predictions.append(pred)

    # 2. Horizontal flip prediction
    images_flip = torch.flip(images, dims=[3])  # Flip width dimension

    with torch.no_grad():
        pred_flip = model(images_flip, text_ids, text_lens)
        pred_flip = torch.clamp(pred_flip, 0.0, 1.0)

        # Unflip predictions: cx_flip = 1 - cx
        pred_flip_unflipped = pred_flip.clone()
        pred_flip_unflipped[:, 0] = 1.0 - pred_flip[:, 0]  # Flip cx back

        predictions.append(pred_flip_unflipped)

    return predictions


def merge_tta_predictions(tta_preds: List[torch.Tensor]) -> torch.Tensor:
    """
    Merge TTA predictions using averaging.

    Args:
        tta_preds: List of [B, 4] predictions

    Returns:
        [B, 4] averaged predictions
    """
    # Stack and average
    stacked = torch.stack(tta_preds, dim=0)  # [N, B, 4]
    avg_pred = stacked.mean(dim=0)  # [B, 4]
    return avg_pred


@torch.no_grad()
def run_inference(
    models: List,
    loader: DataLoader,
    output_csv: str,
    device,
    enable_tta: bool = True,
):
    """
    Run inference with TTA and optional ensemble.

    Args:
        models: List of models (single or ensemble)
        loader: Data loader
        output_csv: Output CSV file path
        device: Device
        enable_tta: Enable Test Time Augmentation
    """
    for model in models:
        model.eval()

    results = []
    use_ensemble = len(models) > 1

    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Inference (TTA={enable_tta}, Ensemble={use_ensemble})")

        for query_ids, images, text_ids, text_lens, orig_images, metas in pbar:
            # Move to device
            images = images.to(device)
            text_ids = text_ids.to(device)
            text_lens = text_lens.to(device)

            batch_size = len(query_ids)
            ensemble_predictions = []

            # Ensemble: run each model
            for model in models:
                if enable_tta:
                    # TTA for this model
                    tta_preds = apply_tta(model, images, text_ids, text_lens, device)
                    merged_pred = merge_tta_predictions(tta_preds)
                else:
                    # No TTA, just original
                    pred = model(images, text_ids, text_lens)
                    merged_pred = torch.clamp(pred, 0.0, 1.0)

                ensemble_predictions.append(merged_pred)

            # Ensemble: average all model predictions
            if use_ensemble:
                stacked = torch.stack(ensemble_predictions, dim=0)  # [M, B, 4]
                final_pred = stacked.mean(dim=0)  # [B, 4]
            else:
                final_pred = ensemble_predictions[0]

            # Convert to numpy
            final_pred = final_pred.cpu().numpy()  # [B, 4]

            # Process each sample
            for i in range(batch_size):
                orig_w, orig_h = metas[i]["orig_size"]
                bbox_norm = final_pred[i]  # [4] in [0, 1]

                # Denormalize to original image size
                bbox_abs = denormalize_bbox(bbox_norm.tolist(), orig_w, orig_h)
                x, y, w, h = bbox_abs

                results.append({
                    "query_id": query_ids[i],
                    "query_text": metas[i]["query_text"],
                    "pred_x": x,
                    "pred_y": y,
                    "pred_w": w,
                    "pred_h": h,
                })

    # Save to CSV
    df = pd.DataFrame(
        results,
        columns=["query_id", "query_text", "pred_x", "pred_y", "pred_w", "pred_h"],
    )
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')

    print(f"\n✅ Submission saved to: {output_csv}")
    print(f"📊 Total predictions: {len(results)}")


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description="FINAL Test Inference with TTA & Ensemble")

    parser.add_argument("--test_dir", type=str, default="./data/test", help="Test data directory")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints_final/best_model.pt",
                       help="Model checkpoint path")
    parser.add_argument("--ensemble_checkpoints", type=str, nargs='+', default=None,
                       help="Additional checkpoints for ensemble")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--output_csv", type=str, default="./submission_final.csv", help="Output CSV file")
    parser.add_argument("--img_size", type=int, default=512, help="Input image size")
    parser.add_argument("--enable_tta", action="store_true", default=True, help="Enable TTA")
    parser.add_argument("--disable_tta", action="store_true", help="Disable TTA (faster)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    seed_everything(args.seed)

    enable_tta = args.enable_tta and not args.disable_tta

    print("="*70)
    print("🚀 FINAL OPTIMIZED INFERENCE - Maximum IoU Performance")
    print("="*70)
    print(f"Device: {DEVICE}")
    print(f"Test Directory: {args.test_dir}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Batch Size: {args.batch_size}")
    print(f"TTA Enabled: {enable_tta}")

    # Prepare model checkpoints
    checkpoint_paths = [args.checkpoint]
    if args.ensemble_checkpoints:
        checkpoint_paths.extend(args.ensemble_checkpoints)
        print(f"Ensemble Models: {len(checkpoint_paths)}")
    print("="*70)

    # Load models
    print("\n🧠 Loading models...")
    models, vocab = load_ensemble_models(checkpoint_paths, DEVICE)

    # Create test dataset
    print("\n📂 Loading test data...")
    test_ds = TestDataset(args.test_dir, vocab, img_size=args.img_size)

    if len(test_ds) == 0:
        print("❌ No test samples found!")
        return

    # Create data loader
    loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn_test,
    )

    # Run inference
    print("\n🔮 Running inference...")
    run_inference(models, loader, args.output_csv, DEVICE, enable_tta)

    print("\n" + "="*70)
    print("🎉 Inference Complete!")
    print("="*70)
    print(f"✅ Submission file ready: {args.output_csv}")
    print("👉 Upload this file to the competition platform")
    print("="*70)


if __name__ == "__main__":
    main()
