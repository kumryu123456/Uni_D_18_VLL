import os
import json
import random
from glob import glob
from typing import List, Dict, Any, Tuple

from PIL import Image
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    get_scheduler,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_ID = "microsoft/Florence-2-large-ft"

IMAGE_DEFAULT_W, IMAGE_DEFAULT_H = 2480, 3508  # 문서 해상도 기본값


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_category_from_ann(ann: Dict[str, Any]) -> str:

    cname = str(ann.get("class_name", "") or "").lower()
    cid = str(ann.get("class_id", "") or "")

    # 표
    if "표" in cname or "table" in cname:
        return "table"

    # 차트 계열
    if "차트" in cname or "chart" in cname or "그래프" in cname or "graph" in cname:
        if "꺾은선" in cname or "line" in cname:
            return "chart_line"
        if "세로 막대" in cname:
            return "chart_bar_v"
        if "가로 막대" in cname:
            return "chart_bar_h"
        if "원형" in cname or "pie" in cname:
            return "chart_pie"
        if "영역형" in cname or "area" in cname:
            return "chart_area"
        if "혼합형" in cname or "mixed" in cname:
            return "chart_mixed"
        if "분산형" in cname or "scatter" in cname:
            return "chart_scatter"
        if "방사형" in cname or "radar" in cname:
            return "chart_radar"
        return "chart_other"

    if "다이어그램" in cname or cid.startswith("V03"):
        return "diagram"

    if cid.startswith("V"):
        return "visual_other"

    return "other"

def is_visual_ann(a: Dict[str, Any]) -> bool:
    cid = str(a.get("class_id", "") or "")
    cname = str(a.get("class_name", "") or "")
    has_q = bool(str(a.get("visual_instruction", "") or "").strip())
    looks_visual = cid.startswith("V") or any(
        k in cname for k in ["표", "차트", "그래프", "chart", "table"]
    )
    return has_q and looks_visual


def get_image_path(json_path: str, data: Dict[str, Any], jpg_root: str) -> str:
    src = data.get("source_data_info", {})
    jpg_name = src.get("source_data_name_jpg", None)

    if jpg_name:
        cand = os.path.join(jpg_root, jpg_name)
        if os.path.exists(cand):
            return cand

        # json 경로 기준으로 json → jpg 폴더 치환
        maybe = json_path.replace(os.sep + "json" + os.sep, os.sep + "jpg" + os.sep)
        maybe = os.path.join(os.path.dirname(maybe), jpg_name)
        if os.path.exists(maybe):
            return maybe

        base = os.path.basename(json_path)
        jpg_base = base.replace("MI3", "MI2").rsplit(".", 1)[0] + ".jpg"
        sibling = os.path.join(jpg_root, jpg_base)
        if os.path.exists(sibling):
            return sibling

    raise FileNotFoundError(f"[get_image_path] JPG not found for {json_path}")


def normalize_bbox_to_florence(bbox: List[float], img_w: int, img_h: int) -> str:
    x, y, w, h = bbox
    x1, y1 = x, y
    x2, y2 = x + w, y + h

    def to_loc(v, size):
        return max(0, min(999, int(v / size * 999)))

    lx1 = to_loc(x1, img_w)
    ly1 = to_loc(y1, img_h)
    lx2 = to_loc(x2, img_w)
    ly2 = to_loc(y2, img_h)

    return f"<loc_{lx1}><loc_{ly1}><loc_{lx2}><loc_{ly2}>"


class FlorenceBBoxDataset(Dataset):
    def __init__(
        self,
        json_dirs: List[str],
        jpg_dirs: List[str],
        max_samples: int | None = None,
        split_name: str = "train",
    ):
        self.samples = []

        assert len(json_dirs) == len(
            jpg_dirs
        ), "json_dirs와 jpg_dirs 길이가 같아야 합니다."

        total_invalid = 0

        for json_dir, jpg_dir in zip(json_dirs, jpg_dirs):
            if not os.path.isdir(json_dir):
                continue

            json_files = sorted(glob(os.path.join(json_dir, "*.json")))

            for jf in json_files:
                try:
                    data = read_json(jf)
                except Exception:
                    total_invalid += 1
                    continue

                ann_list = (
                    data.get("learning_data_info", {})
                    .get("annotation", [])
                )

                try:
                    img_path = get_image_path(jf, data, jpg_dir)
                except FileNotFoundError:
                    continue

                img_w, img_h = data.get(
                    "source_data_info", {}
                ).get("document_resolution", [IMAGE_DEFAULT_W, IMAGE_DEFAULT_H])

                for ann in ann_list:
                    if not is_visual_ann(ann):
                        continue

                    q = str(ann.get("visual_instruction", "")).strip()
                    bbox = ann.get("bounding_box", None)
                    if not bbox or len(bbox) < 4:
                        continue

                    loc_tokens = normalize_bbox_to_florence(
                        bbox[:4], img_w, img_h
                    )

                    category = get_category_from_ann(ann)

                    self.samples.append(
                        {
                            "image_path": img_path,
                            "question": "<CAPTION_TO_PHRASE_GROUNDING>" + q,
                            "answer": loc_tokens,
                            "bbox": bbox[:4],
                            "img_size": (img_w, img_h),
                            "instance_id": ann.get("instance_id", ""),
                            "category": category,
                        }
                    )

        print(f"Total samples ({split_name}): {len(self.samples)}")
        print(f"Total invalid json skipped: {total_invalid}")

        if max_samples is not None and len(self.samples) > max_samples:
            # 1) 카테고리별 버킷 만들기
            from collections import defaultdict
            buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            for s in self.samples:
                cat = s.get("category", "unknown")
                buckets[cat].append(s)

            # 2) 각 버킷 셔플
            rng = random.Random(42)
            for cat in buckets:
                rng.shuffle(buckets[cat])

            # 3) 라운드로빈으로 카테고리 균등하게 뽑기
            selected = []
            active_cats = [c for c in buckets.keys() if buckets[c]]

            while len(selected) < max_samples and active_cats:
                for cat in list(active_cats):
                    if not buckets[cat]:
                        active_cats.remove(cat)
                        continue
                    selected.append(buckets[cat].pop())
                    if len(selected) >= max_samples:
                        break

            self.samples = selected
            print(
                f"[{split_name}] Using balanced subset: {len(self.samples)} / max_samples={max_samples}"
            )

            # 4) 최종 카테고리 분포 로그
            cat_counts = {}
            for s in self.samples:
                c = s.get("category", "unknown")
                cat_counts[c] = cat_counts.get(c, 0) + 1
            print(f"[{split_name}] Category distribution in subset: {cat_counts}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        image = Image.open(s["image_path"]).convert("RGB")
        return (
            s["question"],
            s["answer"],
            image,
            {
                "bbox": s["bbox"],
                "img_size": s["img_size"],
                "instance_id": s["instance_id"],
                "image_path": s["image_path"],
            },
        )


def make_collate_fn(processor, device):
    def collate_fn(batch):
        questions, answers, images, metas = zip(*batch)

        inputs = processor(
            text=list(questions),
            images=list(images),
            return_tensors="pt",
            padding=True,
        )

        return inputs, list(answers), list(metas)

    return collate_fn


def load_model_and_processor(revision: str | None = None):
    print(f"Loading model: {MODEL_ID} (revision={revision})")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        revision=revision,
        torch_dtype=torch.float32,
    ).to(DEVICE)

    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        revision=revision,
    )

    # vision tower freeze (필요하면)
    if hasattr(model, "vision_tower"):
        for p in model.vision_tower.parameters():
            p.requires_grad = False

    return model, processor


import re


def parse_florence_output_to_bbox(text: str, img_w: int, img_h: int):
    matches = re.findall(r"<loc_(\d+)>", text)
    if len(matches) < 4:
        return img_w / 4, img_h / 4, img_w / 2, img_h / 2

    lx1, ly1, lx2, ly2 = map(int, matches[:4])
    x1 = lx1 / 999 * img_w
    y1 = ly1 / 999 * img_h
    x2 = lx2 / 999 * img_w
    y2 = ly2 / 999 * img_h
    return x1, y1, x2 - x1, y2 - y1


def iou_xywh(pred, gt):
    px, py, pw, ph = pred
    gx, gy, gw, gh = gt
    px1, py1, px2, py2 = px, py, px + pw, py + ph
    gx1, gy1, gx2, gy2 = gx, gy, gx + gw, gy + gh

    ix1, iy1 = max(px1, gx1), max(py1, gy1)
    ix2, iy2 = min(px2, gx2), min(py2, gy2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = pw * ph + gw * gh - inter
    return float(inter / union) if union > 0 else 0.0

def train_one_epoch(
    model,
    processor,
    dataloader,
    optimizer,
    scheduler,
    epoch: int,
):
    model.train()
    total_loss = 0.0

    pbar = tqdm(dataloader, desc=f"Train {epoch+1}")

    for step, (inputs, answers, metas) in enumerate(pbar, start=1):
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(DEVICE)

        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)

        seq_len = inputs["input_ids"].shape[1]

        answer_enc = processor.tokenizer(
            text=answers,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=seq_len,
            return_token_type_ids=False,
        )

        labels = answer_enc.input_ids.to(DEVICE)
        pad_id = processor.tokenizer.pad_token_id
        labels[labels == pad_id] = -100

        outputs = model(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            attention_mask=inputs.get("attention_mask", None),
            labels=labels,
        )
        loss = outputs.loss

        if not torch.isfinite(loss):
            print(f"!!! Non-finite loss detected: {loss.item()}")
            optimizer.zero_grad()
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        avg_loss = total_loss / step

        pbar.set_postfix(loss=f"{avg_loss:.4f}")

    return total_loss / len(dataloader)


def evaluate_miou(model, processor, dataloader):
    model.eval()
    ious = []
    pbar = tqdm(dataloader, desc="Eval mIoU")

    with torch.no_grad():
        for step, (inputs, answers, metas) in enumerate(pbar, start=1):
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(DEVICE)

            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)

            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=32,
                num_beams=3,
                do_sample=False,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
            )

            texts = processor.batch_decode(
                generated_ids, skip_special_tokens=False
            )

            for text, meta in zip(texts, metas):
                img_w, img_h = meta["img_size"]
                gt_box = meta["bbox"]
                pred_box = parse_florence_output_to_bbox(text, img_w, img_h)
                ious.append(iou_xywh(pred_box, gt_box))

            # ➜ 지금까지의 평균 mIoU를 실시간으로 표시
            if ious:
                cur_miou = float(np.mean(ious))
                pbar.set_postfix(mIoU=f"{cur_miou:.4f}")

    if not ious:
        return 0.0
    return float(np.mean(ious))


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)  # 기본 3 epoch 정도
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)  # 1e-4 → 5e-5로 살짝 낮춤
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--max_train_samples", type=int, default=8000)
    parser.add_argument("--max_valid_samples", type=int, default=2000)
    parser.add_argument(
        "--train_press_json", type=str, default="./data/train/press_json"
    )
    parser.add_argument(
        "--train_press_jpg", type=str, default="./data/train/press_jpg"
    )
    parser.add_argument(
        "--train_report_json", type=str, default="./data/train/report_json"
    )
    parser.add_argument(
        "--train_report_jpg", type=str, default="./data/train/report_jpg"
    )
    parser.add_argument(
        "--valid_press_json", type=str, default="./data/valid/press_json"
    )
    parser.add_argument(
        "--valid_press_jpg", type=str, default="./data/valid/press_jpg"
    )
    parser.add_argument(
        "--valid_report_json", type=str, default="./data/valid/report_json"
    )
    parser.add_argument(
        "--valid_report_jpg", type=str, default="./data/valid/report_jpg"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/florence2_bbox",
    )

    args = parser.parse_args()

    seed_everything(42)

    train_json_dirs = [args.train_press_json, args.train_report_json]
    train_jpg_dirs = [args.train_press_jpg, args.train_report_jpg]

    valid_json_dirs = [args.valid_press_json, args.valid_report_json]
    valid_jpg_dirs = [args.valid_press_jpg, args.valid_report_jpg]

    train_ds = FlorenceBBoxDataset(
        train_json_dirs,
        train_jpg_dirs,
        max_samples=args.max_train_samples,
        split_name="train",
    )
    valid_ds = FlorenceBBoxDataset(
        valid_json_dirs,
        valid_jpg_dirs,
        max_samples=args.max_valid_samples,
        split_name="valid",
    )

    # ===== 모델 & 프로세서 로드 =====
    model, processor = load_model_and_processor(args.revision)


    collate_fn = make_collate_fn(processor, DEVICE)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    # ===== Optim / Scheduler =====
    optimizer = AdamW(model.parameters(), lr=args.lr)
    num_training_steps = len(train_loader) * args.epochs
    warmup_steps = int(num_training_steps * 0.03)  # 전체 step의 3% 정도를 warmup

    scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    best_miou = 0.0

    # ===== Training Loop =====
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(
            model, processor, train_loader, optimizer, scheduler, epoch
        )
        print(f"[Epoch {epoch+1}] train loss = {train_loss:.4f}")

        miou = evaluate_miou(model, processor, valid_loader)
        print(f"[Epoch {epoch+1}] valid mIoU = {miou:.4f}")

        if miou > best_miou:
            best_miou = miou
            save_dir = os.path.join(args.output_dir, "best")
            os.makedirs(save_dir, exist_ok=True)
            model.save_pretrained(save_dir)
            processor.save_pretrained(save_dir)
            print(f"New best mIoU {best_miou:.4f} saved to {save_dir}")


if __name__ == "__main__":
    main()