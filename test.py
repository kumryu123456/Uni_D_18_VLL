import os
import json
import re
from glob import glob
from typing import Dict, Any, List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from tqdm import tqdm

from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoModelForCausalLM,
)

# ==============================
# 기본 설정
# ==============================
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODEL_ID = "microsoft/Florence-2-large-ft"  # 학습 때 썼던 base 모델 ID


# ==============================
# 유틸 함수들
# ==============================
def seed_everything(seed: int = 42):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_image_path(json_path: str, data: Dict[str, Any], jpg_root: str) -> str:
    """
    MI3 json → MI2 jpg 경로 찾기.
    train에서 사용하던 것과 동일한 로직을 test에 맞게 재사용.
    - jpg_root: ./data/test/images  같은 디렉토리
    """
    src = data.get("source_data_info", {})
    jpg_name = src.get("source_data_name_jpg", None)

    # case 1: 메타데이터에 jpg_name이 명시되어 있는 경우
    if jpg_name:
        cand = os.path.join(jpg_root, jpg_name)
        if os.path.exists(cand):
            return cand

        # json 경로 기준으로 json → jpg 폴더 치환 (혹시 구조가 비슷한 경우 대비)
        maybe = json_path.replace(os.sep + "json" + os.sep, os.sep + "jpg" + os.sep)
        maybe = os.path.join(os.path.dirname(maybe), jpg_name)
        if os.path.exists(maybe):
            return maybe

        # MI3 → MI2 이름만 바꿔서 시도
        base = os.path.basename(json_path)  # 예: MI3_000001.json
        jpg_base = base.replace("MI3", "MI2").rsplit(".", 1)[0] + ".jpg"
        sibling = os.path.join(jpg_root, jpg_base)
        if os.path.exists(sibling):
            return sibling

    # case 2: 메타데이터가 없으면 파일명 기반으로 매칭
    base = os.path.basename(json_path)
    stem = os.path.splitext(base)[0]  # 예: 000001.json → 000001
    cand1 = os.path.join(jpg_root, stem + ".jpg")
    cand2 = os.path.join(jpg_root, stem.replace("MI3", "MI2") + ".jpg")

    if os.path.exists(cand1):
        return cand1
    if os.path.exists(cand2):
        return cand2

    raise FileNotFoundError(f"[get_image_path] JPG not found for json={json_path}")


def parse_florence_output_to_bbox(
    text: str, img_w: int, img_h: int
) -> Tuple[float, float, float, float]:
    """
    Florence-2 출력에서 <loc_?> 토큰 4개를 파싱해서
    실제 이미지 좌표계(x, y, w, h)로 변환.
    """
    matches = re.findall(r"<loc_(\d+)>", text)
    if len(matches) < 4:
        # 실패 시: 이미지 중앙에 적당한 박스
        return img_w / 4, img_h / 4, img_w / 2, img_h / 2

    lx1, ly1, lx2, ly2 = map(int, matches[:4])
    x1 = lx1 / 999 * img_w
    y1 = ly1 / 999 * img_h
    x2 = lx2 / 999 * img_w
    y2 = ly2 / 999 * img_h
    return x1, y1, x2 - x1, y2 - y1


def is_visual_ann(a: Dict[str, Any]) -> bool:
    """
    train에서 쓰던 것과 비슷하게,
    차트/표 등 + visual_instruction 있는 것만 사용.
    """
    cid = str(a.get("class_id", "") or "")
    cname = str(a.get("class_name", "") or "")
    has_q = bool(str(a.get("visual_instruction", "") or "").strip())
    looks_visual = cid.startswith("V") or any(
        k in cname for k in ["표", "차트", "그래프", "chart", "table"]
    )
    return has_q and looks_visual


# ==============================
# 모델 로더
# ==============================
def load_finetuned_model(model_dir: str):
    """
    - config/구조는 항상 base MODEL_ID에서 가져오고
    - weight만 model_dir(checkpoint)에서 로드한다.
    이렇게 해야 vision_config assertion 에러를 피할 수 있음.
    """
    print(f"[load_finetuned_model] base model: {MODEL_ID}")
    print(f"[load_finetuned_model] finetuned weights from: {model_dir}")

    # 1) base config + 모델 구조
    base_config = AutoConfig.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        config=base_config,
        trust_remote_code=True,
        torch_dtype=torch.float32,  # 추론은 fp32로 안전하게
    ).to(DEVICE)

    # 2) fine-tuned 가중치 로드
    weight_path_bin = os.path.join(model_dir, "pytorch_model.bin")
    weight_path_sf = os.path.join(model_dir, "model.safetensors")

    if os.path.exists(weight_path_bin):
        state_dict = torch.load(weight_path_bin, map_location="cpu")
        print(f"  - loaded weights from {weight_path_bin}")
    elif os.path.exists(weight_path_sf):
        from safetensors.torch import load_file

        state_dict = load_file(weight_path_sf)
        print(f"  - loaded weights from {weight_path_sf}")
    else:
        raise FileNotFoundError(
            f"No weights found in {model_dir} (expected pytorch_model.bin or model.safetensors)"
        )

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"  - missing keys   : {len(missing)}")
    print(f"  - unexpected keys: {len(unexpected)}")

    # 3) processor는 fine-tune 시 저장한 디렉토리에서 로드
    processor = AutoProcessor.from_pretrained(
        model_dir,
        trust_remote_code=True,
    )

    return model, processor


# ==============================
# Test Dataset
# ==============================
class FlorenceTestDataset(Dataset):
    """
    test 디렉토리 구조 가정:
      data/test/
        ├─ query/   : *.json (MI3_....json)
        └─ images/  : *.jpg  (MI2_....jpg)

    - json_path + get_image_path()로 이미지 경로를 찾는다.
    - query_id  : annotation.instance_id
    - query_text: annotation.visual_instruction
    """

    def __init__(self, test_dir: str):
        json_dir = os.path.join(test_dir, "query")
        jpg_root = os.path.join(test_dir, "images")

        json_files = sorted(glob(os.path.join(json_dir, "*.json")))
        self.samples: List[Dict[str, Any]] = []

        if not json_files:
            print(f"[FlorenceTestDataset] No json files found in {json_dir}")

        for jf in json_files:
            try:
                data = read_json(jf)
            except Exception as e:
                print(f"[WARN] failed to read {jf}: {e}")
                continue

            # 이미지 경로 (json 하나당 이미지 하나라고 가정)
            try:
                img_path = get_image_path(jf, data, jpg_root)
            except FileNotFoundError as e:
                # print(f"[WARN] {e}")
                continue

            # annotation 리스트에서 visual한 것만 사용
            ann_list = (
                data.get("learning_data_info", {})
                .get("annotation", [])
            )

            for ann in ann_list:
                if not is_visual_ann(ann):
                    continue

                instance_id = str(ann.get("instance_id", "") or "").strip()
                qtext = str(ann.get("visual_instruction", "") or "").strip()

                if not instance_id or not qtext:
                    continue

                self.samples.append(
                    {
                        "query_id": instance_id,          # 🔹 CSV용
                        "query_text": qtext,              # 🔹 CSV용
                        "question_for_model": "<CAPTION_TO_PHRASE_GROUNDING>" + qtext,
                        "image_path": img_path,
                    }
                )

        print(f"[TestDataset] Loaded {len(self.samples)} items")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = Image.open(s["image_path"]).convert("RGB")
        img_w, img_h = img.size

        meta = {
            "img_size": (img_w, img_h),
            "image_path": s["image_path"],
            "query_text": s["query_text"],
        }

        # 모델 입력: (query_id, question_for_model, image, meta)
        return s["query_id"], s["question_for_model"], img, meta


def make_collate_fn(processor):
    def collate_fn(batch):
        qids, questions, images, metas = zip(*batch)
        inputs = processor(
            text=list(questions),
            images=list(images),
            return_tensors="pt",
            padding=True,
        )
        return list(qids), inputs, list(metas)

    return collate_fn


# ==============================
# Inference 루프
# ==============================
def run_test(model, processor, loader: DataLoader, output_csv: str = "./submission.csv"):
    model.eval()
    results = []

    with torch.no_grad():
        pbar = tqdm(loader, desc="Test inference")

        for qids, inputs, metas in pbar:
            # device로 올리기
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(DEVICE)

            # pixel_values dtype 맞추기
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)

            # Florence-2 generate
            gen_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=32,
                num_beams=3,
                do_sample=False,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
            )

            texts = processor.batch_decode(gen_ids, skip_special_tokens=False)

            for qid, text, meta in zip(qids, texts, metas):
                img_w, img_h = meta["img_size"]
                x, y, w, h = parse_florence_output_to_bbox(text, img_w, img_h)

                results.append(
                    {
                        "query_id": qid,
                        "query_text": meta["query_text"],  # 🔹 CSV에 visual_instruction 그대로
                        "pred_x": x,
                        "pred_y": y,
                        "pred_w": w,
                        "pred_h": h,
                    }
                )

    # 컬럼 순서 명시
    df = pd.DataFrame(
        results,
        columns=["query_id", "query_text", "pred_x", "pred_y", "pred_w", "pred_h"],
    )
    df.to_csv(output_csv, index=False)
    print(f"[Done] Saved submission to {output_csv}")


# ==============================
# main
# ==============================
def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", type=str, default="./data/test")
    parser.add_argument("--model_dir", type=str, default="./outputs/florence2_bbox/best")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output_csv", type=str, default="./my_submission.csv")
    args = parser.parse_args()

    seed_everything(42)

    # 1) 모델 / 프로세서 로드
    model, processor = load_finetuned_model(args.model_dir)

    # 2) Dataset / DataLoader
    test_ds = FlorenceTestDataset(args.test_dir)
    loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=make_collate_fn(processor),
    )

    # 3) Inference & CSV 저장
    run_test(model, processor, loader, args.output_csv)


if __name__ == "__main__":
    main()