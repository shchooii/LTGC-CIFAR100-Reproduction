# draw_t2i.py (CIFAR-100-LT + LTGC 스타일)

import os
import argparse
import pandas as pd
from tqdm import tqdm

from dalle_gen import dalle_gen, description_refine, get_cls_template
from data_txt.cifar100_label_mapping import get_readable_name
from clip_filter import clip_filter   # 레포에 이미 있는 함수라고 가정


# --------------------
# arguments
# --------------------
parser = argparse.ArgumentParser(description="Generate synthetic images from extended descriptions")

parser.add_argument(
    "-ext", "--extended_description_path",
    default="descriptions_data/cifar100lt_extended.csv",
    type=str,
    help="Path to extended description CSV (label,description)"
)
parser.add_argument(
    "-d", "--data_dir",
    default="data/CIFAR100_LT",
    type=str,
    help="Base directory for CIFAR100_LT (train/test/gen_train)"
)
parser.add_argument(
    "--per_class",
    default=20,
    type=int,
    help="Max number of generated images per class"
)
parser.add_argument(
    "-t", "--thresh",
    default=0.6,
    type=float,
    help="Threshold for CLIP filter (similarity score)"
)
parser.add_argument(
    "-r", "--max_rounds",
    default=2,
    type=int,
    help="Max refinement rounds per image"
)

args = parser.parse_args()


# --------------------
# load extended descriptions
# --------------------
# llm_extension.py에서 label,description 형태로 저장했다고 가정
df = pd.read_csv(args.extended_description_path)

if "label" not in df.columns or "description" not in df.columns:
    # 혹시 헤더 없이 저장된 경우를 대비한 fallback
    df = pd.read_csv(args.extended_description_path, header=None, names=["label", "description"])

grouped_list = df.groupby("label")["description"].apply(list).to_dict()

out_root = os.path.join(args.data_dir, "gen_train")
os.makedirs(out_root, exist_ok=True)

START_LABEL = 80
print(f"=== USING START_LABEL = {START_LABEL} ===")
print(f"=== grouped_list labels: {sorted(map(int, grouped_list.keys()))[:10]} ... ===")

for label, texts in sorted(grouped_list.items(), key=lambda kv: int(kv[0])):
    label_int = int(label)

    if label_int < START_LABEL:
        print(f"[SKIP] label {label_int} < START_LABEL({START_LABEL})")
        continue

    print(f"[RUN ] label {label_int} >= START_LABEL({START_LABEL})")
    cls_name = get_readable_name(label_int).split(", ")[0]

    save_dir = os.path.join(out_root, str(label_int))
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n[Class {label_int}] {cls_name} - {len(texts)} descriptions")

    # 클래스 템플릿 (CLIP filter용)
    cls_feature_template = get_cls_template(label_int)

    used = 0
    pbar = tqdm(texts, desc=f"Label {label_int}")

    for text_i, text in enumerate(pbar):
        if used >= args.per_class:
            break

        # 1차: 원래 description으로 이미지 생성
        fname = os.path.join(save_dir, f"{label_int}_{text_i}.png")
        img_path = dalle_gen(text, fname, saved=True)
        if img_path is None:
            continue

        # CLIP filter
        score = clip_filter(img_path, cls_feature_template)

        # 필요하면 몇 번까지 refine
        round_idx = 0
        while score < args.thresh and round_idx < args.max_rounds:
            refined_text = description_refine(text, cls_name)
            refine_fname = os.path.join(save_dir, f"{label_int}_{text_i}_refine{round_idx}.png")
            img_path_refine = dalle_gen(refined_text, refine_fname, saved=True)
            if img_path_refine is None:
                break
            score = clip_filter(img_path_refine, cls_feature_template)
            round_idx += 1

        if score >= args.thresh:
            used += 1
