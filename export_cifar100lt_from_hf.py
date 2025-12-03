# export_cifar100lt_from_hf.py
import os
import json
from collections import Counter

from datasets import load_dataset

def save_split(ds_split, split_name, root_dir, txt_path, class_counter):
    """
    ds_split: HF datasets split (train or test)
    split_name: "train" or "test"
    root_dir: e.g. "data/CIFAR100_LT"
    txt_path: e.g. "data_txt/CIFAR100_LT/CIFAR100_LT_train.txt"
    class_counter: Counter to accumulate class counts
    """
    os.makedirs(root_dir, exist_ok=True)
    os.makedirs(os.path.dirname(txt_path), exist_ok=True)

    lines = []

    # HF Image feature -> PIL 쓰기 위해
    ds_split = ds_split.with_format("pil")

    for idx, example in enumerate(ds_split):
        img = example["img"]                    # PIL.Image
        label = int(example["fine_label"])      # 0~99

        filename = f"{idx:05d}.png"
        rel_path = f"{split_name}/{filename}"
        abs_path = os.path.join(root_dir, rel_path)

        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        img.save(abs_path)

        lines.append(f"{rel_path} {label}\n")
        class_counter[label] += 1

    with open(txt_path, "w") as f:
        f.writelines(lines)

    print(f"[{split_name}] saved {len(lines)} images and txt to {txt_path}")


def main():
    config_name = "r-100"  # r-10, r-20, r-50, r-100, r-200 중 택1

    print(f"Loading HuggingFace dataset tomas-gajarsky/cifar100-lt ({config_name})...")
    ds = load_dataset(
        "tomas-gajarsky/cifar100-lt",
        name=config_name,
        trust_remote_code=True,
    )

    root_dir = "data/CIFAR100_LT"
    txt_dir = "data_txt/CIFAR100_LT"
    os.makedirs(txt_dir, exist_ok=True)

    class_counter = Counter()

    save_split(
        ds_split=ds["train"],
        split_name="train",
        root_dir=root_dir,
        txt_path=os.path.join(txt_dir, "CIFAR100_LT_train.txt"),
        class_counter=class_counter,
    )

    save_split(
        ds_split=ds["test"],
        split_name="test",
        root_dir=root_dir,
        txt_path=os.path.join(txt_dir, "CIFAR100_LT_test.txt"),
        class_counter=Counter(),
    )

    num_classes = max(class_counter.keys()) + 1
    class_counts = [0] * num_classes
    for cls_idx, cnt in class_counter.items():
        class_counts[cls_idx] = cnt

    out_json = os.path.join(txt_dir, "cifar100lt_class_count.json")
    with open(out_json, "w") as f:
        json.dump(
            {"num_classes": num_classes, "class_counts": class_counts},
            f,
            indent=2
        )

    print(f"Saved class counts for {num_classes} classes to {out_json}")
    print("class_counts:", class_counts)


if __name__ == "__main__":
    main()
