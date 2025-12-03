# make_cifar100lt_class_count.py
import os
import json
from collections import Counter

def main():
    train_txt = "data_txt/CIFAR100_LT/CIFAR100_LT_train.txt"
    out_json = "data_txt/CIFAR100_LT/cifar100lt_class_count.json"

    if not os.path.exists(train_txt):
        raise FileNotFoundError(f"{train_txt} 가 없습니다. 경로를 확인하세요.")

    counts = Counter()

    with open(train_txt, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 형식: "train/xxx.png  63"
            parts = line.split()
            if len(parts) < 2:
                continue
            label = int(parts[1])
            counts[label] += 1

    num_classes = max(counts.keys()) + 1

    class_counts = [0] * num_classes
    for cls_idx, cnt in counts.items():
        class_counts[cls_idx] = cnt

    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(
            {
                "num_classes": num_classes,
                "class_counts": class_counts,
            },
            f,
            indent=2
        )

    print(f"Saved class counts for {num_classes} classes to {out_json}")
    print("class_counts:", class_counts)

if __name__ == "__main__":
    main()
