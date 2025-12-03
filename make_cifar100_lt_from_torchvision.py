# make_cifar100_lt_from_torchvision.py
import os
import json
from collections import Counter

import numpy as np
from PIL import Image
from torchvision.datasets import CIFAR100

"""
원본 CIFAR-100(train 50000, test 10000)을 받아서
imbalance ratio r로 CIFAR-100-LT를 만들고,
LTGC에서 쓰는 포맷으로 (이미지 + txt + class_count.json) 저장하는 스크립트
"""

def get_img_num_per_cls(data_length, cls_num, imb_type='exp', imb_factor=1/100):
    img_max = data_length / cls_num  # 50000 / 100 = 500
    img_num_per_cls = []
    if imb_type == 'exp':
        for cls_idx in range(cls_num):
            num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
            img_num_per_cls.append(int(num))
    elif imb_type == 'step':
        for cls_idx in range(cls_num // 2):
            img_num_per_cls.append(int(img_max))
        for cls_idx in range(cls_num // 2):
            img_num_per_cls.append(int(img_max * imb_factor))
    else:  # 'none'
        img_num_per_cls.extend([int(img_max)] * cls_num)
    return img_num_per_cls


def gen_imbalanced_indices(targets, img_num_per_cls):
    targets_np = np.array(targets, dtype=np.int64)
    classes = np.unique(targets_np)

    new_indices = []
    num_per_cls_dict = {}

    for the_class, the_img_num in zip(classes, img_num_per_cls):
        idx = np.where(targets_np == the_class)[0]
        np.random.shuffle(idx)
        selec_idx = idx[:the_img_num]
        new_indices.extend(selec_idx.tolist())
        num_per_cls_dict[int(the_class)] = int(the_img_num)

    return new_indices, num_per_cls_dict


def save_split(dataset, indices, split_name, root_dir, txt_path, count_counter=None):
    os.makedirs(root_dir, exist_ok=True)
    os.makedirs(os.path.dirname(txt_path), exist_ok=True)

    lines = []

    for new_i, orig_idx in enumerate(indices):
        img, label = dataset[orig_idx]  # img: PIL.Image (transform=None이면)

        if not isinstance(img, Image.Image):
            # 혹시 transform 걸려 있으면 여기서 PIL로 변환
            from torchvision.transforms.functional import to_pil_image
            img = to_pil_image(img)

        filename = f"{new_i:05d}.png"
        rel_path = f"{split_name}/{filename}"
        abs_path = os.path.join(root_dir, rel_path)

        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        img.save(abs_path)

        label = int(label)
        lines.append(f"{rel_path} {label}\n")

        if count_counter is not None:
            count_counter[label] += 1

    with open(txt_path, "w") as f:
        f.writelines(lines)

    print(f"[{split_name}] saved {len(lines)} samples to {txt_path}")


def main():
    # ==========================
    # 1) 설정
    # ==========================
    data_root_raw = "./data_raw"   # CIFAR100 원본 다운로드 위치
    out_img_root = "data/CIFAR100_LT"
    txt_dir = "data_txt/CIFAR100_LT"

    imb_type = "exp"
    imb_factor = 1 / 100  # r-100 (tail 제일 심한 버전)
    cls_num = 100

    os.makedirs(txt_dir, exist_ok=True)

    # ==========================
    # 2) 원본 CIFAR-100 다운로드
    # ==========================
    print("Downloading CIFAR-100 (if not exists)...")
    train_dataset = CIFAR100(root=data_root_raw, train=True, download=True, transform=None)
    test_dataset = CIFAR100(root=data_root_raw, train=False, download=True, transform=None)

    targets = train_dataset.targets  # 길이 50000, 라벨 0~99
    data_length = len(targets)

    # ==========================
    # 3) long-tail per-class 개수 계산 + 인덱스 샘플링
    # ==========================
    img_num_per_cls = get_img_num_per_cls(
        data_length=data_length,
        cls_num=cls_num,
        imb_type=imb_type,
        imb_factor=imb_factor,
    )
    indices_lt, num_per_cls_dict = gen_imbalanced_indices(targets, img_num_per_cls)

    print("img_num_per_cls (train LT):", img_num_per_cls)
    print("Total selected train samples:", len(indices_lt))

    # ==========================
    # 4) train LT 이미지 + txt 저장
    # ==========================
    from collections import Counter
    train_counter = Counter()

    train_txt_path = os.path.join(txt_dir, "CIFAR100_LT_train.txt")
    save_split(
        dataset=train_dataset,
        indices=indices_lt,
        split_name="train",
        root_dir=out_img_root,
        txt_path=train_txt_path,
        count_counter=train_counter,
    )

    # ==========================
    # 5) test는 전체 balanced 세트 그대로 사용
    # ==========================
    test_indices = list(range(len(test_dataset)))
    test_txt_path = os.path.join(txt_dir, "CIFAR100_LT_test.txt")
    save_split(
        dataset=test_dataset,
        indices=test_indices,
        split_name="test",
        root_dir=out_img_root,
        txt_path=test_txt_path,
        count_counter=None,
    )

    # ==========================
    # 6) class_count.json 저장 (train 기준)
    # ==========================
    num_classes = cls_num
    class_counts = [0] * num_classes
    for cls_idx in range(num_classes):
        class_counts[cls_idx] = int(train_counter[cls_idx])

    out_json = os.path.join(txt_dir, "cifar100lt_class_count.json")
    with open(out_json, "w") as f:
        json.dump(
            {"num_classes": num_classes, "class_counts": class_counts},
            f,
            indent=2
        )

    print(f"Saved class counts to {out_json}")
    print("class_counts:", class_counts)


if __name__ == "__main__":
    main()
