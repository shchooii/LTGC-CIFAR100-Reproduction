import torch
# from torchvision import transforms
# from lt_dataloaders import ImageNetLTDataLoader
# from data_txt.imagenet_label_mapping import get_readable_name
from lt_dataloaders import CIFAR100LTDataLoader
from data_txt.cifar100_label_mapping import get_readable_name
from gpt4v import gpt4v_observe
from utils import sample_counter
import os
import json
import csv
import argparse
import csv
from openai import OpenAI
import json
import numpy as np


def get_tail_indices(class_count_path, num_tail=30):
    """
    클래스별 이미지 수 정보가 들어있는 json을 읽어서
    하위 num_tail개 클래스를 tail로 반환
    """
    with open(class_count_path, "r") as f:
        stats = json.load(f)

    class_counts = stats["class_counts"]  # 길이 100짜리 리스트라고 가정
    num_classes = stats["num_classes"]    # 100

    # 샘플 수 기준으로 오름차순 정렬한 클래스 인덱스
    sorted_idx = sorted(range(num_classes), key=lambda i: class_counts[i])

    tail_indices = sorted_idx[:num_tail]
    return tail_indices, class_counts

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--data_dir", type=str, default="data/CIFAR100_LT")
    parser.add_argument("-m", "--max_per_class", type=int, default=20)  # 클래스당 desc 최대 개수
    parser.add_argument("-f", "--class_count_file", type=str,
                        default="data_txt/CIFAR100_LT/cifar100lt_class_count.json")
    parser.add_argument("-exi", "--existing_out", type=str,
                        default="descriptions_data/cifar100lt_existing.csv")
    parser.add_argument("--num_tail", type=int, default=30)  # tail로 취급할 클래스 개수

    return parser.parse_args()


# imagenet_loader = ImageNetLTDataLoader(data_dir=args.data_dir, 
#                                        batch_size=1, 
#                                        shuffle=False, 
#                                        num_workers=4, 
#                                        training=True)


client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def describe_with_gpt4v(image_tensor, class_name):
    """
    image_tensor: (1, 3, H, W) 같은 PyTorch 텐서라고 가정
    class_name: "porcupine" 같은 문자열
    여기서는 pseudo-code. 실제로는 이미지를 파일로 저장한 뒤
    client.images.generate나 client.chat.completions.create(..., image=...) 형태로 호출.
    """
    prompt = (
        f"A photo of the class {class_name}, "
        "{with distinctive features}{in specific scenes}."
        " Please describe this image briefly following that template."
    )
    # 실제 코드에서는 image를 base64로 인코딩해서 vision 모델에 전달해야 함.
    # 여기선 로직 구조만 보여주기.
    completion = client.chat.completions.create(
        model="gpt-4o-mini",  # 또는 vision 모델
        messages=[
            {"role": "system", "content": "You are a vision-language assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    text = completion.choices[0].message.content
    return text



def main():
    args = parse_args()

    tail_indices, class_counts = get_tail_indices(
        args.class_count_file,
        num_tail=args.num_tail
    )
    print("Tail classes:", tail_indices)

    loader = CIFAR100LTDataLoader(
        data_dir=args.data_dir,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        training=True
    )

    desc_counter = {c: 0 for c in tail_indices}

    rows = []

    for batch in loader:
        if len(batch) == 3:
            images, labels, indices = batch
        else:
            images, labels = batch
            indices = None

        label = int(labels[0])
        if label not in tail_indices:
            continue  

        if desc_counter[label] >= args.max_per_class:
            continue

        class_name = get_readable_name(label)

        description = describe_with_gpt4v(images, class_name)

        img_path = loader.dataset.img_path[indices[0]] if indices is not None else ""

        rows.append([img_path, label, description])
        desc_counter[label] += 1

        if all(desc_counter[c] >= args.max_per_class for c in tail_indices):
            break

    # 4. CSV로 저장
    os.makedirs(os.path.dirname(args.existing_out), exist_ok=True)
    with open(args.existing_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "label", "description"])
        writer.writerows(rows)

    print(f"Saved {len(rows)} descriptions to {args.existing_out}")


if __name__ == "__main__":
    main()
    