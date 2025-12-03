# dalle_gen.py (CIFAR-100-LT 버전)

from openai import OpenAI
from data_txt.cifar100_label_mapping import get_readable_name
import base64
from io import BytesIO
from PIL import Image
import os
import requests

client = OpenAI(api_key='')


def dalle_gen(input_text, saved_path, size="256x256", saved=True):
    """
    DALL·E 2를 사용해서 이미지 생성하고, 32x32로 리사이즈해서 저장.
    """
    try:
        if len(input_text) > 1000:
            input_text = input_text[:1000]

        # ✅ gpt-image-1 말고 dall-e-2 사용
        response = client.images.generate(
            model="dall-e-2",
            prompt=input_text,
            size=size,   # 256x256 / 512x512 / 1024x1024 가능
            n=1,
        )

        image_url = response.data[0].url
        r = requests.get(image_url)
        if r.status_code != 200:
            print(f"Failed to download image: HTTP {r.status_code}")
            return None

        img = Image.open(BytesIO(r.content)).convert("RGB")

        # CIFAR 사이즈로 줄이기
        img = img.resize((32, 32))

        if saved:
            os.makedirs(os.path.dirname(saved_path), exist_ok=True)
            img.save(saved_path)
            print(f"Saved to {saved_path}")

        return saved_path

    except Exception as e:
        print(f"An error occurred in dalle_gen: {e}")
        return None


def description_refine(input_text, cls_name):
    """
    CLIP 점수 낮을 때, 설명을 해당 클래스답게 refine
    """
    user_content = (
        "This description does not seem to be representative of the class "
        + cls_name +
        ". Could you refine it to enhance the distinctive features of class "
        + cls_name
    )

    completion = client.chat.completions.create(
        model="gpt-4o-mini",  # 비용 줄이려면 mini 계열
        messages=[
            {"role": "user", "content": input_text},
            {"role": "user", "content": user_content}
        ]
    )

    output = completion.choices[0].message.content
    return output


def get_cls_template(cls_index, filename="data_txt/CIFAR100_LT/class_templates.txt"):
    """
    CIFAR-100 클래스용 템플릿.
    draw_t2i.py에서 get_cls_template(label_int) 이렇게 1개 인자로 부르도록 맞춤.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # 이미 저장된 템플릿 있으면 먼저 찾기
    if os.path.exists(filename):
        with open(filename, "r") as file:
            for line in file:
                index, saved_template = line.strip().split(":", 1)
                if int(index) == int(cls_index):
                    return saved_template

    # 없으면 새로 생성
    cls_name = get_readable_name(int(cls_index)).split(", ")[0]

    template = (
        "Template: A photo of the class " + cls_name +
        " with {feature 1}{feature 2}{...}."
    )
    user_content = (
        "Please use the Template to summarize the most distinctive features of class "
        + cls_name
    )

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": template},
            {"role": "user", "content": user_content}
        ]
    )

    output = completion.choices[0].message.content.strip()

    # 새 템플릿 캐시
    with open(filename, "a") as file:
        file.write(f"{int(cls_index)}:{output}\n")

    return output
