from openai import OpenAI
import pandas as pd
from data_txt.cifar100_label_mapping import get_readable_name
import csv
import argparse
import os

# --------------------
# arguments
# --------------------
parser = argparse.ArgumentParser(description='LLM-based description extension')

parser.add_argument(
    '-exi', '--existing_description_path',
    default='descriptions_data/cifar100lt_existing.csv',  # <-- 네가 만든 파일 이름에 맞춤
    type=str,
    help='File path to the existing description file'
)
parser.add_argument(
    '-m', '--max_generate_num',
    default=30,  # 클래스당 최종 설명 개수 (existing + 새로 생성)
    type=int,
    help='Maximum number of descriptions per class (including existing ones)'
)
parser.add_argument(
    '-ext', '--extended_description_path',
    default='descriptions_data/cifar100lt_extended.csv',
    type=str,
    help='File path to the extended description file'
)
args = parser.parse_args()

# --------------------
# OpenAI client
# --------------------
# env에 OPENAI_API_KEY 설정해 두고 그냥 OpenAI()만 쓰는 게 안전
# export OPENAI_API_KEY="sk-..."
client = OpenAI(api_key='')

# --------------------
# load existing descriptions
# --------------------
# lmm_i2t.py에서 저장한 CSV는:
# image_path,label,description
df = pd.read_csv(args.existing_description_path)

# label별 description 리스트 / 텍스트 묶기
grouped_texts = (
    df.groupby('label')['description']
      .apply(lambda x: '\n'.join(x))
      .to_dict()
)
grouped_list = (
    df.groupby('label')['description']
      .apply(list)
      .to_dict()
)

# extended csv 초기화: 헤더 한 번 쓰고 시작
os.makedirs(os.path.dirname(args.extended_description_path), exist_ok=True)
with open(args.extended_description_path, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['label', 'description'])  # header

# append 모드로 다시 열기
ext_file = open(args.extended_description_path, mode='a', newline='')
writer = csv.writer(ext_file)

for label, text in grouped_texts.items():
    label_int = int(label)
    current_all_description = grouped_list[label]  # 이미 existing에 있던 문장 리스트

    real_name = get_readable_name(label_int).split(", ")[0]

    system_content = (
        "You will follow the Template to describe the object. "
        "Template: A photo of the class " + real_name +
        " {with distinctive features}{in specific scenes}."
    )

    # self-reflection 루프
    while len(current_all_description) < args.max_generate_num:
        # 지금까지의 모든 설명을 컨텍스트로 넣어줌
        current_description = "\n".join(current_all_description)

        user_content = (
            "Besides these descriptions mentioned above, "
            "please use the same Template to list other possible "
            "{distinctive features} and {specific scenes} for the class "
            + real_name
        )

        completion = client.chat.completions.create(
            # 모델은 너가 쓰는 걸로 (요즘은 gpt-4.1, gpt-4o, gpt-4.1-mini 등)
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": current_description},
                {"role": "user", "content": user_content}
            ]
        )

        output = completion.choices[0].message.content.strip()

        # bullet / 줄바꿈 포맷에 따라 분리
        if '\n- ' in output:
            sentences = [s.strip('- ').strip() for s in output.split('\n') if s.strip()]
        elif '\n\n- ' in output:
            sentences = [s.strip('- ').strip() for s in output.split('\n\n') if s.strip()]
        elif '\n\n' in output:
            sentences = [s.strip() for s in output.split('\n\n') if s.strip()]
        elif '\n' in output:
            sentences = [s.strip() for s in output.split('\n') if s.strip()]
        else:
            sentences = [output] if output else []

        # 혹시 파싱 실패하면 루프 탈출 (무한 루프 방지)
        if len(sentences) == 0:
            break

        # 이미 있는 문장과 중복 제거
        new_sentences = []
        existing_set = set([s.strip() for s in current_all_description])
        for s in sentences:
            if s not in existing_set:
                new_sentences.append(s)
                existing_set.add(s)

        if len(new_sentences) == 0:
            # 새로 추가된 문장이 없으면 더 이상 늘어나지 않으니까 break
            break

        # 현재 클래스 설명 리스트에 추가
        current_all_description.extend(new_sentences)

        # CSV에 기록
        for s in new_sentences:
            writer.writerow([label_int, s])

        # max_generate_num 넘었으면 정리하고 break
        if len(current_all_description) >= args.max_generate_num:
            break

ext_file.close()
print(f"Extended descriptions saved to {args.extended_description_path}")
