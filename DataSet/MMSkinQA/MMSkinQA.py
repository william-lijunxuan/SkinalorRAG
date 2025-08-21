import pandas as pd
import json
import datetime
import os

input_path = '/home/william/dataset/skin/MM-SkinQA/VQA.csv'

timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
output_dir = '/home/william/dataset/skin/MM-SkinQA'
output_path = os.path.join(output_dir, f"MM-SkinQA_{timestamp}.json")


os.makedirs(output_dir, exist_ok=True)


df = pd.read_csv(input_path)


records = []
for _, row in df.iterrows():
    image_name = row['image'].replace('dataset/', '', 1)
    records.append({
        "image_name": image_name,
        "question": row['question'],
        "answer": row['answer'],
        "question_type": "open"
    })


with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(records, f, ensure_ascii=False, indent=2)

print(f"save file：{output_path}")
