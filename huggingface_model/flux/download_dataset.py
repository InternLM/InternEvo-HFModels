import torchvision.transforms as transforms

# 使用 read_parquet 加载parquet文件
import pandas as pd
import cv2
import json
import numpy as np
from pandas import read_parquet

data_path = [
    "/mnt/petrelfs/xiongyingtong/InternEvo-HFModels/huggingface_model/flux/origin_data/images_and_json_chunk_1.parquet",
    "/mnt/petrelfs/xiongyingtong/InternEvo-HFModels/huggingface_model/flux/origin_data/images_and_json_chunk_2.parquet"
]

dst_path = "/mnt/petrelfs/xiongyingtong/InternEvo-HFModels/huggingface_model/flux/data/"

count = 0

for i, path in enumerate(data_path):

    data = read_parquet(path)

    images = data['image']
    texts = data['json_data']

    for j in range(len(images)):
        image_bytes = images[j]['bytes']
        image_np = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        cv2.imwrite( dst_path + f"{count}.jpg", image_np)
        json_data = eval(texts[j])
        json_str = json.dumps(json_data)
        json_file_path = dst_path + f"{count}.json"
        with open(json_file_path, "w") as file:
            file.write(json_str)
        count = count + 1
    
print("completed!")

