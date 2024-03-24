import pandas as pd
import os
import cv2
import json
from producer import MyProducer
from config import bootstrap_servers, preprocessing_topic, annotation_path, img_dir


def read_data(annotation_path, img_dir):
    producer = MyProducer(bootstrap_servers, preprocessing_topic)
    df = pd.read_csv(annotation_path).loc[:, ['image_id', 'Smiling']]
    for i, record in df.iterrows():
        img_name = record["image_id"]
        smiling = record["Smiling"]

        image_path = os.path.join(img_dir, img_name)
        y = smiling
        img = cv2.imread(image_path).tolist()
        data_dict = {"img": (i, img, y)}
        producer.produce_message('1', json.dumps(data_dict))
        producer.flush()

    # 162770
    

if __name__ == "__main__":
    read_data(annotation_path, img_dir)