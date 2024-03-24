from kafka_classes.consumer import MyConsumer
from config import bootstrap_servers, preprocessing_topic
import numpy as np
import pickle as pkl
from config import save_preprocessing_dir
import os


def preprocessing():
    consumer = MyConsumer(bootstrap_servers, preprocessing_topic)
    while True:
        data = consumer.poll()
        i, img, y = data["img"]
        img = np.array(img)

        preprocessed_img = np.transpose(img / 255, (2, 0, 1))
        preprocessed_y = 1 if y == 1 else 0
        
        with open(os.path.join(save_preprocessing_dir, f"{i}.pkl"), "wb") as f: 
            pkl.dump((preprocessed_img, preprocessed_y), f)
        


if __name__ == "__main__":
    preprocessing()
