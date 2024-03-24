from torch.utils.data import Dataset, DataLoader
import torch
import os
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tqdm import tqdm
from producer import MyProducer
import json
import time
from config import bootstrap_servers, save_preprocessing_dir, img_dir, metrics_topic
import pickle as pkl


class CelebaTrainDataset(Dataset):

    def __init__(self, data_dir, split_idx=162770):
        self.data_dir = data_dir
        self.split_idx = split_idx
        # self.df = pd.read_csv(annot_path).loc[:162770, ['image_id', 'Smiling']]

    def __len__(self):
        return len(os.listdir(self.data_dir)[:self.split_idx])

    def __getitem__(self, idx):
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()

        with open(os.path.join(save_preprocessing_dir, f"{idx}.pkl"), "rb") as f:
            data = pkl.load(f)

        return data


class CelebaTestDataset(Dataset):

    def __init__(self, data_dir, split_idx=162770):
        self.data_dir = data_dir
        self.split_idx = split_idx

    def __len__(self):
        return len(os.listdir(self.data_dir)[self.split_idx:])

    def __getitem__(self, idx):
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()
        offset_idx = idx + self.split_idx

        with open(os.path.join(save_preprocessing_dir, f"{offset_idx}.pkl"), "rb") as f:
            data = pkl.load(f)

        return data


if __name__ == "__main__":
    producer = MyProducer(bootstrap_servers, metrics_topic)
    celeba_train_dataset = CelebaTrainDataset(save_preprocessing_dir)
    celeba_test_dataset = CelebaTestDataset(save_preprocessing_dir)

    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.train()
    model = model.cuda()

    num_epochs = 100
    bs = 32
    lr = 0.00004
    momentum = 0.9
    train_eval_step = 10
    test_eval_step = 10

    train_dataloader = DataLoader(celeba_train_dataset,
                                  batch_size=bs,
                                  shuffle=True,
                                  num_workers=0)
    val_dataloader = DataLoader(celeba_test_dataset,
                                batch_size=bs,
                                shuffle=True,
                                num_workers=0)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_accuracy = 0
        train_eval_loss = 0
        train_eval_accuracy = 0
        i = 1
        produce_data = {}
        for imgs, labels in tqdm(train_dataloader):
            imgs = imgs.cuda().float()
            labels = labels.cuda()

            optimizer.zero_grad()

            outputs = model(imgs)

            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_accuracy += torch.sum(preds == labels.data)

            train_eval_loss += loss.item()
            train_eval_accuracy += torch.sum(preds == labels.data)

            if i % train_eval_step == 0:
                print(
                    f"Train: Loss: {train_eval_loss / (train_eval_step * bs)}, Accuracy: {train_eval_accuracy / (train_eval_step * bs)}"
                )
                produce_data["train_loss"] = train_eval_loss / (train_eval_step * bs)
                produce_data["train_accuracy"] = train_eval_accuracy / (train_eval_step * bs)
                train_eval_loss = 0
                train_eval_accuracy = 0

            if i % test_eval_step == 0:
                model.eval()
                test_eval_loss = 0
                test_eval_accuracy = 0
                with torch.no_grad():
                    for imgs, labels in val_dataloader:
                        imgs = imgs.cuda().float()
                        labels = labels.cuda()
                        outputs = model(imgs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                        test_eval_loss += loss.item()
                        test_eval_accuracy += torch.sum(preds == labels.data)
                print(
                    f"Test: Loss: {test_eval_loss / len(celeba_test_dataset)}, Accuracy: {test_eval_accuracy / len(celeba_test_dataset)}"
                )
                produce_data["test_loss"] = test_eval_loss / len(celeba_test_dataset)
                produce_data["test_accuracy"] = test_eval_accuracy / len(celeba_test_dataset)


                produce_data["train_accuracy"] = produce_data["train_accuracy"].item()
                produce_data["test_accuracy"] = produce_data["test_accuracy"].item()
                producer.produce_message('1', json.dumps(produce_data))
                producer.flush()
                time.sleep(1)
            i += 1

        epoch_accuracy = epoch_accuracy / len(celeba_train_dataset)
        print(
            f"Epoch #{epoch}, Loss: {epoch_loss}, Accuracy: {epoch_accuracy}")
