# TODO: clean up unused imports
import numpy as np
import pickle
from PIL import Image
import time
import shutil
import random
import argparse
import os
import re
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch
from torchvision.models import resnet18
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from enum import Enum, verify, UNIQUE
from collections import Counter
from tqdm import tqdm
import sqlite3


db_path = '/ds/videos/opticalflow-BOBSL/ASL/state.db'

STATE_EXTRACTING = 1
STATE_COMPLETED = 2
STATE_FAILED = 3

def mark_features_as(state, cur, con, video_id, model_name, variant):
    assert_valid_model_name(model_name)
    if variant is None:
        variant = ""
    else:
        variant = "_" + variant
    res = cur.execute(f"UPDATE videos SET features_{model_name}{variant} = ?1 WHERE yt_id = ?2 AND download = 2 AND optical_flow = 2 AND features_{model_name}{variant} = 1 RETURNING *", (state, video_id))
    result = res.fetchone()
    print(result)
    assert result is not None
    con.commit()


def select_and_update(new_state, cur, con, video_id, model_name, variant):
    assert_valid_model_name(model_name)
    cur.execute("BEGIN")
    try:
        if variant is None:
            variant = ""
        else:
            variant = "_" + variant
        res = cur.execute(f"SELECT 1 FROM videos WHERE yt_id = ?1 AND download = 2 AND optical_flow = 2 AND features_{model_name}{variant} = 0 LIMIT 1", (video_id,))
        result = res.fetchone()
        if result is None:
            print("Video already processed!")
            cur.execute("ROLLBACK")
            return False
        res = cur.execute(f"UPDATE videos SET features_{model_name}{variant} = ?1 WHERE yt_id = ?2 AND download = 2 AND optical_flow = 2 AND features_{model_name}{variant} = 0 RETURNING *", (new_state, video_id))
        result = res.fetchone()
        print(result)
        assert result is not None
        cur.execute("COMMIT")
    except sqlite3.Error as e:
        print("failed!")
        cur.execute("ROLLBACK")
        raise e
    return True


def assert_valid_model_name(model_name):
    assert model_name in ["resnet18", "resnet34", "resnet101", "vgg16"], f"Invalid resnet_model_name {model_name}"


class FeatureExtractor():
    def __init__(self, video_names, frames_path, features_path, model_name, variant=None):
        self.video_names = video_names
        self.frames_path = frames_path
        self.features_path = features_path
        self.model_name = model_name
        self.variant = variant
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        assert_valid_model_name(self.model_name)
        match self.model_name:
            case "resnet18":
                model = models.resnet18(pretrained=True)
            case "resnet34":
                model = models.resnet34(pretrained=True)
            case "resnet101":
                model = models.resnet101(pretrained=True)
            case "vgg16":
                model = models.vgg16(pretrained=True)
            case _:
                raise ValueError(f"Invalid resnet_model_name {model_name}")

        if 'resnet' in self.model_name:
            model = nn.Sequential(
            *list(model.children())[:-1],  # Remove the final classification layer
            nn.AdaptiveAvgPool2d((1, 1))     # Add Adaptive Average Pooling layer "takes care if the input size is different from 224x224"
            ).eval()
            self.model = model.to(self.device)
        elif self.model_name == "vgg16":
            model = nn.Sequential(*list(model.children())[:-1])  # VGG16's features contain the convolutional part of the network
            model.add_module('global_avg_pool', nn.AdaptiveAvgPool2d((1, 1)))  # Add a global average pooling layer
            model.eval()
            self.model = model.to(self.device)
        else:
            raise ValueError(f'Unsupported model_name {self.model_name}')

        for param in model.parameters():
            param.requires_grad = False

    def extract_features(self, frame_path):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # vgg and resnet pretrained on imagenet
        ])
        img = Image.open(frame_path).convert('RGB')
        img = transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.model(img).squeeze().cpu().numpy()


    def run(self):
        con = sqlite3.connect(db_path, timeout=60)
        cur = con.cursor()
        while True:
            if len(self.video_names) == 0:
                con.close()
                break
            video_name = self.video_names.pop()
            we_got_the_job = select_and_update(STATE_EXTRACTING, cur, con, video_name, self.model_name, self.variant)
            if not we_got_the_job:
                #print("We did not get the job")
                continue
            video_path = os.path.join(self.frames_path, video_name)
            video_features_path = os.path.join(self.features_path, self.model_name, video_name)
            os.makedirs(video_features_path, exist_ok=True)

            for frame_name in tqdm(os.listdir(video_path), unit='frames'):
                frame_path = os.path.join(video_path, frame_name)
                if frame_name == "fps.txt":
                    shutil.copy(frame_path, video_features_path)
                    continue
                feature_name = f"{frame_name.rsplit('.', maxsplit=1)[0]}.npy"
                feature_path = os.path.join(video_features_path, feature_name)
                if os.path.isfile(feature_path):
                    print(video_name, f"Skipping frame {feature_name} since it already exists")
                    continue
                features = self.extract_features(frame_path)
                np.save(feature_path, features)
            mark_features_as(STATE_COMPLETED, cur, con, video_name, self.model_name, self.variant)

