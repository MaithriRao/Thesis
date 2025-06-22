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
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
import json
from torch.utils.data.sampler import WeightedRandomSampler
from collections import Counter
from torchvision import models
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from enum import Enum, verify, UNIQUE

class Category(Enum):
    O = 0
    B = 1
    I = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class bobsl_dataset(Dataset):  
    def __init__(self, dataset, frames_path, cache_dir='cache'):
        self.dataset = dataset
        self.frames_path = frames_path

    def get_feature_path(self, video_name, frame_idx):
        return os.path.join(self.frames_path, video_name, f"{str(frame_idx).zfill(7)}.npy")

    def __len__(self):
        return len(self.dataset)

    def load_features(self, frame_path):
        img_frame = np.load(frame_path)
        return img_frame

    @verify(UNIQUE)
    class Place(Enum):
        BEGINNING = 1
        MIDDLE = 2
        END = 3
        BOTH = 4

    def __getitem__(self, idx):
        interval_frames, is_beginning, is_end = self.dataset[idx]
        frames_data = []
        labels = []


        # with interval_Seconds
        for interval_frame in interval_frames:
            (video_name, frame_number) = interval_frame['frame']
            label = interval_frame['label']
            frame_path = self.get_feature_path(video_name, frame_number)
            img_frame = self.load_features(frame_path)
            img_frame_tensor = torch.tensor(img_frame)
            frames_data.append(img_frame_tensor)
            labels.append(label)



        labels = torch.tensor(labels, dtype=torch.float32)
        features = torch.stack(frames_data)
        if is_beginning:
            if is_end:
                place = self.Place.BOTH
            else:
                place = self.Place.BEGINNING
        else:
            if is_end:
                place = self.Place.END
            else:
                place = self.Place.MIDDLE
        return features, labels, place

class BOBSL_DataLoader1():
    def __init__(self, BATCH_SIZE, num_workers, train_set, test_set, val_set, frames_path, sentences_path):
        self.BATCH_SIZE=BATCH_SIZE
        self.num_workers = num_workers
        self.frame_count = {}
        self.train_set = train_set
        self.test_set =  test_set
        self.val_set = val_set
        self.frames_path = frames_path
        self.sentences_path = sentences_path
        self.fps = 25
        self.image_filename_num_digits = 7
        self.frame_to_sentence = {}


    def run(self):
        boundary_frames_train = self.cal_bound(self.train_set)
        label_counts, class_weights = self.claculate_weights(boundary_frames_train)

        weight_train_info = (label_counts, class_weights)

        train_loader = self.train_val_test(boundary_frames_train, 'train')

        boundary_frames_val = self.cal_bound(self.val_set)
        label_counts_val, class_weights_val = self.claculate_weights(boundary_frames_val)
        
        weight_val_info = (label_counts_val, class_weights_val)

        val_loader = self.train_val_test(boundary_frames_val, 'val')

        boundary_frames_test = self.cal_bound(self.test_set)
        label_counts_test, class_weights_test = self.claculate_weights(boundary_frames_test)
        test_loader  = self.train_val_test(boundary_frames_test, 'test')

        return train_loader,  label_counts, class_weights, val_loader, label_counts_val, class_weights_val, test_loader, label_counts_test, class_weights_test

    def claculate_weights(sel, dataset):
        label_counts = Counter()

        for group in dataset:
            interval_frames, _, _ = group
            for frame_info in interval_frames:
                label = frame_info['label']
                label_counts[label] += 1
        total_counts = sum(label_counts.values())
        class_labels = sorted(label_counts.keys()) 

        class_weights = {label: total_counts / count for label, count in label_counts.items()}
        return label_counts, class_weights   

    def timestamp_to_frame(self, timestamp, fps):
        hh, mm, ss, ms = map(float, re.split(r'[:.]', timestamp))
        total_seconds = hh * 3600 + mm * 60 + ss + ms / 1000
        frame_number = round(total_seconds * self.fps)
        return frame_number


    def cal_bound(self, data_set, interval_seconds=15):
        valid_frames = set()

        for (alignment_type, video_names) in data_set.items():
            for video_name in video_names:
                video_file_path = os.path.join(self.frames_path, video_name)

                for filename in os.listdir(video_file_path):
                    frame_number = os.path.splitext(filename)[0]
                    if frame_number.isdigit():
                        valid_frames.add((video_name, int(frame_number)))

        boundary_frames = {}
        for (video_name, frame) in valid_frames:
            boundary_frames[(video_name, frame)] = Category.O.value  # all frames are 0 initially


        invalid_frames = set()

        for (alignment_type, video_names) in data_set.items():
            for video_name in video_names:
                sentence_file_path = os.path.join(self.sentences_path, alignment_type, f'{video_name}.vtt')

                # Read and process the .vtt file
                with open(sentence_file_path, 'r') as file:
                    lines = file.readlines()
                    i = 0
                    while i < len(lines):
                        # Use a regular expression to extract timestamps
                        timestamp_match = re.match(r'(\d{2}:\d{2}:\d{2}.\d{3}) --> (\d{2}:\d{2}:\d{2}.\d{3})', lines[i].strip())
                        if timestamp_match:
                            start_time, end_time = timestamp_match.groups()
                            # print("start and end time here", start_time, end_time)
                            sentence = lines[i + 1].strip()

                            # Calculate frame numbers for start and end times
                            start_frame = self.timestamp_to_frame(start_time, self.fps)
                            
                            # Calculate end frame by rounding down the total seconds for end time
                            end_frame = self.timestamp_to_frame(end_time, self.fps)

                            if start_frame + 1 >= end_frame:
                                print(f'WARNING: Subtitle unit is too short in file {sentence_file_path}, ignoring. Beginning: {start_time} ({start_frame}), end: {end_time} ({end_frame})')
                                i += 1
                                continue

                            boundary_frames[(video_name, start_frame)] = Category.B.value
                            # Assign the sentence to frames within the range
                            for frame in range(start_frame + 1, end_frame):
                                if (video_name, frame) not in valid_frames:
                                    invalid_frames.add((video_name, frame))
                                    continue
                                boundary_frames[(video_name, frame)] = Category.I.value
                            i += 1

                        i += 1

        if len(invalid_frames) != 0:
            print('WARNING: Skipping invalid frames', invalid_frames)

        all_frames = []
        current_interval_frames = []
        last_video_name = None
        is_beginning = True

        for (video_name, frame) in sorted(valid_frames):
            different_video_started = video_name != last_video_name and last_video_name != None
            if len(current_interval_frames) >= interval_seconds * self.fps or different_video_started:
                if different_video_started:
                    is_end = True
                else:
                    is_end = False
                all_frames.append((current_interval_frames, is_beginning, is_end))
                if different_video_started:
                    is_beginning = True
                else:
                    is_beginning = False
                current_interval_frames = []

            current_interval_frames.append({
                'frame': (video_name, frame),
                'label': boundary_frames[(video_name, frame)],
            })
            last_video_name = video_name

        if current_interval_frames:
            is_end = True
            all_frames.append((current_interval_frames, is_beginning, is_end))

        lengths = Counter()
        for current_interval_frames, _, _ in all_frames:
            length = len(current_interval_frames)
            lengths.update([length])
        print(f'Interval lengths: {lengths}')

        return all_frames


    def collate_fn(self, batch):
        features_batch, labels_batch, places_batch = zip(*batch)

        feature_lengths = [len(seq) for seq in features_batch]
        label_lengths = [len(seq) for seq in labels_batch]

        # Sort sequence lengths and features accordingly
        sorted_lengths, sorted_indices = torch.sort(torch.tensor(feature_lengths), descending=True)
        features_batch_sorted = [features_batch[i] for i in sorted_indices]
        sorted_labels_batch = [labels_batch[i] for i in sorted_indices]
        sorted_places_batch = [places_batch[i] for i in sorted_indices]

        max_sequence_length = max(feature_lengths)  
        device = features_batch_sorted[0].device
        features_batch_sorted = [seq.to(device) for seq in features_batch_sorted]

        # Pad features with zeros
        features_batch_padded = pad_sequence(
            [torch.cat([seq, torch.zeros(max_sequence_length - len(seq), seq.size(1), device=device)]) 
            if len(seq) < max_sequence_length 
            else seq 
            for seq in features_batch_sorted],
            batch_first=True)

        # Pad labels with a padding value (e.g., -1)
        labels_batch_padded = pad_sequence(sorted_labels_batch, batch_first=True, padding_value=3)

        # Create padding mask
        padding_mask = (features_batch_padded != 0).any(dim=-1)
        return features_batch_padded, labels_batch_padded, feature_lengths, label_lengths   


    def train_val_test(self, dataset, mode):
        assert mode == 'train' or mode == 'val' or mode == 'test'

        cache_dir = {
        'train': '/ds/videos/opticalflow-BOBSL/bobsl/cache/train/',
        'val': '/ds/videos/opticalflow-BOBSL/bobsl/cache/val/',
        'test': '/ds/videos/opticalflow-BOBSL/bobsl/cache/test/'
        }[mode]

        train_val_test_set = bobsl_dataset(

            dataset = dataset,
            frames_path = self.frames_path,
            cache_dir=cache_dir,)

        match mode:
            case 'train':
                name = 'Training'
                shuffle = True
            case 'val':
                name = 'Validation'
                shuffle = False
            case 'test':
                name = 'Test'
                shuffle = False

        length = len(train_val_test_set)
        print(f'==> {name} data: {length}')

        train_val_test_loader = DataLoader(
            dataset=train_val_test_set, 
            batch_size=self.BATCH_SIZE,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn
            )

        return train_val_test_loader
