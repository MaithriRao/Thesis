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
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from enum import Enum, verify, UNIQUE
from collections import Counter

class Category(Enum):
    O = 0
    B = 1
    I = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class bobsl_dataset(Dataset):  
    def __init__(self, dataset, frames_path):
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
        frame_group, is_beginning, is_end = self.dataset[idx]
        frames_data = []
        labels = []

        for frames, label in frame_group:
            for (video_name, frame_number) in frames:
                frame_path = self.get_feature_path(video_name, frame_number)
                img_frame = self.load_features(frame_path)
                img_frame_tensor = torch.tensor(img_frame)
                frames_data.append(img_frame_tensor)
            labels.append(label)

        frames = frames_data
        #print the size of the frames here

        labels = torch.tensor(labels, dtype=torch.float32)
        # print(labels.size())
        features = torch.stack(frames)
        # print("featuressize is",features.size())
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
        # Create a dictionary to map frames to sentences
        self.frame_to_sentence = {}

    def run(self):
        
        self.boundary_frames_train, interval_frames_train = self.cal_bound(self.train_set)
        self.boundary_frames_val, interval_frames_val = self.cal_bound(self.val_set)
        self.boundary_frames_test, interval_frames_test = self.cal_bound(self.test_set)

        train_loader, class_weights, label_counts  = self.train()
        val_loader, class_weights_val, label_counts_val  = self.val()
        test_loader, class_weights_test, label_counts_test = self.test()

        return train_loader, class_weights, val_loader, class_weights_val, test_loader, class_weights_test, interval_frames_test


    def timestamp_to_frame(self, timestamp, fps):
        hh, mm, ss, ms = map(float, re.split(r'[:.]', timestamp))
        total_seconds = hh * 3600 + mm * 60 + ss + ms / 1000
        frame_number = round(total_seconds * self.fps)
        return frame_number


    def cal_bound(self, video_dict, interval_seconds: int = 15):

        valid_frames = set()

        for (alignment_type, video_names) in video_dict.items():
            # Iterate through the frames in the folder
            for video_name in video_names:
                video_file_path = os.path.join(self.frames_path, video_name)

                for filename in os.listdir(video_file_path):
                    # Extract the frame number from the filename
                    frame_number = os.path.splitext(filename)[0] # [0] : "0001" , [1] : ".npy"/.jpg

                    # Check if the frame file exists in the folder
                    if frame_number.isdigit():
                        valid_frames.add((video_name, int(frame_number)))

        boundary_frames = {}
        for (video_name, frame) in valid_frames:
            boundary_frames[(video_name, frame)] = Category.O.value  # all frames are 0 initially


        invalid_frames = set()

        for (alignment_type, video_names) in video_dict.items():
            for video_name in video_names:
                sentence_file_path = os.path.join(self.sentences_path, alignment_type, f'{video_name}.vtt')

                # Read and process the .vtt file
                with open(sentence_file_path, 'r') as file:
                    lines = file.readlines()
                    i = 0
                    while i < len(lines): # continues until i < the total number of lines
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
            pass
            # print('WARNING: Skipping invalid frames', invalid_frames)


        # Group consecutive frames by label in the current interval
        def group_frames(current_interval_frames):
            grouped_frames = []
            current_group = None
            current_label = None

            for interval_frame in current_interval_frames:
                label = interval_frame['label']
                if label != current_label:
                    if current_label is not None:
                        grouped_frames.append((current_group, current_label))
                    current_group = []
                    current_label = label
                current_group.append(interval_frame['frame'])

            # Add the last group
            if current_label is not None:
                grouped_frames.append((current_group, current_label))

            return grouped_frames

        interval_lists = []
        all_interval_frames = []
        current_interval_frames = []
        last_video_name = None
        is_beginning = True

        for (video_name, frame) in sorted(valid_frames):
            different_video_started = video_name != last_video_name and last_video_name != None
            if len(current_interval_frames) >= interval_seconds * self.fps or different_video_started:
                grouped_frames = group_frames(current_interval_frames)
                all_interval_frames.append(current_interval_frames)
                if different_video_started:
                    is_end = True
                else:
                    is_end = False
                interval_lists.append((grouped_frames, is_beginning, is_end))
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

        # Add the last interval if it's not empty
        if current_interval_frames:
            grouped_frames = group_frames(current_interval_frames)
            all_interval_frames.append(current_interval_frames)
            is_end = True
            interval_lists.append((grouped_frames, is_beginning, is_end))

        lengths = Counter()
        for grouped_frames, _, _ in interval_lists:
            length = 0
            for frames, label in grouped_frames:
                length += len(frames)
            lengths.update([length])

        return interval_lists, all_interval_frames
    
    def calculate_weights(self, interval_lists):
        group_counter = Counter()
        frame_counter = Counter()

        for grouped_frames, is_beginning, is_end in interval_lists:
            for group, label in grouped_frames:
                group_counter[label] += 1  # Count each group once
                frame_counter[label] += len(group)  # Count frames for reference
        
        # Calculate total number of groups
        total_groups = sum(group_counter.values())

        ordered_labels = [0, 1, 2]
        
        # Calculate inverse frequency weights based on groups
        weights = {}
        for label in ordered_labels:
            count = group_counter.get(label, 0)  # Get count, default to 0 if label not found
            if count > 0:
                weights[label] = total_groups / count
            else:
                weights[label] = 0  # Handle case where there are no groups for this label

        return weights, group_counter 


    def collate_fn(self, batch):
        # Batch is a list of (features, labels) tuples
        features_batch, labels_batch, places_batch = zip(*batch)

        feature_lengths = [len(seq) for seq in features_batch]
        label_lengths = [len(seq) for seq in labels_batch]

        # Sort sequence lengths and features accordingly
        sorted_lengths, sorted_indices = torch.sort(torch.tensor(feature_lengths), descending=True)
        features_batch_sorted = [features_batch[i] for i in sorted_indices]


        sorted_labels_batch = [labels_batch[i] for i in sorted_indices]
        max_labels_length = max(label_lengths)


        # Add BOS tag to each label sequence
        bos_tag = torch.tensor([4], dtype=torch.long) 
        labels_batch_bos = [torch.cat([bos_tag, label]) for label in sorted_labels_batch]

        max_sequence_length = max(feature_lengths)  
        device = features_batch_sorted[0].device
        features_batch_sorted = [
        seq.view(seq.size(0), -1)  # Flatten [sequence_length, channels, height, width] to [sequence_length, channels * height * width]
        for seq in features_batch_sorted
        ]
        features_batch_sorted = [seq.to(device) for seq in features_batch_sorted]

        # Pad features with zeros
        features_batch_padded = pad_sequence(
            [torch.cat([seq, torch.zeros(max_sequence_length - len(seq), seq.size(1), device=device)])
            if len(seq) < max_sequence_length 
            else seq 
            for seq in features_batch_sorted],
            batch_first=True)

        # Pad labels with a padding value (e.g., -1)
        labels_batch_padded = pad_sequence(labels_batch_bos, batch_first=True, padding_value=3) 

        # Create padding mask
        padding_mask = (features_batch_padded != 0).any(dim=-1)

        return features_batch_padded, labels_batch_padded, feature_lengths, label_lengths 
 
        
    def train(self):
        training_set = bobsl_dataset(
            dataset = self.boundary_frames_train,
            frames_path = self.frames_path,
            )   
        # print('==> Training data:', len(self.training_set))

        train_loader = DataLoader(
            dataset= training_set, 
            batch_size=self.BATCH_SIZE,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
            collate_fn=self.collate_fn
            )

        class_weights, label_counts = self.calculate_weights(self.boundary_frames_train) 
        # print("class weights", class_weights)
        # print("label counts", label_counts)  

        return train_loader, class_weights, label_counts 


    def val(self):
        validation_set = bobsl_dataset(
            dataset = self.boundary_frames_val,
            frames_path = self.frames_path,
            )
        print('==> Validation data :', len(validation_set))

        val_loader = DataLoader(
            dataset=validation_set, 
            batch_size=self.BATCH_SIZE, 
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn)

        class_weights_val, label_counts_val = self.calculate_weights(self.boundary_frames_val)    

        return val_loader, class_weights_val, label_counts_val

    def test(self):
        test_set = bobsl_dataset(
            dataset = self.boundary_frames_test,
            frames_path = self.frames_path,
            )
        print('==> Test data :', len(test_set))

        test_loader = DataLoader(
            dataset=test_set, 
            batch_size=self.BATCH_SIZE, 
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn
            )
        class_weights_test, label_counts_test  = self.calculate_weights(self.boundary_frames_test)
        return test_loader, class_weights_test, label_counts_test  

