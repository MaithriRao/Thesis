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
import torchvision.transforms as transforms


torch.multiprocessing.set_start_method('spawn', force=True)
class Category(Enum):
    O = 0
    B = 1
    I = 2

class bobsl_dataset(Dataset):  
    def __init__(self, dataset, frames_path, cache_dir='cache', transform=None):
        self.dataset = dataset
        self.frames_path = frames_path
        self.transform = transform
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def get_feature_path(self, video_name, frame_idx):
        # return os.path.join(self.frames_path, video_name, 'flow_'+ f"{str(frame_idx).zfill(7)}.png")
        return os.path.join(self.frames_path, video_name, 'flow_'+ f"{str(frame_idx).zfill(7)}.npy")

    # def get_cache_path(self, video_name, frame_idx):
    #     # Create a subdirectory in cache_dir for each video_name
    #     video_cache_dir = os.path.join(self.cache_dir, video_name)
    #     os.makedirs(video_cache_dir, exist_ok=True)
    #     return os.path.join(video_cache_dir, f"flow_{str(frame_idx).zfill(7)}.pt")

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

        for frames in frame_group:
            (video_name, frame_number) = frames['frame']
            label = frames['label']
            frame_path = self.get_feature_path(video_name, frame_number)    
            img_frame = self.load_features(frame_path)    
            img_frame_tensor = torch.tensor(img_frame)
       
            
            frames_data.append(img_frame_tensor)
            labels.append(label)

        labels = torch.tensor(labels, dtype=torch.float32)
        # print(labels.size())
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
    def __init__(self, BATCH_SIZE, num_workers, train_set, val_set, test_set, frames_path, sentences_path):
        self.BATCH_SIZE=BATCH_SIZE
        self.num_workers = num_workers
        self.frame_count = {}
        self.train_set = train_set
        self.val_set = val_set
        self.test_set =  test_set
        self.frames_path = frames_path
        self.sentences_path = sentences_path
        self.image_filename_num_digits = 7
        self.frame_to_sentence = {}

    def run(self):
        self.cal_bound('train')
        self.cal_bound('val')
        self.cal_bound('test')
        train_loader, label_counts, class_weights  = self.train()
        print("class_weights in train", class_weights)
        val_loader, label_counts_val, class_weights_val  = self.val()
        print("class_weights in val", class_weights_val)
        test_loader = self.test()

        return train_loader, label_counts, class_weights, val_loader, label_counts_val, class_weights_val, test_loader

    def calculate_weights(sel, dataset):
        label_counts = Counter()

        for group in dataset:
            interval_frames, _, _ = group
            for frame_info in interval_frames:
                label = frame_info['label']
                label_counts[label] += 1
        total_counts = sum(label_counts.values())
        class_labels = sorted(label_counts.keys()) 
        # class_weights = [total_counts / label_counts[i] for i in range(len(label_counts))] # from reference code and no normalization
        class_weights = {label: total_counts / count for label, count in label_counts.items()} # from here normalization

        # weight_sum = sum(class_weights.values())
        # normalized_class_weights = {label: weight / weight_sum for label, weight in class_weights.items()}

        # return label_counts, normalized_class_weights  
        return label_counts, class_weights
 
    @staticmethod
    def timestamp_to_frame(timestamp, fps):
        hh, mm, ss, ms = map(float, re.split(r'[:,]', timestamp))
        total_seconds = hh * 3600 + mm * 60 + ss + ms / 1000
        frame_number = round(total_seconds * fps)
        return frame_number

    def cal_bound(self, mode: str, max_frames: int = 375):
        assert mode == 'train' or mode == 'val' or mode == 'test'

        match mode:
            case 'train':
                video_names = self.train_set
            case 'val':
                video_names = self.val_set
            case 'test':
                video_names = self.test_set

        video_id_to_srt_file_mapping = {}
        for filename in os.listdir(self.sentences_path):
            match = re.match(r'.*\[(.+)\]', filename)
            if match is None:
                print(f"Failed to parse video id from {filename}")
                continue
            (video_id,) = match.groups()
            video_id_to_srt_file_mapping[video_id] = filename

        valid_frames = set()
        for video_name in video_names:
            #f_path = f_path.rstrip('/')
            video_file_path = os.path.join(self.frames_path, video_name)

            for filename in os.listdir(video_file_path):
                if filename == 'fps.txt':
                    continue
                filename = os.path.splitext(filename)[0]
                match = re.match(r'.*(\d{7})$', filename)
                if match is None:
                    print(f"Failed to parse frame number from {filename}")
                    continue
                (frame_number,) = match.groups()
                valid_frames.add((video_name, int(frame_number)))

        boundary_frames = {}
        for (video_name, frame) in sorted(valid_frames):
            boundary_frames[(video_name, frame)] = Category.O.value # all frames are 0 initially


        invalid_frames = set()
        for video_name in video_names:
            fps = None
            with open(os.path.join(self.frames_path, video_name, 'fps.txt'), 'r') as file:
                fps = file.read().strip()
                # print(fps)
                fps = float(fps)
            # Read and process the .srt file
            sentence_file_path = os.path.join(self.sentences_path, video_id_to_srt_file_mapping[video_name])

            # print(f'Reading srt file {sentence_file_path}')
            with open(sentence_file_path, 'r') as file:
                lines = file.readlines()
                if len(lines) == 0:
                    print(f'WARNING: Empty file {sentence_file_path}, skipping...')
                    continue
                # print(lines)
                i = 0
                while i < len(lines):
                    line = lines[i].strip()
                    if not line.isdigit():
                        i += 1
                        continue
                    # Move to the next line
                    i += 1
                    line = lines[i].strip()
                    # Use a regular expression to extract timestamps
                    timestamp_match = re.match(r'(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})', line)
                    if not timestamp_match:
                        raise ValueError(f'Timestamp expected but not found in file {sentence_file_path}')

                    start_time, end_time = timestamp_match.groups()
                    # print("start and end time here", start_time, end_time)
                    i += 1
                    sentence_pieces = []
                    while i < len(lines):
                        line = lines[i].rstrip('\n')
                        if line.strip() == '':
                            break
                        sentence_pieces.append(line)
                        i += 1

                    # Calculate frame numbers for start and end times
                    start_frame = self.timestamp_to_frame(start_time, fps)

                    # Calculate end frame by rounding down the total seconds for end time
                    end_frame = self.timestamp_to_frame(end_time, fps)

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
                        # print(f'==> {video_name} {frame} is a boundary frame')
                        boundary_frames[(video_name, frame)] = Category.I.value ##All the frames belonging to sentences are marked as 1(not boundary) except the strat and end frame of sentences and rest marked 0(boundary)

                    i += 1

        if len(invalid_frames) != 0:
            pass
            # print('WARNING: Skipping invalid frames', sorted(invalid_frames))


        all_frames = []
        current_interval_frames = []
        last_video_name = None
        is_beginning = True   

        for (video_name, frame) in sorted(valid_frames):
            different_video_started = video_name != last_video_name and last_video_name != None
            if len(current_interval_frames) >= max_frames or different_video_started:
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
        # Add the last interval if it's not empty
        if current_interval_frames:
            is_end = True
            all_frames.append((current_interval_frames, is_beginning, is_end))

        lengths = Counter()
        for current_interval_frames, _, _ in all_frames:
            length = len(current_interval_frames)
            lengths.update([length])
        print(f'Interval lengths: {lengths}')


        match mode:
            case 'train':
                self.boundary_frames_train = all_frames
                # print('==> Training data:', self.boundary_frames_train)

            case 'val':
                self.boundary_frames_val = all_frames
                # print('==> Validation data:', self.boundary_frames_val)

            case 'test':
                self.boundary_frames_test = all_frames  

    def collate_fn(self, batch):
        # Batch is a list of (features, labels) tuples
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
        labels_batch_padded = pad_sequence(sorted_labels_batch, batch_first=True, padding_value=3)

        # Create padding mask
        padding_mask = (features_batch_padded != 0).any(dim=-1)

        return features_batch_padded, labels_batch_padded, feature_lengths, label_lengths 
        # print("features_batch_padded", features_batch_padded.size())

        return features_batch_padded, labels_batch_padded, feature_lengths, label_lengths
      

    def train(self):
        random.seed(42)

        train_set = bobsl_dataset(
            dataset = self.boundary_frames_train,
            frames_path = self.frames_path,
            cache_dir='/ds/videos/opticalflow-BOBSL/ASL/cache/train/',
            transform = transforms.Compose([  
            transforms.Resize([224,224]),
            transforms.ToTensor(),
            ]),
            )   
        print('==> Training data:', len(train_set), 'frames', train_set[1][0].size())

        train_loader = DataLoader(
            dataset=train_set, 
            batch_size=self.BATCH_SIZE,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
            collate_fn=self.collate_fn
            )
        label_counts, class_weights = self.calculate_weights(self.boundary_frames_train)
        print("label counts", label_counts)
        print("class weights", class_weights)

        return train_loader, label_counts, class_weights

    def val(self):

        val_set = bobsl_dataset(
            dataset = self.boundary_frames_val,
            frames_path = self.frames_path,
            cache_dir='/ds/videos/opticalflow-BOBSL/ASL/cache/val/',
            transform = transforms.Compose([  
            transforms.Resize([224,224]),
            transforms.ToTensor(),
            ]),
            )
        print('==> Validation data :', len(val_set), 'frames', val_set[1][0].size())

        val_loader = DataLoader(
            dataset=val_set, 
            batch_size=self.BATCH_SIZE, 
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn)

        label_counts_val, class_weights_val = self.calculate_weights(self.boundary_frames_val)  

        return val_loader, label_counts_val, class_weights_val

    def test(self):

        test_set = bobsl_dataset(
            dataset = self.boundary_frames_test,
            frames_path = self.frames_path,
            cache_dir='/ds/videos/opticalflow-BOBSL/ASL/cache/test/',
            transform = transforms.Compose([  
            transforms.Resize([224,224]),
            transforms.ToTensor(),
            ]),)
        print('==> Test data :', len(test_set), 'frames', test_set[1][0].size())

        test_loader = DataLoader(
            dataset=test_set, 
            batch_size=self.BATCH_SIZE, 
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn)

        return test_loader  

if __name__ == '__main__':
    data_loader = BOBSL_DataLoader1(
        BATCH_SIZE=1,
        num_workers=2,
        train_set = [
            '0bodeyCThJM', '0pHAEC2ydR0', '0TgSMo2M8G8', '2Wsw0KI-yQg',
        ],
        val_set = [
            'UznY5SfH0RI',  'Zz9jsv6y7aA',
        ],
        test_set = [
            'UznY5SfH0RI', 
        ],
        
        frames_path = '/ds/videos/opticalflow-BOBSL/ASL/features/extracted_features/vgg16/',
        sentences_path = '/ds/videos/opticalflow-BOBSL/ASL/subtitles/',
    )
    train_loader, val_loader, test_loader = data_loader.run()
  
    print("Done")