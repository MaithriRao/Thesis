import numpy as np
from sklearn.metrics import f1_score
import math
from typing import List
import torch


BIO = {"O": 0, "B": 1, "I": 2}



def extract_segments(labels):
    segments = []
    start = None
    current_label = None

    for i, label in enumerate(labels):
        if label != current_label:  # Label has changed
            if current_label is not None:  # End the previous segment
                segments.append({'start': start, 'end': i - 1})  # End at the previous index
            start = i  # Start a new segment
            current_label = label

    # Handle the last segment
    if current_label is not None:
        segments.append({'start': start, 'end': len(labels) - 1})  # End at the last index

    return segments


def segment_percentage(segments: List[dict], segments_gold: List[dict]) -> float:
    """
    segments: [{'start': 1, 'end': 2}, ...]
    """
    if len(segments_gold) == 0:
        return 100.0 if len(segments) == 0 else 0.0 
        # return 1 if len(segments) == 0 else len(segments)
    return (len(segments) / len(segments_gold)) * 100
    # return len(segments) / len(segments_gold)


def frame_f1(probs: torch.Tensor, gold: torch.Tensor, **kwargs) -> float:
  """
  probs: [sequence_length x number_of_classes(3)]
  gold: [sequence_length]
  """
  return f1_score(gold.cpu().numpy(), probs.cpu().numpy(), **kwargs)


def io_probs_to_segments(probs):
    segments = []
    i = 0
    while i < len(probs):
        if probs[i, BIO["I"]] > 50:
            end = len(probs) - 1
            for j in range(i + 1, len(probs)):
                if probs[j, BIO["I"]] < 50:
                    end = j - 1
                    break
            segments.append({"start": i, "end": end})
            i = end + 1
        else:
            i += 1

    return segments

def prepare_labels_for_iou(labels):
    """
    Convert ground truth labels into segments for IoU calculation.

    Args:
        labels (numpy.ndarray): Ground truth labels in shape (num_frames,).

    Returns:
        List[dict]: List of segments with start and end indices.
    """
    segments = []
    start = None

    for idx, value in enumerate(labels):
        if value == BIO["B"]:  # Beginning of a segment
            if start is None:  # Start a new segment
                start = idx
        elif value == BIO["O"]:  # Outside of a segment
            if start is not None:  # End the current segment
                segments.append({"start": start, "end": idx - 1})
                start = None

    # Handle case where segment ends at the last frame
    if start is not None:
        segments.append({"start": start, "end": len(labels) - 1})

    return segments

def probs_to_segments(logits, b_threshold=20., o_threshold=20., threshold_likeliest=False, restart_on_b=True):
    probs = np.round(np.exp(logits.squeeze()) * 100)
    probs = (probs / probs.sum(axis=1, keepdims=True)) * 100  # Normalize probabilities and scale to 0-100 range
    # print(probs)

    if np.all(probs[:, BIO["B"]] < b_threshold):
        # print("Condition is true, returning io_probs_to_segments")
        return io_probs_to_segments(probs)

    segments = []

    segment = {"start": None, "end": None}
    did_pass_start = False
    
    for idx in range(len(probs)):
        # print(f"Index: {idx}, B prob: {probs[idx, BIO['B']]}, I prob: {probs[idx, BIO['I']]}, O prob: {probs[idx, BIO['O']]}")
        b = float(probs[idx, BIO["B"]])
        i = float(probs[idx, BIO["I"]])
        o = float(probs[idx, BIO["O"]])

        if threshold_likeliest:
            b_threshold = max(i, o)
            o_threshold = max(b, i)

        if segment["start"] is None:
            if b > b_threshold:
                segment["start"] = idx
        else:
            if did_pass_start:
                if (restart_on_b and b > b_threshold) or o > o_threshold:
                    segment["end"] = idx - 1

                    # reset
                    segments.append(segment)
                    segment = {"start": None if o > o_threshold else idx, "end": None}
                    did_pass_start = False
            else:
                if b < b_threshold:
                    did_pass_start = True

    if segment["start"] is not None:
        segment["end"] = len(probs)
        segments.append(segment)

    return segments

def segment_IoU(segments: List[dict], segments_gold: List[dict], max_len=1000000) -> float:
    segments_v = np.zeros(max_len)
    for segment in segments:
        segments_v[segment['start']:(segment['end'] + 1)] = 1

    segments_gold_v = np.zeros(max_len)
    for segment in segments_gold:
        segments_gold_v[segment['start']:(segment['end'] + 1)] = 1

    intersection = np.logical_and(segments_v, segments_gold_v)
    union = np.logical_or(segments_v, segments_gold_v)

    if np.sum(union) == 0:
        return 1 if np.sum(intersection) == 0 else 0

    return float(np.sum(intersection) / np.sum(union))    


def find_segments(labels, padding_value = 3):
    segments = []
    start = None
    for i, label in enumerate(labels):
        if label == padding_value:
            # Close any open segment when encountering padding
            if start is not None:
                segments.append((start, i - 1))
                start = None
        elif label == 1:  # Beginning of a new segment
            if start is not None:  # If there's already an ongoing segment, close it
                segments.append((start, i - 1))
            start = i  # Start a new segment
        elif label == 2 and start is not None:  # Inside segment
            continue
        elif start is not None and label != 2:  # End of segment when not inside
            segments.append((start, i - 1))
            start = None
    if start is not None:
        segments.append((start, len(labels) - 1))
    return segments


def calculate_iou(pred_labels, true_labels, classes_of_interest):
  ious = []
  for label in classes_of_interest:
    # if label == padding_idx:
    #   continue  # Skip padding index

    pred_mask = (pred_labels == label)
    true_mask = (true_labels == label)

    # Calculate intersection and union
    intersection = np.sum(pred_mask & true_mask)
    union = np.sum(pred_mask | true_mask)
    iou = intersection / union if union > 0 else 0
    ious.append(iou)

  return ious  



def count_segments(labels):
    """
    Counts the number of distinct segments in the given label sequence.
    Segments are defined as continuous sequences of the same non-zero label.
    """
    if len(labels) == 0:
        return 0
    
    # Convert to numpy array if it's a PyTorch tensor
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    
    segments = 0
    current_label = 0
    
    for label in labels:
        if label != 0 and label != current_label:
            segments += 1
        current_label = label
    
    return segments    