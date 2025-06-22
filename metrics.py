import numpy as np
from sklearn.metrics import f1_score
import math
from typing import List
import torch


def frame_f1(probs: torch.Tensor, gold: torch.Tensor, **kwargs) -> float:
  """
  probs: [sequence_length x number_of_classes(3)]
  gold: [sequence_length]
  """
  return f1_score(gold.cpu().numpy(), probs.cpu().numpy(), **kwargs)


def calculate_iou(pred_labels, true_labels, classes_of_interest):
  ious = []
  for label in classes_of_interest:

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

def calculate_segment_percentage(pred_segments, gt_segments):
    """
    Calculates the percentage of predicted segments relative to ground truth segments.
    
    """
    if gt_segments > 0:
        return (pred_segments / gt_segments) * 100
    else:
        return 0
