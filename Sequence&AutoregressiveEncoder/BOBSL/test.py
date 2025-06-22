import torch
import torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
import random
import sklearn
import numpy as np
import pickle
from PIL import Image
import time
from tqdm import tqdm
import shutil
from random import randint
import argparse
import math
from typing import List

from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from .frame_data import *
import sys
import math
from .model_frame import *
from torch.autograd import Variable
import torch.nn.functional as F
from transformers import AdamW, get_linear_schedule_with_warmup

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import os
from pathlib import Path
import numpy as np
from PIL import Image
from metrics import *

def testing(model, decoder, test_loader, loss_fn_val, device, config):
  best_f1 = 0.0
  model.eval()
  val_loss = 0
  all_predictions = []
  all_labels = []
  all_ious = []
  padding_index = 3
  best_iou = -1
  with torch.no_grad():
    for i, data in tqdm(enumerate(test_loader), total = len(test_loader)):
      features , labels, feature_lengths, label_lengths = data

      features = features.to(device)
      labels =  labels.to(device)

      decoder_output = model(features, feature_lengths)

      trg = labels.view(-1).long()
      outputs = decoder_output.view(-1, model.encoder.output_size)
      # loss = loss_fn1(outputs, trg)

      mask = trg != padding_index
      loss = loss_fn_val((F.log_softmax(outputs, dim=-1))[mask], trg[mask])
      loss = loss.mean()

      val_loss += loss.item()   
      preds = outputs.argmax(dim=1)
      all_predictions.extend(preds.cpu().numpy())  # Store predictions
      all_labels.extend(trg.cpu().numpy())   

  avg_val_loss = val_loss / len(test_loader) 
  # scheduler.step(avg_val_loss)
  # Remove padding indices from predictions and labels
  mask_tensor = (torch.tensor(all_labels) != padding_index)
  all_predictions_tensor =  torch.tensor(all_predictions)   # Filter predictions
  all_labels_tensor = torch.tensor(all_labels)
  filtered_predictions = all_predictions_tensor[mask_tensor]  # Filter predictions
  filtered_labels = all_labels_tensor[mask_tensor] # Filter labels

  frame_f1_score = frame_f1(filtered_predictions, filtered_labels, average='macro')

  ious = calculate_iou(filtered_predictions.cpu().numpy(), filtered_labels.cpu().numpy(), classes_of_interest = [1, 2])
  mean_iou = np.mean(ious)


  gt_segments = count_segments(filtered_labels)
  pred_segments = count_segments(filtered_predictions)
  percentage_of_segments = (pred_segments / gt_segments * 100) if gt_segments > 0 else 0

  print(f"Average validation Loss: {avg_val_loss:.5f}")
  print(f" iou: {mean_iou:.4f}")
  print(f"Percentage of Segments: {percentage_of_segments:.2f}%")
  print(f"validation F1 Score: {frame_f1_score:.5f}")

    
# import heapq


def beam_search_inference(model, test_loader, device, beam_width):
  model.eval()
  all_predictions = []
  all_softmax_outputs = []
  all_ground_truths = []

  with torch.no_grad():
    for i, data in enumerate(test_loader):
      features, labels, feature_lengths, _ = data
      features = features.to(device)
      max_length = labels.shape[1]

      # Get encoder output
      encoder_states, hidden, cell = model.encoder(features, feature_lengths)

      # Initialize the beam
      start_token = torch.tensor([[4]], device=device).squeeze(0)
      start_token = F.one_hot(start_token, num_classes=model.decoder.output_size).float()

      beam = [(0, start_token, hidden, cell, [], [])]  # Added empty list for softmax outputs

      for _ in range(1, max_length):  # Iterate until max_length
        candidates = []
        for cumulative_score, decoder_input, hidden, cell, sequence, softmax_outputs in beam:
          # Forward pass through decoder
          decoder_output, hidden, cell = model.decoder(decoder_input, encoder_states, hidden, cell)
          softmax_output = F.softmax(decoder_output, dim=1)
          log_probs = torch.log(softmax_output)

          top_k_probs, top_k_indices = log_probs.topk(beam_width)
          for prob, idx in zip(top_k_probs[0], top_k_indices[0]):
              new_score = cumulative_score + prob.item()
              new_input = F.one_hot(idx.unsqueeze(0), num_classes=model.decoder.output_size).float()
              new_sequence = sequence + [idx.item()]
              new_softmax_outputs = softmax_outputs + [softmax_output.squeeze().cpu()]
              candidates.append((new_score, new_input, hidden.clone(), cell.clone(), new_sequence, new_softmax_outputs))

          # Select top beam_width candidates
          beam = heapq.nlargest(beam_width, candidates, key=lambda x: x[0])

      # Get the best sequence and corresponding softmax outputs
      best_sequence, best_softmax_outputs = max(beam, key=lambda x: x[0])[4:]
      ground_truth = labels[0, 1:].cpu().numpy().astype(int).tolist()

      all_predictions.append(best_sequence)
      all_softmax_outputs.append(best_softmax_outputs)
      all_ground_truths.append(ground_truth)      
      
      # print(f"Ground truth: {ground_truth}")
      # print(f"Predicted   : {best_sequence}")
      # print(f"best_softmax_outputs: {best_softmax_outputs}")


  return all_predictions, all_ground_truths, all_softmax_outputs

def inference(model, test_loader, device):
  model.eval()  # Set the model to evaluation mode
  all_predictions = []
  all_softmax_outputs = []

  with torch.no_grad():  # Disable gradient calculations
    for i, data in enumerate(test_loader):
      features, labels, feature_lengths, _ = data
      features = features.to(device)

      # Get encoder output (no need for autoregressive behavior here)
      outputs = model(features, feature_lengths)

      # Apply softmax to get probabilities
      softmax_outputs = F.softmax(outputs, dim=-1)

      # Get predicted classes
      predicted_classes = softmax_outputs.argmax(dim=-1)  # Shape: (batch_size, seq_length)

      # Collect predictions and softmax outputs for further analysis
      all_predictions.append(predicted_classes.cpu().numpy())
      all_softmax_outputs.append(softmax_outputs.cpu().numpy())

      # Optionally print ground truth and predictions for debugging
      if labels is not None:
        ground_truth = labels.cpu().numpy()
        # print(f"Ground truth: {ground_truth}")
        # print(f"Predicted: {predicted_classes.cpu().numpy()}")

  return all_predictions, all_softmax_outputs  


def combine_sequences_pred(all_predictions, all_softmax_outputs, sequence_frames=375):
  #here sequence_frames= 375
  combined_preds = []
  current_time = 0

  for preds_chunk, softmax_chunk in zip(all_predictions, all_softmax_outputs):
        # Ensure preds_chunk and softmax_chunk are numpy arrays
        preds_chunk = np.array(preds_chunk)
        softmax_chunk = np.array(softmax_chunk)
        print(f"Sample predictions: {preds_chunk[:5]}")
        print(f"Sample softmax outputs: {softmax_chunk[:5]}")

        # Calculate durations
        probabilities = softmax_chunk[np.arange(len(preds_chunk)), preds_chunk]
        
        total_prob = np.sum(probabilities)
        durations = probabilities / total_prob * sequence_frames

        for pred, dur in zip(preds_chunk, durations):
            combined_preds.append((current_time, current_time + dur, pred))
            current_time += dur

  return combined_preds


def combine_sequences_gt(all_interval_frames_test):
  combined_gt = []
  current_time = 5

  for interval_frames_test in all_interval_frames_test:
    for frame_info in interval_frames_test:
      label = frame_info["label"]
      start_time = current_time
      current_time += 1
      end_time = current_time
      combined_gt.append((start_time, end_time, label))

  return combined_gt


def merge_repeating_predictions(preds):
  result = []
  current_segment = preds[0]
  for next_segment in preds[1:]:
    if next_segment[2] == current_segment[2]:
      current_segment = (current_segment[0], next_segment[1], current_segment[2])
    else:
      result.append(current_segment)
      current_segment = next_segment

  result.append(current_segment)
  return result


def merge_short_predictions(preds, min_segment_duration=1.0):
  result = []
  accumulator = []
  for segment in preds:
    accumulator.append(segment)
    start_time = accumulator[0][0]
    end_time = accumulator[-1][1]
    # Merge with current segment if it's too short
    if end_time - start_time >= min_segment_duration:
      result.append((start_time, end_time, accumulator[-1][2]))
      accumulator = []
  return result


def tokens_to_subtitle(preds):
  result = []
  current_subtitle_start = None
  current_subtitle_end = None
  for segment in preds:
    if segment[2] in [0, 1]: # We're not continuing a subtitle
      if current_subtitle_start is not None: # We're in the middle of a subtitle, end that one first
        result.append((current_subtitle_start, current_subtitle_end))
        current_subtitle_start = None
        current_subtitle_end = None
    if segment[2] == 1: # Beginnig of subtitle
      current_subtitle_start = segment[0] # Start a new subtitle
      current_subtitle_end = segment[1]
    if segment[2] == 2: # Continue a subtitle
      if current_subtitle_start is None: # We're trying to continue a subtitle but we never started one, so start it now
        current_subtitle_start = segment[0] # Start a new subtitle
        current_subtitle_end = segment[1]
      else:
        current_subtitle_end = segment[1]
  if current_subtitle_start is not None: # We're in the middle of a subtitle, end that one
    result.append((current_subtitle_start, current_subtitle_end))
  return result


def generate_srt(subtitles, fps = 25):
  print("fps", fps)
  srt_content = ""
  sentence_count = 1
  for i, (start, end) in enumerate(subtitles, 1):
    srt_content += f"{i}\n"
    srt_content += f"{format_time(start, fps)} --> {format_time(end, fps)}\n"
    srt_content += f"[Sentence Content {i}]\n\n"
  return srt_content

def format_time(frame_number, fps):
  in_seconds = frame_number/fps
  hours = int(in_seconds // 3600)
  in_seconds -= hours * 3600
  minutes = int(in_seconds // 60)
  in_seconds -= minutes * 60
  seconds = int(in_seconds)
  in_seconds -= seconds
  milliseconds = int(in_seconds * 1000)
  return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"   

def IOU(subtitles_pred, subtitles_gt, max_len = 1000000):
  segments_pred = np.zeros(max_len)
  segments_gt = np.zeros(max_len)
  for (pred, gt) in zip(subtitles_pred, subtitles_gt):
    start_pred, end_pred = pred
    start_gt, end_gt = gt
    start_pred, end_pred = int(start_pred), int(end_pred)
    # print(type(start_pred), type(end_pred), type(start_gt), type(end_gt))
    segments_pred[start_pred:(end_pred + 1)] = 1
    segments_gt[start_gt:(end_gt + 1)] = 1

  intersection = np.logical_and(segments_pred, segments_gt)
  union = np.logical_or(segments_pred, segments_gt)

  if np.sum(union) == 0:
    return 1 if np.sum(intersection) == 0 else 0

  return float(np.sum(intersection) / np.sum(union))

            
def main(config):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_loader = BOBSL_DataLoader1(
    BATCH_SIZE=config.batch_size,
    num_workers=2,
      train_set = {'manually-aligned': ['5085357672350628540',
                '5099449889545545450',
                '5099449889545545450', '5085344787448740525','5086465773912997411',
                '5090916219025116054', '5092765202446045704', '5097975856769557081', '5103182645622502287',
                '5146258161124252727', '5172231547180314349', '5218949624445806415', '5366409594815891602',
                '5391892065246550398', '5525084584552992937'],
                'audio-aligned': [ '5087953980081062580', '5090546422340930233',
                '5093160768934007504', '5093521546186871869', '5093895208341624036', '5094257274914886988',
                '5099091689273058848', '5101325932090649776', '5104636062555440832', '5105777664862718183',
                '5106137153625393471', '5108723153434317714', '5113588492387230304', '5120215198258444599',
                '5125058632047937012', '5129558040617440365', '5130309229567302144', '5132160790798818890',
                '5134749367588120522', '5137350828449100326', '5138091711137870315', '5139568320894207394',
                '5141035910389041992', '5142915818404712489', '5145138463150184862', '5145510836814748271']},

    val_set = {'manually-aligned': ['5224144816887051284',
                                    '5242317681679687839', 
                                    '5294309549287947552', '5439409006429129628'],
                                    'audio-aligned': ['5213407827313563421', '5220817935219567388', '5221198038995053656', '5223769865411899831',
                                    '5225978337595504968', '5228592684188581488']},

    test_set = {'manually-aligned': ['5402656112283796519', '6220140001066210692', '6065139785314331208', '6243537694404620461', '6212378136168874626','6040895553921856506',
    '5130659699728866130', '6242043045785611181','6186362230766820709', '6003446875091426252', '6240919482340977071', '6184988700225556709', '6262857316295504035',]},    
    
    frames_path = '/ds/videos/opticalflow-BOBSL/bobsl/extracted_features/resnet101/',
    sentences_path = '/ds/videos/opticalflow-BOBSL/',
   )
    train_loader, _, class_weights_train, val_loader, _, class_weights_val, test_loader, _, class_weights_test  = data_loader.run() 
    input_size = 2048  
    hidden_size = 64
    N_LAYERS = 4
    ENC_DROPOUT = 0
    num_classes = 4
    output_size = 4
    ignore_index = 3
    bidirectional = True
    autoregressive = False

    encoder = AutoregressiveEncoder(input_size, hidden_size, output_size, N_LAYERS, ENC_DROPOUT, bidirectional, autoregressive)
    model = Seq2Seq(encoder)
    model = model.to(device)

    # num_params = count_parameters(model)
    # print(f"Total number of trainable parameters: {num_params}")
    print(f"Training with hidden_size={hidden_size}, num_layers={N_LAYERS}, enc_dropout={ENC_DROPOUT}")
   
    best_model_state_dict = torch.load("best_model_frame64_1e-04.pth")
    model.load_state_dict(best_model_state_dict) #for f1
    # model.load_state_dict(best_model_state_dict['model_state_dict']) # for loss 

    weights_val = torch.zeros(num_classes, dtype=torch.float32)   
    weights_val[0] = class_weights_test[0]
    weights_val[1] = class_weights_test[1]
    weights_val[2] = class_weights_test[2]

    weights_val = weights_val.to(device)
    loss_fn1 = nn.NLLLoss(reduction='none', weight = weights_val)

    testing(model, encoder, test_loader, loss_fn1, device, config)


    # # # Beam search
    # all_predictions, all_softmax_outputs = inference(model, test_loader, device)
    # combined_preds = combine_sequences_pred(all_predictions, all_softmax_outputs)
    # combined_preds = merge_repeating_predictions(combined_preds)
    # combined_preds = merge_short_predictions(combined_preds)
    # subtitles_pred = tokens_to_subtitle(combined_preds)
    # srt_content = generate_srt(subtitles_pred)
    # combined_gt = combine_sequences_gt(interval_frames_test)
    # combined_gt = merge_repeating_predictions(combined_gt)
    # combined_gt = merge_short_predictions(combined_gt)
    # subtitles_gt = tokens_to_subtitle(combined_gt)
    # a = IOU(subtitles_pred, subtitles_gt)
    # print("IOU value is ", a)
    # print("Segment percentage is ", segment_percentage(subtitles_pred, subtitles_gt))

  #  # Write SRT file
  #   with open("frame_by_frame.srt", "w") as f:
  #     f.write(srt_content)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='BOBSL inference on test data')
  args = parser.parse_args()
  batch_sizes = [16]
  for batch_size in batch_sizes:
    args.batch_size = batch_size
    main(args)   