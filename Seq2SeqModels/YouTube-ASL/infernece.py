import torch
import torchvision
from torchvision import datasets
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model_no_attention import *
from model_attention import *
import argparse
from .data import *
from tqdm import tqdm
import torch.nn.functional as F
from metrics import *
import json
import numpy as np
import heapq

def beam_search_inference(model, test_loader, device, beam_width, use_attention):
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
          if use_attention:
            decoder_output, hidden, cell = model.decoder(decoder_input, encoder_states, hidden, cell)
          else:
            decoder_output, hidden, cell = model.decoder(decoder_input, hidden, cell) # for no_attention
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


def combine_sequences_pred(all_predictions, all_softmax_outputs, sequence_frames=375):
  #here sequence_frames= 375
  combined_preds = []
  current_time = 0

  for preds_chunk, softmax_chunk in zip(all_predictions, all_softmax_outputs):
    print("preds_chunk", preds_chunk)
    # Calculate durations
    probabilities = []
    for pred, soft in zip(preds_chunk, softmax_chunk):
      # Assuming soft is a 1D tensor
      probability = soft[pred].item()  # Get the probability of the predicted class
      probabilities.append(probability)

    total_prob = sum(probabilities)
    durations = [d / total_prob * sequence_frames for d in probabilities]

    for pred, dur in zip(preds_chunk, durations):
      combined_preds.append((current_time, current_time + dur, pred))
      current_time += dur

  return combined_preds


def combine_sequences_gt(all_interval_frames_test, fps):
  combined_gt = [] 
  current_time = round(fps / 5) # optical flow calcualtion or ASL

  for interval_frames_test in all_interval_frames_test:
    for frame_info in interval_frames_test:
      label = frame_info["label"]
      start_time = current_time
      current_time += 1
      end_time = current_time
      combined_gt.append((start_time, end_time, label))

  return combined_gt


def merge_repeating_segments(preds):
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


def merge_short_segments(preds, min_segment_duration=1.0):
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


def generate_srt(subtitles, fps):
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
    print(type(start_pred), type(end_pred), type(start_gt), type(end_gt))
    segments_pred[start_pred:(end_pred + 1)] = 1
    segments_gt[start_gt:(end_gt + 1)] = 1

  intersection = np.logical_and(segments_pred, segments_gt)
  union = np.logical_or(segments_pred, segments_gt)

  if np.sum(union) == 0:
    return 1 if np.sum(intersection) == 0 else 0

  return float(np.sum(intersection) / np.sum(union))

def segment_percentage(subtitles_pred, subtitles_gt):
  if len(subtitles_gt) == 0:
    return 1 if len(subtitles_pred) == 0 else len(subtitles_pred)

  return len(subtitles_pred) / len(subtitles_gt)


def main(config):

  device = "cuda" if torch.cuda.is_available() else "cpu"
  print(device)

  with open("asl.json", "r") as f:
    data = json.load(f)
      
  video_names_train = data["train"]
  video_names_val = data["val"]
  video_names_test = data["test"]
  test_video_id = 'SGlWxxp-IBA' 
  data_loader = ASL_DataLoader(
    BATCH_SIZE= config.batch_size,
    num_workers=1,
    train_set=['xNzCVbdwnEI'],  
    val_set=['xNzCVbdwnEI'],     
    test_set= [test_video_id], 
    
    frames_path = '/ds/videos/opticalflow-BOBSL/ASL/features/extracted_features/resnet101/',
    sentences_path = '/ds/videos/opticalflow-BOBSL/ASL/subtitles/',
  )
  _, _, _, _, test_loader, class_weights_test, interval_frames_test, video_id_to_fps = data_loader.run()
  input_size = 2048  
  hidden_size = 128
  N_LAYERS = 2
  ENC_DROPOUT = 0.2
  DEC_DROPOUT = 0.1

  encoder = Encoder(input_size=input_size, hidden_size=hidden_size, num_layers = N_LAYERS, dropout = ENC_DROPOUT)
  decoder = Decoder(hidden_size=hidden_size, output_size=5, num_layers = N_LAYERS, dropout = DEC_DROPOUT)

  encoder = encoder.to(device)
  decoder = decoder.to(device)

  if config.use_attention:
    model = Seq2SeqAttention(encoder, decoder, device).to(device)
    print("Using Seq2Seq model with Attention")
  else:
    model = Seq2SeqNoAttention(encoder, decoder, device).to(device)
    print("Using Seq2Seq model without Attention")
  
  best_model_state_dict = torch.load("best_model_att128_b16_29.pth", map_location=torch.device(device))
  model.load_state_dict(best_model_state_dict) #for f1
  # model.load_state_dict(best_model_state_dict['model_state_dict']) # for loss 

  beam_width = 4
  #Beam search
  all_predictions, all_ground_truths, all_softmax_outputs = beam_search_inference(model, test_loader, device, beam_width, use_attention)
  combined_preds = combine_sequences_pred(all_predictions, all_softmax_outputs)
  combined_preds = merge_repeating_segments(combined_preds)
  combined_preds = merge_short_segments(combined_preds)

  subtitles_pred = tokens_to_subtitle(combined_preds)

  combined_gt = combine_sequences_gt(interval_frames_test, video_id_to_fps[test_video_id])
  combined_gt = merge_repeating_segments(combined_gt)
  combined_gt = merge_short_segments(combined_gt)
  subtitles_gt = tokens_to_subtitle(combined_gt)

  srt_content = generate_srt(subtitles_pred, video_id_to_fps[test_video_id])

  # Write SRT file
  with open("SGlWxxp-IBA_sample1", "w") as f:
    f.write(srt_content)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description= 'Inference on Youtube_ASL test data')
  parser.add_argument('--use_attention', action='store_true', help='Use the Seq2Seq model with attention')
  args = parser.parse_args()
  batch_sizes = [1]
  for batch_size in batch_sizes:
    args.batch_size = batch_size
    main(args)