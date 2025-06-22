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
import json
from torch.utils.data import Dataset, DataLoader
from .data import *
import sys
import math
from .attention import *
import torch.nn.functional as F
import os
from pathlib import Path
from .loss import *
from metrics import *

def train(model, decoder, train_loader, val_loader, loss_fn, loss_fn_val, device, config):

  random.seed(42)
  best_f1 = 0.0
  n_epochs = 45
  early_stopping = EarlyStopping(patience=100, verbose=True)
  best_model_state_dict = None  
  padding_index = 3
  best_iou = -1

  all_validation_losses = []

  optimizer = torch.optim.Adam(model.parameters(), lr=1e-05)
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.7)

  for epoch in range(n_epochs):
    model.train()
    total_loss = 0
    all_predictions = []
    all_labels = []

    for i, data in tqdm(enumerate(train_loader), total = len(train_loader)):
      features , labels, feature_lengths, label_lengths = data

      features = features.to(device)

      labels =  labels.to(device)
      
      optimizer.zero_grad()  

      one_hot_labels = F.one_hot(labels.long(), num_classes=decoder.output_size).float()

      output = model(features, feature_lengths, one_hot_labels, teacher_force_ratio=0.7)
      output_dim = output.shape[-1]
        
      output = output[:, 1:, :].reshape(-1, output_dim)  # Remove BOS token from output
  
      trg = labels[:, 1:].reshape(-1).long() 

      # for nlloss   
      mask = trg != padding_index 
      loss = loss_fn((F.log_softmax(output, dim=-1))[mask], trg[mask])
      loss = loss.mean()

      # Backward pass
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
      optimizer.step()
      total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    print(f"Epoch [{epoch+1}/{n_epochs}], Average training Loss: {avg_loss:.7f}")

    model.eval()
    val_loss = 0
    all_ious = []
    all_f1s = []
    all_percentage_segments = [] 
    with torch.no_grad():
      for i, data in tqdm(enumerate(val_loader), total = len(val_loader)):
        features , labels, feature_lengths, label_lengths = data

        features = features.to(device)
        labels =  labels.to(device)

        one_hot_labels = F.one_hot(labels.long(), num_classes=decoder.output_size).float()
        
        output = model(features, feature_lengths, one_hot_labels, teacher_force_ratio=0.0)
        output_dim = output.shape[-1]
        
        output = output[:, 1:, :].reshape(-1, output_dim)  # Remove BOS token from output
              
        trg = labels[:, 1:].reshape(-1).long()  

        #for nlloss
        mask = trg != padding_index 
        loss = loss_fn_val((F.log_softmax(output, dim=-1))[mask], trg[mask])
        loss = loss.mean()

        val_loss += loss.item()

        preds = output.argmax(dim=1)
        all_predictions.extend(preds.cpu().numpy())  # Store predictions
        all_labels.extend(trg.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader) 
    
    all_validation_losses.append(avg_val_loss)
    # Remove padding indices from predictions and labels
    mask_tensor = (torch.tensor(all_labels) != padding_index)
    all_predictions_tensor =  torch.tensor(all_predictions)   # Filter predictions
    all_labels_tensor = torch.tensor(all_labels)
    filtered_predictions = all_predictions_tensor[mask_tensor]  # Filter predictions
    filtered_labels = all_labels_tensor[mask_tensor] # Filter labels

    frame_f1_score = frame_f1(filtered_predictions, filtered_labels, average='macro') 
    scheduler.step(frame_f1_score)

    ious = calculate_iou(filtered_predictions.cpu().numpy(), filtered_labels.cpu().numpy(), classes_of_interest = [1, 2])
    mean_iou = np.mean(ious)


    gt_segments = count_segments(filtered_labels)
    pred_segments = count_segments(filtered_predictions)
    percentage_of_segments = (pred_segments / gt_segments * 100) if gt_segments > 0 else 0

    print(f"Epoch [{epoch+1}/{n_epochs}], Average validation Loss: {avg_val_loss:.5f}")
    print(f"Epoch [{epoch+1}/{n_epochs}], iou: {mean_iou:.4f}")
    print(f"Epoch [{epoch+1}/{n_epochs}], Percentage of Segments: {percentage_of_segments:.2f}%")
    print(f"Epoch [{epoch+1}/{n_epochs}], Validation F1 Score: {frame_f1_score:.5f}")
    
    if mean_iou > best_iou:
        best_iou = mean_iou 
    print(f"Best IoU so far: {best_iou:.7f}")  

    if frame_f1_score > best_f1:
      best_f1 = frame_f1_score
      epochs_without_improvement = 0
      torch.save(model.state_dict(), 'best_model_att128_b16_new_feat.pth') # best_model_att128_ep100
      print(f"New best F1 score: {best_f1:.4f}")
    else:
      epochs_without_improvement += 1

    early_stopping(avg_val_loss, model)
    if early_stopping.early_stop:
      print("Early stopping triggered")
      break


class EarlyStopping:
  def __init__(self, patience=10, verbose=False, delta=0, path='checkpoint_att128_b16_new_feat.pth'): # checkpoint_att128_ep100
    self.patience = patience
    self.verbose = verbose
    self.counter = 0
    self.best_loss = None
    self.early_stop = False
    self.val_loss_min = float('inf')
    self.delta = delta
    self.path = path

  def __call__(self, val_loss, model):
    if self.best_loss is None:
      self.best_loss = val_loss
      self.save_checkpoint(val_loss, model)
    elif val_loss > self.best_loss + self.delta:
      self.counter += 1
      if self.counter >= self.patience:
        self.early_stop = True
    else:
      self.best_loss = val_loss
      self.save_checkpoint(val_loss, model)
      self.counter = 0

  def save_checkpoint(self, val_loss, model):
    if self.verbose:
      print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
    torch.save({
      'model_state_dict': model.state_dict(),
      'val_loss': val_loss,
      'best_loss': self.best_loss,
      'counter': self.counter
    }, self.path)
    self.val_loss_min = val_loss

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
def main(config):

  device = "cuda" if torch.cuda.is_available() else "cpu"
  
  with open("asl.json", "r") as f:
    data = json.load(f)
      
  video_names_train = data["train"]
  video_names_val = data["val"]
  video_names_test = data["test"]

  #Prepare DataLoader
  data_loader = ASL_DataLoader(
  BATCH_SIZE= config.batch_size,
  num_workers=2,
  train_set= video_names_train,
  val_set = video_names_val,  
  test_set=video_names_test,
  
  frames_path = '/ds/videos/opticalflow-BOBSL/ASL/features/extracted_features_new/resnet101/',
  sentences_path = '/ds/videos/opticalflow-BOBSL/ASL/subtitles/',
  )
  train_loader, class_weights, val_loader, class_weights_val, _, _, _, _ = data_loader.run()

  input_size = 2048  
  hidden_size = 128 
  num_layer = 2
  enc_dropout = 0.2
  dec_dropout = 0.1


  print(f"Training with  batch_size = {config.batch_size}, hidden_size={hidden_size}, num_layers={num_layer}, enc_dropout={enc_dropout}, dec_dropout={dec_dropout}")
  encoder = Encoder(input_size = input_size, hidden_size = hidden_size, num_layers = num_layer, dropout = enc_dropout).to(device)
  decoder = Decoder(hidden_size =hidden_size, output_size = 5, num_layers = num_layer, dropout = dec_dropout).to(device)

  # Initialize Seq2Seq model

  model = Seq2Seq(encoder, decoder, device).to(device)
  ignore_index = 3

  num_classes = 5 
  weights = torch.zeros(num_classes, dtype=torch.float32)

  weights[0] = class_weights[0] 
  weights[1] = class_weights[1]  
  weights[2] = class_weights[2]  


  weights = weights.to(device)
  loss_fn = nn.NLLLoss(reduction='none', weight = weights)

  weights_val = torch.zeros(num_classes, dtype=torch.float32)   
  weights_val[0] = class_weights_val[0]
  weights_val[1] = class_weights_val[1]
  weights_val[2] = class_weights_val[2]

  weights_val = weights_val.to(device)

  loss_fn_val = nn.NLLLoss(reduction='none', weight = weights_val)

  train(model, decoder, train_loader, val_loader, loss_fn, loss_fn_val, device, config)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='ASL data')
  parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
  parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
  parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')

  args = parser.parse_args()

  batch_sizes = [16]
  for batch_size in batch_sizes:
    args.batch_size = batch_size
    main(args)


