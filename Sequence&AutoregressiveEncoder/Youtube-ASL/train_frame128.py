import torch.nn as nn
import random
import numpy as np
import time
from tqdm import tqdm
import shutil
import argparse
import torch.optim as optim
from .frame_data import *
from model_frame import *
import torch.nn.functional as F
import os
from metrics import *
from torch.optim.lr_scheduler import ReduceLROnPlateau

def train(model, train_loader, val_loader, loss_fn, loss_fn1, device, config):

  random.seed(42)
  n_epochs = 25
  learning_rate = 1e-4
  best_f1 = 0 
  padding_index = 3
  early_stopping = EarlyStopping(patience=100, verbose=True)
  best_iou = -1
  threshold_likeliest=False

  # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#   scheduler =  StepLR(optimizer, step_size=10, gamma=0.1)
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.4, patience=3)
  # scheduler = ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.7) #min is for loss max is for accuracy, f1
  for epoch in range(n_epochs):
    model.train()
    total_loss = 0
    all_predictions = []
    all_labels = []
    all_ious = []

    for i, data in tqdm(enumerate(train_loader), total = len(train_loader)):
      features , labels, feature_lengths, label_lengths = data

      features = features.to(device)

      labels =  labels.to(device)
      
      optimizer.zero_grad()  

      outputs = model(features, feature_lengths)
      trg = labels.view(-1).long()
      outputs = outputs.view(-1, model.encoder.output_size)
      # loss = loss_fn(outputs, trg) # for corss entropy loss

      #nlloss
      mask = trg != padding_index
      loss = loss_fn((F.log_softmax(outputs, dim=-1))[mask], trg[mask])
      loss = loss.mean()

      # Backward pass
      loss.backward()
      optimizer.step()
      total_loss += loss.item()
                         

    avg_loss = total_loss / len(train_loader) 

    print(f"Epoch [{epoch+1}/{n_epochs}], Average training Loss: {avg_loss:.7f}")

    model.eval()
    val_loss = 0
    all_f1_scores = []

    with torch.no_grad():
      for i, data in tqdm(enumerate(val_loader), total = len(val_loader)):
        features , labels, feature_lengths, label_lengths = data

        features = features.to(device)
        labels =  labels.to(device)

        decoder_output = model(features, feature_lengths)

        trg = labels.view(-1).long()
        outputs = decoder_output.view(-1, model.encoder.output_size)
        # loss = loss_fn1(outputs, trg)

        mask = trg != padding_index
        loss = loss_fn1((F.log_softmax(outputs, dim=-1))[mask], trg[mask])
        loss = loss.mean()

        val_loss += loss.item()   
        preds = outputs.argmax(dim=1)
        all_predictions.extend(preds.cpu().numpy())  # Store predictions
        all_labels.extend(trg.cpu().numpy())   

    avg_val_loss = val_loss / len(val_loader) 
    scheduler.step(avg_val_loss)
    # Remove padding indices from predictions and labels
    mask_tensor = (torch.tensor(all_labels) != padding_index)
    all_predictions_tensor =  torch.tensor(all_predictions)   # Filter predictions
    all_labels_tensor = torch.tensor(all_labels)
    filtered_predictions = all_predictions_tensor[mask_tensor]  # Filter predictions
    filtered_labels = all_labels_tensor[mask_tensor] # Filter labels

    frame_f1_score = frame_f1(filtered_predictions, filtered_labels, average='macro') 
    
    # scheduler.step(frame_f1_score)

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
      # # Save the best model
      # torch.save(model.state_dict(), 'best_model64.pth')
      print(f"New best F1 score: {best_f1:.4f}")
    else:
      epochs_without_improvement += 1

    early_stopping(avg_val_loss, model)
    if early_stopping.early_stop:
      print("Early stopping triggered")
      break
   
   
   
class EarlyStopping:
  def __init__(self, patience=10, verbose=False, delta=0, path='checkpoint_f64.pth'):
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
    # torch.save({
    #   'model_state_dict': model.state_dict(),
    #   'val_loss': val_loss,
    #   'best_loss': self.best_loss,
    #   'counter': self.counter
    # }, self.path)
    self.val_loss_min = val_loss

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
 
def main(config):

  device = "cuda" if torch.cuda.is_available() else "cpu"
  # device = 'cpu'
  print(device)
  with open("asl.json", "r") as f:
    data = json.load(f)
      
  video_names_train = data["train"]
  video_names_val = data["val"]
  video_names_test = data["test"]

  ### here the frames_path is different from the one in the Dataloader ####
  data_loader = BOBSL_DataLoader1(
    BATCH_SIZE= config.batch_size,
    num_workers=2,
    train_set = video_names_train,
      val_set = video_names_val,
      test_set = [
          'YS_tSEI6560', # same as val now
      ],
      
    frames_path = '/ds/videos/opticalflow-BOBSL/ASL/features/extracted_features_new/resnet101/',
    sentences_path = '/ds/videos/opticalflow-BOBSL/ASL/subtitles/',
    ) 
  train_loader, _, class_weights_train, val_loader, _, class_weights_val, test_loader = data_loader.run()   

  input_size = 2048
  hidden_size = 128 #try 1024
  num_layers = [2, 4, 6, 8]
  enc_dropouts = [0.3, 0.2, 0.1, 0]
  output_size = 4
  bidirectional = True
  autoregressive = False

  for N_LAYERS in num_layers:
    for ENC_DROPOUT in enc_dropouts:
          
      encoder = AutoregressiveEncoder(input_size, hidden_size, output_size, N_LAYERS, ENC_DROPOUT, bidirectional, autoregressive)
      model = Seq2Seq(encoder)
      model = model.to(device)
      num_params = count_parameters(model)
      print(f"Total number of trainable parameters: {num_params}")

      print(f"Training with hidden_size={hidden_size}, num_layers={N_LAYERS}, enc_dropout={ENC_DROPOUT}")


      ignore_index = 3
      num_classes = 4 # Adjust this if you have more or fewer classes
      weights = torch.zeros(num_classes, dtype=torch.float32)

      weights[0] = class_weights_train[0]  # Outside
      weights[1] = class_weights_train[1]  # Beginning
      weights[2] = class_weights_train[2]

      weights = weights.to(device)    
      loss_fn = nn.NLLLoss(reduction='none', weight = weights)

      # loss_fn = nn.CrossEntropyLoss(weight=weights, ignore_index=ignore_index)


      weights_val = torch.zeros(num_classes, dtype=torch.float32)   
      weights_val[0] = class_weights_val[0]
      weights_val[1] = class_weights_val[1]
      weights_val[2] = class_weights_val[2]
      weights_val = weights_val.to(device)

      loss_fn1 = nn.NLLLoss(reduction='none', weight=weights_val)
      # loss_fn1 = nn.CrossEntropyLoss(weight= weights_val, ignore_index=ignore_index)


      train(model, train_loader, val_loader, loss_fn, loss_fn1, device, config)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='BOBSL motion stream')
  parser.add_argument('--epochs', default=20, type=int, metavar='N', help='number of total epochs')
  # parser.add_argument('--lr', default=1e-4, type=float, metavar='LR', help='initial learning rate')
  parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
  parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
  parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')

  args = parser.parse_args()

  batch_sizes = [16]
  for batch_size in batch_sizes:
    args.batch_size = batch_size
    main(args)