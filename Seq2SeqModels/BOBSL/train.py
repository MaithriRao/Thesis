import torch.nn as nn
import random
import numpy as np
import time
from tqdm import tqdm
import shutil
import argparse
import torch.optim as optim
from .Dataloader import *
from Seq2SeqModels.model_no_attention import *
from Seq2SeqModels.model_attention import *
import torch.nn.functional as F
import os
from metrics import *

def move_data_to_device(data, device):
    features, labels, feature_lengths, label_lengths = data
    features = features.to(device)
    labels = labels.to(device)
    return features, labels, feature_lengths, label_lengths

def process_model_output(model, features, feature_lengths, labels, one_hot_labels, teacher_force_ratio, padding_index):
    # print("Features shape:", features.shape) # [batch_size, seq_len, input_size] # [16, 375, 2048]
    output = model(features, feature_lengths, one_hot_labels, teacher_force_ratio=teacher_force_ratio) # [batch_size, seq_len, num_classes] [16, 9, 5]
    output_dim = output.shape[-1] # num_classes
    output = output[:, 1:, :].reshape(-1, output_dim) # [ batch_size * (output_seq_len -1), num_classes] # [128, 5] # Remove BOS token from output
    # print("Labels shape before reshape:", labels.shape) # [batch_size, num_classes] # [16, 9]
    trg = labels[:, 1:].reshape(-1).long()  # (batch_size * labels-1 ) # [128] # Remove BOS token from target labels
    # print("Target shape:", trg.shape)
    output = F.log_softmax(output, dim=-1)
    mask = trg != padding_index
    return output, trg, mask

def calculate_metrics(all_predictions, all_labels, padding_index):
    mask_tensor = (torch.tensor(all_labels) != padding_index)
    all_predictions_tensor = torch.tensor(all_predictions)
    all_labels_tensor = torch.tensor(all_labels)
    filtered_predictions = all_predictions_tensor[mask_tensor]
    filtered_labels = all_labels_tensor[mask_tensor]
    f1_score = frame_f1(filtered_predictions, filtered_labels, average='macro')
    ious = calculate_iou(filtered_predictions.cpu().numpy(), filtered_labels.cpu().numpy(), classes_of_interest=[1, 2])
    mean_iou = np.mean(ious)
    gt_segments = count_segments(filtered_labels)
    pred_segments = count_segments(filtered_predictions)
    percentage_of_segments = calculate_segment_percentage(pred_segments, gt_segments)
    return f1_score, mean_iou, percentage_of_segments

def train_epoch(model, train_loader, optimizer, loss_fn, decoder, device, padding_index):
    model.train()
    total_loss = 0
    for data in tqdm(train_loader, total=len(train_loader), desc="Training"):
        features, labels, feature_lengths, label_lengths = move_data_to_device(data, device)
        optimizer.zero_grad()
        one_hot_labels = F.one_hot(labels.long(), num_classes=decoder.output_size).float()
        output, trg, mask = process_model_output(model, features, feature_lengths, labels, one_hot_labels, 0.7, padding_index)
        loss = loss_fn(output[mask], trg[mask]).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def validate_epoch(model, val_loader, loss_fn_val, decoder, device, padding_index):
    model.eval()
    val_loss = 0
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for data in tqdm(val_loader, total=len(val_loader), desc="Validation"):
            features, labels, feature_lengths, label_lengths = move_data_to_device(data, device)
            one_hot_labels = F.one_hot(labels.long(), num_classes=decoder.output_size).float()
            output, trg, mask = process_model_output(model, features, feature_lengths, labels, one_hot_labels, 0.0, padding_index)
            loss = loss_fn_val(output[mask], trg[mask]).mean()
            val_loss += loss.item()
            preds = output.argmax(dim=1)
            all_predictions.extend(preds.cpu().numpy())
            all_labels.extend(trg.cpu().numpy())
    return val_loss / len(val_loader), all_predictions, all_labels


class EarlyStopping:
  def __init__(self, patience, verbose=False, use_attention=False, delta=0, best_model_path='checkpoint.pth'):
    self.patience = patience
    self.verbose = verbose
    self.counter = 0
    self.best_loss = None
    self.early_stop = False
    self.val_loss_min = float('inf')
    self.delta = delta
    attention_suffix = "_attention" if use_attention else "_no_attention"
    self.best_model_path = f'{best_model_path}{attention_suffix}.pth'

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
        print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
    torch.save({'model_state_dict': model.state_dict(), 'val_loss': val_loss}, self.best_model_path)
    self.val_loss_min = val_loss

    
def train(model, decoder, train_loader, val_loader, loss_fn, loss_fn_val, device, config):
    random.seed(42)
    best_f1 = 0.0
    best_iou = -1
    n_epochs = config.n_epochs
    early_stopping = EarlyStopping(patience=10, verbose=True, use_attention=config.use_attention)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.6, patience=4)
    padding_index = 3

    attention_prefix = "attention" if config.use_attention else "no_attention"
    best_model_filename = f'best_model_{attention_prefix}.pth'

    for epoch in range(n_epochs):
        avg_train_loss = train_epoch(model, train_loader, optimizer, loss_fn, decoder, device, padding_index)
        avg_val_loss, all_predictions, all_labels = validate_epoch(model, val_loader, loss_fn_val, decoder, device, padding_index)
        f1_score, mean_iou, percentage_of_segments = calculate_metrics(all_predictions, all_labels, padding_index)
        scheduler.step(f1_score)

        print(f"Epoch [{epoch+1}/{n_epochs}], Train Loss: {avg_train_loss:.5f}, Val Loss: {avg_val_loss:.5f}, IoU: {mean_iou:.4f}, Segments: {percentage_of_segments:.2f}%, F1: {f1_score:.5f}")
        
        if mean_iou > best_iou:
            best_iou = mean_iou
            print(f"Best IoU so far: {best_iou:.7f}")

        if f1_score > best_f1:
            best_f1 = f1_score
            torch.save(model.state_dict(), best_model_filename)
            print(f"New best F1 score: {best_f1:.4f}")

        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    if best_model_state_dict is not None:
        model.load_state_dict(best_model_state_dict)
    return model


def create_weighted_loss(class_weights, device, num_classes=5):
    """
    Creates a weighted loss function using the provided class weights.

    Args:
        class_weights (list or tensor): A list or tensor of class weights.
        device (torch.device): The device to which the weights should be moved.
        num_classes (int): The number of classes. Default is 5.

    Returns:
        loss_fn (nn.Module): The weighted loss function.
    """

    # Initialize weights tensor
    weights = torch.zeros(num_classes, dtype=torch.float32)

    # Assign provided class weights
    for i, weight in enumerate(class_weights):
        weights[i] = weight

    # Move weights to the specified device
    weights = weights.to(device)

    # Create and return the weighted loss function
    loss_fn = nn.NLLLoss(reduction='none', weight=weights)
    return loss_fn    

def setup_device():
    """Sets up the device (CPU or CUDA)."""
    return "cuda" if torch.cuda.is_available() else "cpu"

def setup_dataloaders(config):
    """Sets up the data loaders."""
    data_loader = BOBSL_DataLoader1(
        BATCH_SIZE=config.batch_size,
        num_workers=2,
        train_set = {'manually-aligned': ['5085357672350628540',
              '5099449889545545450',
              '5099449889545545450', '5085344787448740525','5086465773912997411',
              '5090916219025116054', '5092765202446045704', '5097975856769557081', '5103182645622502287',
              '514625816112nll4252727', '5172231547180314349', '5218949624445806415', '5366409594815891602',
              '5391892065246550398', '5525084584552992937'],
              'audio-aligned': [ '5087953980081062580', '5090546422340930233',
              '5093160768934007504', '5093521546186871869', '5093895208341624036', '5094257274914886988',
              '5099091689273058848', '5101325932090649776', '5104636062555440832', '5105777664862718183',
              '5106137153625393471', '5108723153434317714', '5113588492387230304', '5120215198258444599',
              '5125058632047937012', '5129558040617440365', '5130309229567302144', '5132160790798818890',
              '5134749367588120522', '5138091711137870315', '5139568320894207394',
              '5141035910389041992', '5142915818404712489', '5145138463150184862', '5145510836814748271'
              ]},

        val_set = {'manually-aligned': ['5224144816887051284',
                    '5242317681679687839', 
                    '5294309549287947552', '5439409006429129628'],
                    'audio-aligned': ['5213407827313563421', '5220817935219567388', '5221198038995053656', '5223769865411899831',
                    '5225978337595504968', '5228592684188581488'
                    ]},

        test_set = {'manually-aligned': ['5402656112283796519']},    
        
        frames_path = '/ds/videos/opticalflow-BOBSL/bobsl/extracted_features_1/resnet101/',
        sentences_path = '/ds/videos/opticalflow-BOBSL/',
    )
    return data_loader.run()  
    
def setup_model_and_loss(device, class_weights, class_weights_val, config):
    """Sets up the model and loss functions."""
    input_size = 2048
    hidden_size = 128
    num_layer = 2
    enc_dropout = 0.2
    dec_dropout = 0.1

    encoder = Encoder(input_size=input_size, hidden_size=hidden_size, num_layers=num_layer, dropout=enc_dropout)
    decoder = Decoder(hidden_size=hidden_size, output_size=5, num_layers=num_layer, dropout=dec_dropout)

    if config.use_attention:
        model = Seq2SeqAttention(encoder, decoder, device).to(device)
        print("Using Seq2Seq model with Attention")
    else:
        model = Seq2SeqNoAttention(encoder, decoder, device).to(device)
        print("Using Seq2Seq model without Attention")


    train_loss = create_weighted_loss(class_weights, device)
    val_loss = create_weighted_loss(class_weights_val, device)
    return model, decoder, train_loss, val_loss   

def main(config):
    device = setup_device()
    train_loader, class_weights, val_loader, class_weights_val, _, _, _ = setup_dataloaders(config)
    model, decoder, train_loss, val_loss = setup_model_and_loss(device, class_weights, class_weights_val, config)
    train(model, decoder, train_loader, val_loader, train_loss, val_loss, device, config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BOBSL data training for Sign Language Segmentation')
    parser.add_argument('--use_attention', action='store_true', help='Use the Seq2Seq model with attention')
    parser.add_argument('--n_epochs', default=45, type=int, help='number of training epochs')
    parser.add_argument('--learning_rate', default=1e-05, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=16, type=int, help='batch size')

    args = parser.parse_args()
    main(args)  

# run code
# python -m train --use_attention # with attention
# python -m train # wihtout attention


