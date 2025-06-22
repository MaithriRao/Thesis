import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F


class AutoregressiveEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout, bidirectional=True, autoregressive=False):
        super(AutoregressiveEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.autoregressive = autoregressive
        self.bidirectional = bidirectional
        
        lstm_input_size = input_size + output_size if autoregressive else input_size
        lstm_hidden_size = hidden_size // 2 if bidirectional else hidden_size
       
        self.lstm = nn.LSTM(lstm_input_size, lstm_hidden_size, num_layers, bidirectional=bidirectional, batch_first=True)
        self.fc_out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_seq, input_lengths):

        batch_size, max_seq_length, _ = input_seq.shape

        device = input_seq.device
        
        if self.autoregressive:
            outputs = torch.zeros(batch_size, max_seq_length, self.output_size, device=device)
            prev_output = torch.zeros(batch_size, 1, self.output_size, device=device)
            hidden = None

            for t in range(max_seq_length):
                current_input = torch.cat([input_seq[:, t:t+1, :], prev_output], dim=-1)
                       
                # Create a tensor of ones for the current time step
                current_lengths = torch.ones(batch_size, dtype=torch.long, device=device).cpu().long()
             
                packed_input = pack_padded_sequence(current_input, current_lengths, batch_first=True, enforce_sorted=False)
            
                output, hidden = self.lstm(packed_input, hidden)
                
                output, _ = pad_packed_sequence(output, batch_first=True)
              
                output = self.dropout(output)
                pred = self.fc_out(output)
                outputs[:, t:t+1, :] = pred
                prev_output = pred.detach()
        
            return outputs, hidden
        else:
            packed_input_seq = pack_padded_sequence(input_seq, input_lengths, batch_first=True, enforce_sorted=False)
            output, hidden = self.lstm(packed_input_seq)
            output, _ = pad_packed_sequence(output, batch_first=True)
            output = self.dropout(output)
            outputs = self.fc_out(output)
        
        return outputs

class Seq2Seq(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, src, feature_lengths):
        if isinstance(self.encoder, AutoregressiveEncoder) and self.encoder.autoregressive:
            outputs, _ = self.encoder(src, feature_lengths)
        else:
            outputs = self.encoder(src, feature_lengths)
        return outputs