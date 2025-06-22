import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import random
import math


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bilstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True) #batch_first=True is removed
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_seq, input_lengths):
        input_lengths_tensor = torch.tensor(input_lengths)
        sorted_lengths, sorted_indices = torch.sort(input_lengths_tensor, descending=True)
        input_seq_sorted = input_seq[sorted_indices]
        # Pack sequences
        packed_input_seq = pack_padded_sequence(input_seq_sorted, sorted_lengths, batch_first=True, enforce_sorted=True)

        # LSTM forward pass
        output, (hidden, cell) = self.bilstm(packed_input_seq)

        # Unpack output
        output, _ = pad_packed_sequence(output, batch_first=True)


        output = self.dropout(output)

        # Reorder output sequences to match original order
        _, original_indices = torch.sort(sorted_indices)
        output = output[original_indices]
        hidden = hidden[:, original_indices]
        cell = cell[:, original_indices]
        return output, hidden, cell


class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers, dropout):
        super().__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.LSTM(input_size = 5, hidden_size = hidden_size, num_layers = num_layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.fc_out = nn.Linear(hidden_size * 2, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        input = input.unsqueeze(1)
        output, (hidden, cell) = self.rnn(input, (hidden, cell))
        prediction = self.fc_out(output.squeeze(1))
        return prediction, hidden, cell      

class Seq2SeqNoAttention(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2SeqNoAttention, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device       
        assert encoder.hidden_size == decoder.hidden_size, "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.num_layers == decoder.num_layers, "Encoder and decoder must have equal number of layers!"

    def forward(self, src, feature_lengths, trg, teacher_force_ratio):

     
        batch_size = src.shape[0]
        max_sequence_length = trg.shape[1]
        trg_vocab_size = self.decoder.output_size
        outputs = torch.zeros(batch_size, max_sequence_length, trg_vocab_size).to(self.device)
        output, hidden, cell = self.encoder(src, feature_lengths)
    
        input = trg[:, 0, :]  # Assuming trg is one-hot encoded

        for t in range(1, max_sequence_length):
            output, hidden, cell = self.decoder(input, hidden, cell)     

            outputs[:, t, :] = output
            teacher_force = torch.rand(1).item() < teacher_force_ratio
            top1 = output.argmax(1)
            input = trg[:, t, :] if teacher_force else F.one_hot(top1, num_classes=trg_vocab_size).float()
     
        return outputs