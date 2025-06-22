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
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc_hidden = nn.Linear(hidden_size * 2, hidden_size)  # Transform to match decoder hidden size
        self.fc_cell = nn.Linear(hidden_size * 2, hidden_size) 

    def forward(self, input_seq, input_lengths):
        # Sort sequences by length (required for packing)
        input_lengths_tensor = torch.tensor(input_lengths)
        sorted_lengths, sorted_indices = torch.sort(input_lengths_tensor, descending=True)
        input_seq_sorted = input_seq[sorted_indices]

        # Pack sequences
        packed_input_seq = pack_padded_sequence(input_seq_sorted, sorted_lengths, batch_first=True, enforce_sorted=True)

        # LSTM forward pass
        encoder_states, (hidden, cell) = self.lstm(packed_input_seq)

        # Unpack output
        encoder_states, _ = pad_packed_sequence(encoder_states, batch_first=True)
        encoder_states = self.dropout(encoder_states)

        # Reorder output sequences to match original order
        _, original_indices = torch.sort(sorted_indices)
        encoder_states = encoder_states[original_indices]
        hidden = hidden[:, original_indices]
        cell = cell[:, original_indices]

        hidden = self.fc_hidden(torch.cat((hidden[-2], hidden[-1]), dim=1).unsqueeze(0))  # Combine last layer's forward and backward hidden states
        cell = self.fc_cell(torch.cat((cell[-2], cell[-1]), dim=1).unsqueeze(0))

        hidden = hidden.repeat(self.num_layers, 1, 1)  # Repeat across layers
        cell = cell.repeat(self.num_layers, 1, 1) 


        return encoder_states, hidden, cell

class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers, dropout):
        super(Decoder, self).__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size 
        self.num_layers = num_layers
        self.rnn = nn.LSTM(self.hidden_size * 2 + output_size, hidden_size , num_layers, batch_first=True)  # No embedding
        self.energy = nn.Linear(self.hidden_size * 3, 1)
        self.fc = nn.Linear(self.hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=0)
        self.relu = nn.ReLU()


    def forward(self, x, encoder_states, hidden, cell):
        #  (one-hot encoded or class index)
        x = x.unsqueeze(1)  # Add sequence dimension (1, N, hidden_size*2)
        
        sequence_length = encoder_states.shape[1]
        batch_size = encoder_states.shape[0]
        
        h_reshaped = hidden[-1].unsqueeze(1).repeat(1, sequence_length, 1)  # (batch_size, seq_length, hidden_size)


        concat = torch.cat((h_reshaped, encoder_states), dim=2)

        # Calculate attention weights
        energy = self.relu(self.energy(concat))
        attention = self.softmax(energy)
        # print("Attention Weights:", attention)

        # Compute context vector
        context_vector = torch.bmm(attention.permute(0, 2, 1), encoder_states)  # (N, 1, hidden_size*2) 

        # Combine context vector with input
        rnn_input = torch.cat((context_vector, x), dim=2) 

        assert rnn_input.shape[2] == self.rnn.input_size, f"Expected input size {self.rnn.input_size}, but got {rnn_input.shape[2]}"
    
        outputs, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))

        outputs = self.dropout(outputs) 


        predictions = self.fc(outputs.squeeze(1))  # (N, output_size)

        return predictions, hidden, cell

class Seq2SeqAttention(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2SeqAttention, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, source, source_length, target, teacher_force_ratio):
        batch_size = source.shape[0]
        target_len = target.shape[1]
        target_vocab_size = self.decoder.output_size

        outputs = torch.zeros(batch_size, target_len, target_vocab_size).to(self.device)
        encoder_states, hidden, cell = self.encoder(source, source_length)

        # First input will be <SOS> token
        x = target[:, 0, :] 

        for t in range(1, target_len):
            output, hidden, cell = self.decoder(x, encoder_states, hidden, cell)
            outputs[:, t, :] = output
            best_guess = output.argmax(1)

            teacher_force =random.random() < teacher_force_ratio
            x = target[:, t, :] if teacher_force else F.one_hot(best_guess, target_vocab_size).float()

        return outputs    

