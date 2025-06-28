import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

class InvarTSModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, bidirection=False, num_layers=2, lambda_reg=0.1):
        super(InvarTSModel, self).__init__()
        self.input_dim = input_dim  # Number of input features (e.g., temperature, humidity, etc.)
        self.hidden_dim = hidden_dim  # Hidden dimension
        self.bidirection = bidirection
        self.num_layers = num_layers
        self.lambda_reg = lambda_reg  # Regularization parameter lambda

        # Define LSTM layer
        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=bidirection)

        # Fully connected layers
        if bidirection:
            self.fc1 = nn.Linear(hidden_dim * 2, 128)  # Account for bidirectional output
        else:
            self.fc1 = nn.Linear(hidden_dim, 128)
        self.fc2 = nn.Linear(128, 1)  # Output a single SSC value

        # Invariant weight matrix W_ino (scalar for SSC regression, adjusted to 1x1)
        self.W_ino = nn.Parameter(torch.ones(1))

        # Dropout and activation
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x, flag = "pred"):
        # Ensure input is float32
        x = x.float()
        if len(x.shape) == 2:  # Only batch_size and input_dim, missing seq_len
            x = x.unsqueeze(1)  # Add seq_len dimension

        # LSTM forward pass
        lstm_out, _ = self.lstm(x)  # lstm_out: (batch_size, seq_len, hidden_dim or hidden_dim*2 if bidirectional)
        
        # Take the last timestep's output
        x = lstm_out[:, -1, :]  # (batch_size, hidden_dim or hidden_dim*2)
        
        # Fully connected layers
        x = self.relu(self.fc1(x))  # Apply ReLU activation
        rep = self.dropout(x)  # Apply dropout
        rep = self.fc2(rep)  # (batch_size, 1)
        
        # Apply W_ino with Hadamard product (broadcasting to batch_size)
        Y = rep * self.W_ino  # (batch_size, 1)
        if flag == "feat":
            return x
        else:
            return Y.squeeze()  # (batch_size,)
