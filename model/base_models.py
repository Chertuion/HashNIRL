import torch.nn as nn

class OneDCNN(nn.Module):
    def __init__(self, input_dim, hidden_size=128):
        super(OneDCNN, self).__init__()

        self.conv1 = nn.Conv1d(1, hidden_size//2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_size//2, hidden_size, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(hidden_size, hidden_size * 2, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(hidden_size * 2 * input_dim, hidden_size)
        self.fc2 = nn.Linear(128, 1)

        self.dropout = nn.Dropout(0.4)
        self.relu = nn.ReLU()

    def forward(self, x, return_data = "pred"):
        x = x.float()
        x = x.unsqueeze(1)
        x = self.relu(self.conv1(x))
        x = self.dropout(x)
        x = self.relu(self.conv2(x))
        x = self.dropout(x)
        x = self.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x_rep = self.relu(self.fc1(x))
        output = self.fc2(x_rep)
        if return_data == "pred":
            return output.squeeze()
        else:
            return output.squeeze(), x_rep

class LSTM(nn.Module):
    def __init__(self, input_dim, bidirection, hidden_size=128, num_layers=2):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=bidirection)
        if bidirection:
            self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        else:
            self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, x, return_data = "pred"):
        x = x.float()
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]
        x = self.relu(self.fc1(x))
        x_rep = self.dropout(x)
        output = self.relu(self.fc2(x_rep))
        if return_data == "pred":
            return output.squeeze()
        else:
            return output.squeeze(), x_rep

import torch
import torch.nn as nn

class GRU(nn.Module):
    def __init__(self, input_dim, bidirection, hidden_size=128, num_layers=2):
        super(GRU, self).__init__()
        self.gru = nn.GRU(input_size=input_dim,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        batch_first=True,
                        bidirectional=bidirection)
        if bidirection:
            self.fc1 = nn.Linear(hidden_size * 2, 128)
        else:
            self.fc1 = nn.Linear(hidden_size, 128)
        self.fc2 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.float()
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        gru_out, _ = self.gru(x)
        x = gru_out[:, -1, :]
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        output = self.fc2(x)
        return output.squeeze()

import torch
import torch.nn as nn
from mamba_ssm import Mamba

class MambaNIR(nn.Module):
    def __init__(self, input_dim, bidirection, hidden_size=380, num_layers=2):
        super(MambaNIR, self).__init__()
        self.hidden_size = hidden_size
        self.bidirection = bidirection
        self.num_layers = num_layers
        self.mamba_layers = nn.ModuleList()
        current_dim = input_dim
        for i in range(num_layers):
            if bidirection:
                self.mamba_layers.append(
                    Mamba(
                        d_model=current_dim,
                        d_state=16,
                        d_conv=4,
                        expand=2
                    )
                )
                self.mamba_layers.append(
                    Mamba(
                        d_model=current_dim,
                        d_state=16,
                        d_conv=4,
                        expand=2
                    )
                )
                current_dim = hidden_size * 2
            else:
                self.mamba_layers.append(
                    Mamba(
                        d_model=current_dim,
                        d_state=16,
                        d_conv=4,
                        expand=2
                    )
                )
                current_dim = hidden_size
        fc_input_dim = hidden_size * 4 if bidirection else hidden_size
        self.fc1 = nn.Linear(fc_input_dim, 128)
        self.fc2 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.float()
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        for i, mamba in enumerate(self.mamba_layers):
            if self.bidirection and i % 2 == 1:
                x_backward = torch.flip(x, dims=[1])
                x_backward = mamba(x_backward)
                x = torch.cat([x, torch.flip(x_backward, dims=[1])], dim=-1)
            else:
                x = mamba(x)
        x = x[:, -1, :]
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        output = self.fc2(x)
        return output.squeeze()

class HashMambaNIR(nn.Module):
    def __init__(self, input_dim, bidirection, hidden_size=380, num_layers=2, hash_bit=32):
        super(HashMambaNIR, self).__init__()
        self.hidden_size = hidden_size
        self.bidirection = bidirection
        self.num_layers = num_layers
        self.mamba_layers = nn.ModuleList()
        current_dim = input_dim
        for i in range(num_layers):
            if bidirection:
                self.mamba_layers.append(
                    Mamba(
                        d_model=current_dim,
                        d_state=16,
                        d_conv=4,
                        expand=2
                    )
                )
                self.mamba_layers.append(
                    Mamba(
                        d_model=current_dim,
                        d_state=16,
                        d_conv=4,
                        expand=2
                    )
                )
                current_dim = hidden_size * 2
            else:
                self.mamba_layers.append(
                    Mamba(
                        d_model=current_dim,
                        d_state=16,
                        d_conv=4,
                        expand=2
                    )
                )
                current_dim = hidden_size
        fc_input_dim = hidden_size * 4 if bidirection else hidden_size
        self.hash_fc = nn.Linear(fc_input_dim, hash_bit)
        self.fc1 = nn.Linear(hash_bit, 128)
        self.fc2 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = x.float()
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        for i, mamba in enumerate(self.mamba_layers):
            if self.bidirection and i % 2 == 1:
                x_backward = torch.flip(x, dims=[1])
                x_backward = mamba(x_backward)
                x = torch.cat([x, torch.flip(x_backward, dims=[1])], dim=-1)
            else:
                x = mamba(x)
        x = x[:, -1, :]
        hash_binary = torch.sign(self.tanh(self.hash_fc(x)))
        x = self.fc1(hash_binary)
        x = self.dropout(x)
        output = self.fc2(x)
        return output.squeeze()
