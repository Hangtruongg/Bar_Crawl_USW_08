import torch
import torch.nn as nn

# -----------------------------
# Models
# -----------------------------
class FeedForwardNet(nn.Module):
    def __init__(self, input_size, hidden_sizes=[64,32], activation='relu', dropout=0.2, output_size=1):
        super().__init__()
        layers = []
        prev = input_size
        if activation=='relu':
            act_fn = nn.ReLU()
        elif activation=='tanh':
            act_fn = nn.Tanh()
        elif activation=='leakyrelu':
            act_fn = nn.LeakyReLU()
        else:
            act_fn = nn.ReLU()
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(act_fn)
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, output_size))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)

class Conv1DNet(nn.Module):
    """1D Conv on flattened sequence"""
    def __init__(self, input_size, hidden_channels=[16,32], kernel_size=3, output_size=1):
        super().__init__()
        layers = []
        prev_channels = 1  # treat input as single channel
        seq_len = input_size
        for out_ch in hidden_channels:
            layers.append(nn.Conv1d(prev_channels, out_ch, kernel_size=kernel_size, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool1d(2))
            prev_channels = out_ch
            seq_len = seq_len // 2  # after pooling
        self.conv = nn.Sequential(*layers)
        self.fc = nn.Linear(prev_channels*seq_len, output_size)
        
    def forward(self, x):
        x = x.unsqueeze(1)  # add channel dimension
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)