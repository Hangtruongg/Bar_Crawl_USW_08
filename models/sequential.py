# sequential.py
import torch
import torch.nn as nn

# ----------------------------------------------------
# LSTM — Unidirectional LSTM
# ----------------------------------------------------
class LSTMNet(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, num_layers=2, dropout=0.2):
        super(LSTMNet, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        out = out[:, -1, :]  # last time step
        out = self.fc(out)
        return out


# ----------------------------------------------------
# GRU — Gated Recurrent Unit
# ----------------------------------------------------
class GRUNet(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, num_layers=2, dropout=0.2):
        super(GRUNet, self).__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, hn = self.gru(x)
        out = out[:, -1, :]  # last time step
        out = self.fc(out)
        return out


# ----------------------------------------------------
# BiLSTM — Bidirectional LSTM
# ----------------------------------------------------
class BiLSTMNet(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, num_layers=2, dropout=0.2):
        super(BiLSTMNet, self).__init__()
        self.bilstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_size * 2, 1)  # *2 for bidirectional

    def forward(self, x):
        out, (hn, cn) = self.bilstm(x)
        out = out[:, -1, :]  # last time step
        out = self.fc(out)
        return out
