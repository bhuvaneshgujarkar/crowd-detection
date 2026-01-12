import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(512, 128, batch_first=True)
        self.fc = nn.Linear(128, 2)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])
