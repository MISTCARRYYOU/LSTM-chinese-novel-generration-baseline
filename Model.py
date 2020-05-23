import torch
import torch.nn as nn


class nn_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, X, hidden):
        _, hidden = self.lstm(X, hidden)
        output = self.out(hidden[0])
        return output, hidden

    def initHidden(self):
        return (torch.zeros(1, 1, self.hidden_size).cuda(),
                torch.zeros(1, 1, self.hidden_size).cuda())

    def initHidden_test(self):
        return (torch.zeros(1, 1, self.hidden_size),
                torch.zeros(1, 1, self.hidden_size))