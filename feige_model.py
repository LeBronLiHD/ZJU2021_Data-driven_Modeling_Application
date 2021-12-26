# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

class Tem_Pre_Lstm(nn.Module):
    def __init__(self, hidden_dim=64, num_layers=2, output_size=1, input_size=6, dropout=0.5):
        super(Tem_Pre_Lstm, self).__init__()
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_dim, num_layers, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(hidden_dim * num_layers, num_layers * hidden_dim * 2)
        self.linear2 = nn.Linear(num_layers * hidden_dim * 2, output_size)

    def forward(self,x):
        out, (h, c) = self.lstm(x)

        if self.num_layers == 2:
            out = torch.cat((c[0], c[1]), 1)
        if self.num_layers == 3:
            out = torch.cat((c[0], c[1], c[2]), 1)
        if self.num_layers == 4:
            out = torch.cat((c[0], c[1], c[2], c[3]), 1)

        out = self.linear1(out)
        out = torch.relu(out)
        out = self.linear2(out)

        return out
