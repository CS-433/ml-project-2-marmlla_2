import torch
import torch.nn as nn


class GRU_base(nn.Module):
    """GRU base model.
    _init_:
        hidden_size: scalar, number of features in the hidden state.
        num_layers: scalar, number of recurrent layers.
        input_size: scalar, number of expected features in the input x.
        out_features_lin: scalar, size of each output sample.
    """

    def __init__(
        self,
        hidden_size=64,
        num_layers=1,
        input_size=1,
        out_features_lin=128,
        out_features_end=1,
        dropout=0.25,
    ):
        super(GRU_base, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.out_features_lin = out_features_lin
        self.out_features_end = out_features_end
        self.dropout = dropout

        # GRU
        self.gru = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout,
        )

        # fully connected 0
        self.fc_0 = nn.Linear(self.hidden_size, out_features_lin)
        self.fc0_bn = nn.BatchNorm1d(self.hidden_size)

        # fully connected 1 (last)
        self.fc_1 = nn.Linear(out_features_lin, out_features_end)

        self.relu = nn.ReLU()

    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)  # hidden state

        # Propagate input through GRU
        output, hn = self.gru(x, h_0)  # GRU with input, hidden, and internal state

        hn = hn[-1].view(
            -1, self.hidden_size
        )  # reshaping the data for Dense layer next

        out = self.fc_0(self.relu(self.fc0_bn(hn)))
        out = self.fc_1(self.relu(out))

        return out
