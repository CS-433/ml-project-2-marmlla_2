import torch
import torch.nn as nn


class Dense(torch.nn.Module):

    """Dense neural network model.
    _init_:
        num_layers: scalar, number of hidden layers.
        layer_size: scalar, number of nodes per layer.
        input_size: scalar, number of expected features in the input x.
        output_size: scalar, size of each output sample.
        dropout: scalar within [0,1], dropout rate.
    """

    def __init__(
        self, num_layers=1, layer_size=16, input_size=10, output_size=1, dropout=0.25
    ):
        super(Dense, self).__init__()

        # Define activation function
        activation = nn.ReLU

        # Define layers according to input parameters
        modules = []

        modules.append(nn.Linear(input_size, layer_size))  # Input layer
        modules.append(nn.BatchNorm1d(layer_size))  # Batch normalization
        modules.append(nn.Dropout(dropout))  # Dropout
        modules.append(activation())  # Activation function

        for i in range(num_layers):
            if i != num_layers - 1:
                modules.append(nn.Linear(layer_size, layer_size))
                modules.append(nn.BatchNorm1d(layer_size))
                modules.append(nn.Dropout(dropout))
                modules.append(activation())
            else:
                modules.append(nn.Linear(layer_size, output_size))

        self.net = nn.Sequential(*modules)

    def forward(self, Z):
        for mod in self.net:
            Z = mod(Z)
        return Z


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
        num_layers=4,
        input_size=1,
        out_features_lin=128,
        out_features_end=1,
        dropout=0.2,
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
        self.fc0_bn = nn.BatchNorm1d(self.hidden_size)
        # self.dropout0 = nn.Dropout(p=0.2)
        self.fc_0 = nn.Linear(self.hidden_size, out_features_lin)

        # fully connected 1
        self.fc1_bn = nn.BatchNorm1d(out_features_lin)
        # self.dropout1 = nn.Dropout(p=0.2)
        self.fc_1 = nn.Linear(out_features_lin, out_features_end)

        # fully connected 2 (last)
        # self.fc2_bn = nn.BatchNorm1d(out_features_lin)
        # self.dropout2 = nn.Dropout(p=0.2)
        # self.fc_2 = nn.Linear(out_features_lin, out_features_end)

        self.relu = nn.ReLU()

        # nn.init.xavier_uniform_(self.fc_0.weight)
        # nn.init.zeros_(self.fc_0.bias)
        # nn.init.xavier_uniform_(self.fc_1.weight)
        # nn.init.zeros_(self.fc_1.bias)
        # nn.init.xavier_uniform_(self.fc_2.weight)
        # nn.init.zeros_(self.fc_2.bias)

    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to('cuda')  # hidden state
        # Propagate input through GRU
        output, hn = self.gru(x, h_0)  # GRU with input, hidden, and internal state
        


        hn = hn[-1].view(
            -1, self.hidden_size
        )  # reshaping the data for Dense layer next

        #hn = torch.cat((hn, x[:, :, 0]), dim=1)

        out = self.fc_0(self.relu(self.fc0_bn(hn)))
        out = self.fc_1(self.relu(self.fc1_bn(out)))  # self.dropout1(

        # out = self.fc_2(self.dropout2(self.relu(self.fc2_bn(out))))

        return out


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.fcd = nn.Linear(5, 16)
        self.bn = nn.BatchNorm1d(16)
        self.fc1 = nn.Linear(16, 8)
        self.dropout = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(8, 1)

        self.relu = nn.ReLU()

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.xavier_uniform_(self.fcd.weight)
        nn.init.zeros_(self.fcd.bias)

    def forward(self, x):
        x = self.relu(self.fcd(x))
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        out = self.fc2(x)
        return out


class ExchangeRateNet(nn.Module):
    def __init__(self):
        super(ExchangeRateNet, self).__init__()

        # regression
        self.gru_ex_reg = GRU_base()
        self.gru_infl_CH_reg = GRU_base()
        self.gru_infl_US_reg = GRU_base()
        self.gru_interest_CH_reg = GRU_base()
        self.gru_interest_US_reg = GRU_base()
        self.decoder_reg = Decoder()

        # trend
        self.gru_ex_trend = GRU_base()
        self.gru_infl_CH_trend = GRU_base()
        self.gru_infl_US_trend = GRU_base()
        self.gru_interest_CH_trend = GRU_base()
        self.gru_interest_US_trend = GRU_base()
        self.decoder_trend = Decoder()

    def forward(self, inputs):
        enc_out1_reg = self.gru_ex_reg(inputs[:, :, 0].unsqueeze(-1))
        enc_out2_reg = self.gru_ex_reg(inputs[:, :, 1].unsqueeze(-1))
        enc_out3_reg = self.gru_ex_reg(inputs[:, :, 2].unsqueeze(-1))
        enc_out4_reg = self.gru_ex_reg(inputs[:, :, 3].unsqueeze(-1))
        enc_out5_reg = self.gru_ex_reg(inputs[:, :, 4].unsqueeze(-1))
        """
        enc_out1_trend = self.gru_ex_reg(inputs[:, 0])
        enc_out2_trend = self.gru_ex_reg(inputs[:, 1, :])
        enc_out3_trend = self.gru_ex_reg(inputs[:, 2, :])
        enc_out4_trend = self.gru_ex_reg(inputs[:, 3, :])
        enc_out5_trend = self.gru_ex_reg(inputs[:, 4, :])
        """
        out = torch.cat(
            (
                enc_out1_reg,
                enc_out2_reg,
                enc_out3_reg,
                enc_out4_reg,
                enc_out5_reg,
                # enc_out1_trend,
                # enc_out2_trend,
                # enc_out3_trend,
                # enc_out4_trend,
                # enc_out5_trend,
            ),
            dim=1,
        )
        out = self.decoder_reg(out)
        # out2 = self.decoder_trend(out)

        return [
            enc_out1_reg,
            enc_out2_reg,
            enc_out3_reg,
            enc_out4_reg,
            enc_out5_reg,
            # enc_out1_trend,
            # enc_out2_trend,
            # enc_out3_trend,
            # enc_out4_trend,
            # enc_out5_trend,
            out
            # out_trend,
        ]
