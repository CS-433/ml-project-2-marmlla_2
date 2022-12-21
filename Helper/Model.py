import torch
import torch.nn as nn
import torch.nn.functional as F


class Auto_Encoder(nn.Module):
    """Auto encoder model.
    _init_:
        hidden_size: scalar, number of features in the hidden state.
        num_layers: scalar, number of recurrent layers.
        input_size: scalar, number of expected features in the input x.
        out_features_lin: scalar, size of each output sample.
    """

    def __init__(self, input_size=1, nb_channel_conv=3):
        super(Auto_Encoder, self).__init__()

        self.input_size = input_size
        self.nb_channel_conv = nb_channel_conv

        self.relu = nn.LeakyReLU()

        self.conv1 = nn.Conv1d(self.input_size, self.nb_channel_conv, kernel_size=3)
        self.conv2 = nn.Conv1d(
            self.nb_channel_conv, self.nb_channel_conv, kernel_size=3
        )
        self.conv3 = nn.Conv1d(
            self.nb_channel_conv, self.nb_channel_conv, kernel_size=2
        )

        self.conv1_t = nn.ConvTranspose1d(
            self.nb_channel_conv, self.nb_channel_conv, kernel_size=2
        )
        self.conv2_t = nn.ConvTranspose1d(
            self.nb_channel_conv, self.nb_channel_conv, kernel_size=3
        )
        self.conv3_t = nn.ConvTranspose1d(
            self.nb_channel_conv, self.input_size, kernel_size=3
        )

    def forward(self, x, return_latent=False):

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        latent = x  # N, 3, 1
        x = self.relu(self.conv1_t(x))
        x = self.relu(self.conv2_t(x))
        dec_out = self.relu(self.conv3_t(x))
        if return_latent:
            return dec_out, latent
        else:
            return dec_out


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

        # initialize the weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, Z):
        for mod in self.net:
            Z = mod(Z)
        return Z


class LSTM_base(nn.Module):
    """LSTM base model.
    _init_:
        hidden_size: scalar, number of features in the hidden state.
        num_layers: scalar, number of recurrent layers.
        input_size: scalar, number of expected features in the input x.
        out_features_lin: scalar, size of each output sample.
    """

    def __init__(
        self,
        hidden_size=32,
        num_layers=1,
        input_size=1,
        out_features_lin=32,
        out_features_end=1,
        dropout=0.05,
        device="cpu",
    ):
        super(LSTM_base, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.out_features_lin = out_features_lin
        self.out_features_end = out_features_end
        self.dropout = dropout
        self.device = device

        # LSTM
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout,
        )

        # fully connected 0
        self.fc0_bn = nn.BatchNorm1d(self.hidden_size)
        self.fc_0 = nn.Linear(self.hidden_size, out_features_lin)

        # fully connected 1
        self.fc1_bn = nn.BatchNorm1d(out_features_lin)
        self.fc_1 = nn.Linear(out_features_lin, out_features_end)

        self.relu = nn.LeakyReLU()  # nn.ReLU()

        # initialize the weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):

        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(
            self.device
        )  # hidden state
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(
            self.device
        )  # hidden state

        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0))

        hn = hn[-1].view(
            -1, self.hidden_size
        )  # reshaping the data for Dense layer next

        out = self.fc_0(self.relu(self.fc0_bn(hn)))
        out = self.fc_1(self.relu(self.fc1_bn(out)))

        return out


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
        hidden_size=32,
        num_layers=1,
        input_size=1,
        out_features_lin=32,
        out_features_end=1,
        dropout=0.05,
        device="cpu",
    ):
        super(GRU_base, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.out_features_lin = out_features_lin
        self.out_features_end = out_features_end
        self.dropout = dropout
        self.device = device

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
        self.fc_0 = nn.Linear(self.hidden_size, out_features_lin)

        # fully connected 1
        self.fc1_bn = nn.BatchNorm1d(out_features_lin)
        self.fc_1 = nn.Linear(out_features_lin, out_features_end)

        self.relu = nn.LeakyReLU()  # nn.ReLU()

        # initialize the weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(
            self.device
        )  # hidden state

        # Propagate input through GRU
        output, hn = self.gru(x, h_0)

        hn = hn[-1].view(
            -1, self.hidden_size
        )  # reshaping the data for Dense layer next

        out = self.fc_0(self.relu(self.fc0_bn(hn)))
        out = self.fc_1(self.relu(self.fc1_bn(out)))

        return out


class GRU_trend(nn.Module):
    """GRU base model.
    _init_:
        hidden_size: scalar, number of features in the hidden state.
        num_layers: scalar, number of recurrent layers.
        input_size: scalar, number of expected features in the input x.
        out_features_lin: scalar, size of each output sample.
    """

    def __init__(
        self,
        hidden_size=32,
        num_layers=1,
        input_size=1,
        out_features_lin=32,
        out_features_end=1,
        dropout=0.05,
        device="cpu",
    ):
        super(GRU_trend, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.out_features_lin = out_features_lin
        self.out_features_end = out_features_end
        self.dropout = dropout
        self.device = device
        self.num_trend_features = 1

        # GRU
        self.gru = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout,
        )

        # fully connected 0
        self.fc0_bn = nn.BatchNorm1d(self.hidden_size + self.num_trend_features)
        # self.dropout0 = nn.Dropout(p=0.2)
        self.fc_0 = nn.Linear(
            self.hidden_size + self.num_trend_features, out_features_lin
        )

        # fully connected 1
        self.fc1_bn = nn.BatchNorm1d(out_features_lin)
        # self.dropout1 = nn.Dropout(p=0.2)
        self.fc_1 = nn.Linear(out_features_lin, out_features_end)

        # fully connected 2 (last)
        # self.fc2_bn = nn.BatchNorm1d(out_features_lin)
        # self.dropout2 = nn.Dropout(p=0.2)
        # self.fc_2 = nn.Linear(out_features_lin, out_features_end)

        self.relu = nn.LeakyReLU()  # nn.ReLU()

        # nn.init.xavier_uniform_(self.fc_0.weight)
        # nn.init.zeros_(self.fc_0.bias)
        # nn.init.xavier_uniform_(self.fc_1.weight)
        # nn.init.zeros_(self.fc_1.bias)
        # nn.init.xavier_uniform_(self.fc_2.weight)
        # nn.init.zeros_(self.fc_2.bias)

    def forward(self, x, x_trend):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(
            self.device
        )  # hidden state
        # Propagate input through GRU
        output, hn = self.gru(x, h_0)  # GRU with input, hidden, and internal state

        hn = hn[-1].view(
            -1, self.hidden_size
        )  # reshaping the data for Dense layer next

        out = torch.cat((hn, x_trend), dim=1)

        out = self.fc_0(self.relu(self.fc0_bn(out)))
        out = self.fc_1(self.relu(self.fc1_bn(out)))  # self.dropout1(

        # out = self.fc_2(self.dropout2(self.relu(self.fc2_bn(out))))

        return out


class ResUNet(nn.Module):
    def __init__(self):
        super(ResUNet, self).__init__()

        self.conv = nn.Conv1d(36, 32, 3, padding=1)

        self.block1 = nn.Sequential(
            ResBlock(32, 3),
            nn.Conv1d(32, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )

        self.block2 = ResBlock(64, 3)

        self.block3 = nn.Sequential(
            ResBlock(64, 3),
            nn.ConvTranspose1d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )

        self.block4 = nn.Sequential(
            nn.Conv1d(64, 32, 3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            ResBlock(32, 3),
            nn.Conv1d(32, 36, 3, padding=1),
            nn.Sigmoid(),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv(x)
        pool1 = self.block1(x)
        latent = self.block2(pool1)
        up1 = self.block3(latent)
        concat = torch.cat((up1, x), dim=1)
        return self.block4(concat)


class ResBlock(nn.Module):
    def __init__(self, nb_channels, kernel_size):
        super().__init__()
        self.conv1 = nn.Conv1d(
            nb_channels, nb_channels, kernel_size, padding=(kernel_size - 1) // 2
        )
        self.bn1 = nn.BatchNorm1d(nb_channels)
        self.conv2 = nn.Conv1d(
            nb_channels, nb_channels, kernel_size, padding=(kernel_size - 1) // 2
        )
        self.bn2 = nn.BatchNorm1d(nb_channels)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()

    def forward(self, x):
        y = self.bn1(self.conv1(x))
        y = F.relu(y)
        y = self.bn2(self.conv2(y))
        y += x
        y = F.relu(y)
        return y
