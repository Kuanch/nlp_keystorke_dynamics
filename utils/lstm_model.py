import torch
from torch import nn


class LSTM(nn.Module):
    def __init__(self, num_channels, hidden_dim, num_classes, num_layers=5, target_size=512):
        super(LSTM, self).__init__()
        self.num_channels = num_channels
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(num_channels, hidden_dim, num_layers=num_layers, batch_first=False)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2target = nn.Linear(hidden_dim, num_classes)

        self.softmax = torch.nn.Softmax(dim=1)

        # self.classifier = nn.Linear(target_size, num_classes)

    def forward(self, time_stamp):
        """
        Args:
            time_stamp: (num_feature, batch_size, num_channels)
        Output:
            pred_class: (batch_size, num_classes)
        """
        # h.shape = c.shape = (num_layers, num_feature, hidden_dim)
        #                                     num_feature, batch_size, dim_vector
        o, _ = self.lstm(time_stamp.view(-1, len(time_stamp), self.num_channels))
        state = self.hidden2target(o.view(len(time_stamp), len(time_stamp[0]), -1))
        metric = torch.sum(state, dim=1)

        return self.softmax(metric)


class LinearModel(nn.Module):
    def __init__(self, num_channels, hidden_dim, num_classes, num_layers=5, target_size=512):
        super(LinearModel, self).__init__()
        self.num_channels = num_channels
        self.hidden_dim = hidden_dim
        self.lstm = nn.Linear(31, hidden_dim * 2)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2target = nn.Linear(hidden_dim * 2, num_classes)

        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, time_stamp):
        """
        Args:
            time_stamp: (num_feature, batch_size, num_channels)
        Output:
            pred_class: (batch_size, num_classes)
        """
        o = self.lstm(time_stamp.view(len(time_stamp), -1))

        state_metric = self.hidden2target(o)

        return self.softmax(state_metric)
