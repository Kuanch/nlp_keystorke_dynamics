import torch
from torch import nn


class LSTM(nn.Module):
    def __init__(self, num_feature, up_dim, hidden_dim, num_classes, num_layers=5, target_size=256):
        super(LSTM, self).__init__()
        self.num_feature = num_feature
        self.hidden_dim = hidden_dim
        self.input_weighting = nn.Linear(num_feature, up_dim, bias=False)
        self.lstm = nn.LSTM(up_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.hidden = nn.Linear(hidden_dim, num_classes)

    def forward(self, time_stamp):
        """
        Args:
            time_stamp: (batch_size, num_feature)
        Output:
            pred_class: (batch_size, num_classes)
        """
        # h.shape = c.shape = (num_layers, num_feature, hidden_dim)
        #                                     num_feature, batch_size, dim_vector
        input_seq = time_stamp
        weighted_input = self.input_weighting(input_seq)
        o, _ = self.lstm(weighted_input)
        logits = torch.sum(self.hidden(o), dim=1)

        return logits


class LinearModel(nn.Module):
    def __init__(self, num_feature, num_classes):
        super(LinearModel, self).__init__()

        hidden_dim = 256
        self.weighted = nn.Linear(num_feature, hidden_dim)
        self.hidden1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.hidden2_1 = nn.Linear(hidden_dim * 2, hidden_dim * 4)
        self.hidden2_2 = nn.Linear(hidden_dim * 2, hidden_dim * 4)
        self.hidden2_3 = nn.Linear(hidden_dim * 2, hidden_dim * 4)
        self.hidden3 = nn.Linear(hidden_dim * 4, hidden_dim * 2)
        self.residual_weight = nn.ParameterDict({'residual_weight': nn.Parameter(torch.zeros(1))})
        self.hidden4 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.target = nn.Linear(hidden_dim, num_classes)

        # self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, time_stamp):
        """
        Args:
            time_stamp: (num_feature, batch_size, num_feature)
        Output:
            pred_class: (batch_size, num_classes)
        """
        input_t = time_stamp
        input_t = self.weighted(input_t)
        residual = input_t = self.hidden1(input_t)
        input_t = self.hidden2_1(input_t) + self.hidden2_2(input_t) + self.hidden2_3(input_t)
        input_t = self.hidden3(input_t) + residual * self.residual_weight['residual_weight']
        input_t = self.hidden4(input_t)
        input_t = self.target(input_t)

        return input_t
