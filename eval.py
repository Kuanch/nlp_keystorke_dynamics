import torch

from training import get_data_loader
from utils.lstm_model import LSTM, LinearModel


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_model():
    model = LSTM(31, 128, 64, num_classes=51)
    model.load_state_dict(torch.load('model/lstm_51-1000_warmup.pt'))
    model.eval()

    return model


def eval():
    data_loader = get_data_loader()
    lstm = load_model().to(device)
    acc = 0

    with torch.no_grad():
        for d in data_loader:
            x = d['x'].to(device)
            norm = x.norm(p=1, dim=0)
            x_norm = x.div(norm.expand_as(x))
            pred = lstm(x_norm)
            # print(torch.argmax(pred, 1), d['label'])

            _acc = torch.true_divide(torch.sum(torch.argmax(pred, 1) == d['label'].to(device)), 8)
            # print(_acc)
            acc += _acc.item()

        print(acc)
        print(torch.true_divide(acc, len(data_loader)), len(data_loader))


if __name__ == '__main__':
    eval()
