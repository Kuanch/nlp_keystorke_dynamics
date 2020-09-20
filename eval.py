import torch

from training import get_data_loader
from utils.lstm_model import LSTM, LinearModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    # lstm = LSTM(1, 50, num_layers=5, num_classes=51).to(device)
    model = LinearModel(1, 50, num_layers=3, num_classes=51).to(device)
    model.load_state_dict(torch.load('model/linear-4000.pt'))
    model.eval()

    return model


def eval():
    data_loader = get_data_loader()
    lstm = load_model()
    acc = 0

    with torch.no_grad():
        for d in data_loader:
            x = d['x'].to(device)
            pred = lstm(x)

            acc += torch.sum(torch.true_divide(torch.argmax(pred, 1) == d['label'].to(device), 256))

        print(torch.true_divide(acc, len(data_loader)), len(data_loader))


if __name__ == '__main__':
    eval()
