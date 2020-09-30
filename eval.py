import torch
import torch.autograd.profiler as profiler

from training import get_data_loader
from utils.lstm_model import LSTM


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_model():
    model = LSTM(31, 128, 64, num_classes=51)
    model.load_state_dict(torch.load('model/lstm_51-500_13_warmup100_1000_val0001_lr001_b8.pt'))

    return model


def eval():
    data_loader, _ = get_data_loader()
    lstm = load_model().to(device)
    acc = 0
    profiler_x = None

    lstm.eval()
    with torch.no_grad():
        for d in data_loader:
            x = d['x'].to(device)
            label = d['label'].to(device)
            if profiler_x is None:
                profiler_x = x
            pred = lstm(x).to(device)

            _acc = torch.true_divide((torch.argmax(pred, 1) == label).double().sum().item(), 8)
            acc += _acc.item()

        print(torch.true_divide(acc, len(data_loader)), len(data_loader))

    with profiler.profile(record_shapes=True, profile_memory=True, use_cuda=True) as prof:
        with profiler.record_function("model_inference"):
            lstm(profiler_x)
    print(prof.key_averages().table(row_limit=100))
    prof.export_chrome_trace("trace.json")


if __name__ == '__main__':
    eval()
