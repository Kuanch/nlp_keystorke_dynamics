import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from utils.lstm_model import LinearModel
from utils.keystroke_dataset import KeyStrokeDataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True


def get_data_loader():
    data = pd.read_csv('DSL-StrongPasswordData.csv')

    time_stamp = data.iloc[:, 3:]
    np_time_stamp = np.array(time_stamp, dtype=np.float32)

    typist = np.zeros(data.shape[0])
    typist_list = list(set(data.subject))
    for i, s in data.subject.iteritems():
        typist[i] = typist_list.index(s)

    training = KeyStrokeDataset(np_time_stamp, typist)
    sampler = torch.utils.data.RandomSampler(training)
    batch_sampler = torch.utils.data.BatchSampler(sampler, batch_size=256, drop_last=True)
    data_loader = torch.utils.data.DataLoader(
        training, batch_sampler=batch_sampler, num_workers=4,
        collate_fn=None, pin_memory=True)

    return data_loader


def get_model():
    # lstm = LSTM(1, 50, num_layers=5, num_classes=51).to(device)
    model = LinearModel(1, 50, num_layers=3, num_classes=51).to(device)

    return model


def train():
    data_loader = get_data_loader()
    model = get_model()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0.0001)

    writer = SummaryWriter()

    _step = 0
    epoch = 50000
    steps_per_epoch = len(data_loader)
    for e in range(epoch):
        for d in data_loader:
            x = d['x'].to(device)
            labels = d['label'].to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, labels.long())

            loss.backward()
            optimizer.step()
            _step += 1

            if _step % 50 == 0:
                writer.add_scalar("Loss/train", loss, e * steps_per_epoch + _step)
                writer.add_scalar("Loss/lr", scheduler.get_last_lr()[0], e * steps_per_epoch + _step)
        for p in model.parameters():
            writer.add_histogram("Parameters", p.data, (e + 1) * steps_per_epoch)
        _step = 0

        scheduler.step()

        if e % 1000 == 0 and e > 0:
            torch.save(model.state_dict(), 'model/linear-{}.pt'.format(e))

    writer.flush()
    writer.close()


if __name__ == '__main__':
    train()
