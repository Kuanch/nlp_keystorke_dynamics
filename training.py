import math

import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from utils.lstm_model import LSTM
# from utils.lstm_model import LinearModel
from utils.keystroke_dataset import KeyStrokeDataset


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True


def get_data_loader():
    data = pd.read_csv('DSL-StrongPasswordData.csv')

    time_stamp = data.iloc[:, 3:]
    np_time_stamp = np.array(time_stamp, dtype=np.float32)
    # Reshape dataframe before creating dataset, for lstm input
    lstm_input = np.reshape(np_time_stamp, (len(np_time_stamp), 1, 31))

    typist = np.zeros(data.shape[0])
    typist_list = list(set(data.subject))
    # FIXME
    # !!!!!!! Be careful !!!!!!!!
    # since set() is unordered, sort() is a easy work-around
    # be careful about the x and label mis-mapping
    typist_list.sort()
    for i, s in data.subject.iteritems():
        typist[i] = typist_list.index(s)

    training = KeyStrokeDataset(lstm_input, typist)
    sampler = torch.utils.data.RandomSampler(training)
    batch_sampler = torch.utils.data.BatchSampler(sampler, batch_size=8, drop_last=True)
    data_loader = torch.utils.data.DataLoader(
        training, batch_sampler=batch_sampler, num_workers=8,
        collate_fn=None, pin_memory=True)

    return data_loader


def get_model():
    def init(m):
        if isinstance(m, nn.Linear):
            m.weight.data.fill_(0.01)

    model = LSTM(31, 128, 64, num_layers=5, num_classes=51)
    # Initialize all nn.Linear weight to be 0.01
    # model = LinearModel(31, num_classes=16)
    # model.apply(init)

    return model


def get_lr_schedular(optimizer, warm_up_epochs, T_max):
    def warm_up_with_cosine_lr(epoch):
        if epoch <= warm_up_epochs:
            return epoch / warm_up_epochs + 0.001
        else:
            return 0.5 * (math.cos((epoch - warm_up_epochs) / (T_max) * math.pi) + 1)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_cosine_lr)


def train():
    data_loader = get_data_loader()
    model = get_model().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = get_lr_schedular(optimizer, 100, 1000)

    # writer = SummaryWriter('runs/linear_sgd_residual_initw0.01_warmup_cls16_normalize')
    writer = SummaryWriter('runs/lstm_sgd_warmup_cls51_normalize')
    # writer = SummaryWriter('runs/test')

    epoch = 10000
    steps_per_epoch = len(data_loader)
    for e in range(epoch):
        model.train()
        for d in data_loader:
            x = d['x'].to(device)
            norm = x.norm(p=1, dim=0)
            x_norm = x.div(norm.expand_as(x))
            labels = d['label'].to(device)
            pred = model(x_norm)
            loss = loss_fn(pred, labels.long())
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

        if e % 50 == 0 and e > 0:
            writer.add_scalar("Loss/train", loss, e * steps_per_epoch)
            writer.add_scalar("Loss/lr", scheduler.get_last_lr()[0], e * steps_per_epoch)
            '''
            # For LinearModel
            writer.add_histogram("parameters_{}".format('residual'), getattr(model, 'residual_weight')['residual_weight'].data, (e + 1) * steps_per_epoch)
            writer.add_histogram("parameters_{}".format('h1_w'), getattr(model, 'hidden1').weight.data, (e + 1) * steps_per_epoch)
            writer.add_histogram("parameters_{}".format('h4_w'), getattr(model, 'hidden4').weight.data, (e + 1) * steps_per_epoch)
            '''
            # For LSTM
            writer.add_histogram("parameters_{}".format('input_w'), model.input_weighting.weight.data, (e + 1) * steps_per_epoch)
            writer.add_histogram("parameters_{}".format('hh_w0'), model.lstm.weight_hh_l0.data, (e + 1) * steps_per_epoch)
            writer.add_histogram("parameters_{}".format('hh_w4'), model.lstm.weight_hh_l4.data, (e + 1) * steps_per_epoch)
            writer.add_histogram("parameters_{}".format('ih_w0'), model.lstm.weight_ih_l0.data, (e + 1) * steps_per_epoch)
            writer.add_histogram("parameters_{}".format('ih_w4'), model.lstm.weight_ih_l4.data, (e + 1) * steps_per_epoch)
            writer.add_histogram("parameters_{}".format('hidden'), model.hidden.weight.data, (e + 1) * steps_per_epoch)

        scheduler.step()

        if (e % 1000 == 0 and e > 0) or e == epoch - 1:
            torch.save(model.state_dict(), 'model/lstm_51-{}_warmup.pt'.format(e))

            acc = 0
            model.eval()
            with torch.no_grad():
                for d in data_loader:
                    x = d['x'].to(device)
                    norm = x.norm(p=1, dim=0)
                    x_norm = x.div(norm.expand_as(x))
                    pred = model(x_norm)

                    acc += torch.true_divide(torch.sum(torch.argmax(pred, 1) == d['label'].to(device)), 8)

                writer.add_scalar("Loss/accuracy", torch.true_divide(acc, steps_per_epoch), e * steps_per_epoch)

    writer.flush()
    writer.close()


if __name__ == '__main__':
    train()
