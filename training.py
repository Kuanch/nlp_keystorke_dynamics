import math

import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_score, recall_score

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

    validation_split = .005
    shuffle_dataset = True
    random_seed = 42

    dataset = KeyStrokeDataset(lstm_input, typist)
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    valid_sampler = torch.utils.data.SubsetRandomSampler(val_indices)

    batch_sampler = torch.utils.data.BatchSampler(train_sampler, batch_size=8, drop_last=True)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_sampler=batch_sampler, num_workers=8,
        collate_fn=None, pin_memory=True)

    v_batch_sampler = torch.utils.data.BatchSampler(valid_sampler, batch_size=4, drop_last=False)
    val_data_loader = torch.utils.data.DataLoader(
        dataset, batch_sampler=v_batch_sampler, num_workers=8,
        collate_fn=None, pin_memory=True)

    return data_loader, val_data_loader


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
            return 0.5 * (math.cos((epoch - warm_up_epochs) / (T_max) * math.pi) + 1) + 0.001

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_cosine_lr)


def train_board(model, data_loader, e, writer):
    acc = 0
    precision = 0
    recall = 0
    steps_per_epoch = len(data_loader)
    for d in data_loader:
        x = d['x'].to(device)
        pred = torch.argmax(model(x), 1).cpu()
        label = d['label'].cpu()

        acc += torch.true_divide(torch.sum(pred == label), 8)

        precision += precision_score(label, pred, average='macro', zero_division=0)
        recall += recall_score(label, pred, average='macro', zero_division=0)

    writer.add_scalar("Accuracy/train_accuracy", torch.true_divide(acc, steps_per_epoch), e * steps_per_epoch)
    writer.add_scalar("Accuracy/train_precision", torch.true_divide(precision, steps_per_epoch), e * steps_per_epoch)
    writer.add_scalar("Accuracy/train_recall", torch.true_divide(recall, steps_per_epoch), e * steps_per_epoch)


def eval_board(model, data_loader, e, writer):
    acc = 0
    precision = 0
    recall = 0
    steps_per_epoch = len(data_loader)
    for d in data_loader:
        x = d['x'].to(device)
        pred = torch.argmax(model(x), 1).cpu()
        label = d['label'].cpu()

        acc += torch.true_divide(torch.sum(pred == label), 4)

        precision += precision_score(label, pred, average='macro', zero_division=0)
        recall += recall_score(label, pred, average='macro', zero_division=0)

    writer.add_scalar("Accuracy/accuracy", torch.true_divide(acc, steps_per_epoch), e * steps_per_epoch)
    writer.add_scalar("Accuracy/precision", torch.true_divide(precision, steps_per_epoch), e * steps_per_epoch)
    writer.add_scalar("Accuracy/recall", torch.true_divide(recall, steps_per_epoch), e * steps_per_epoch)


def train():
    data_loader, val_data_loader = get_data_loader()
    model = get_model().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = get_lr_schedular(optimizer, 100, 1000)

    # writer = SummaryWriter('runs/linear_sgd_residual_initw0.01_warmup_cls16_normalize')
    writer = SummaryWriter('runs/lstm_sgd_warmup100_1000_cls51_0.01_8_val0001_13')
    # writer = SummaryWriter('runs/test')

    epoch = 5000
    steps_per_epoch = len(data_loader)
    for e in range(epoch):
        model.train()
        for d in data_loader:
            x = d['x'].to(device)
            labels = d['label'].to(device)
            pred = model(x)
            loss = loss_fn(pred, labels.long())
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

        if e % 50 == 0 and e > 0:
            writer.add_scalar("Loss/train", loss, e * steps_per_epoch)
            writer.add_scalar("Loss/lr", scheduler.get_last_lr()[0], e * steps_per_epoch)
            # For LSTM
            writer.add_histogram("parameters_{}".format('input_w'), model.input_weighting.weight.data, (e + 1) * steps_per_epoch)
            writer.add_histogram("parameters_{}".format('hh_w0'), model.lstm.weight_hh_l0.data, (e + 1) * steps_per_epoch)
            writer.add_histogram("parameters_{}".format('hh_w4'), model.lstm.weight_hh_l4.data, (e + 1) * steps_per_epoch)
            writer.add_histogram("parameters_{}".format('ih_w0'), model.lstm.weight_ih_l0.data, (e + 1) * steps_per_epoch)
            writer.add_histogram("parameters_{}".format('ih_w4'), model.lstm.weight_ih_l4.data, (e + 1) * steps_per_epoch)
            writer.add_histogram("parameters_{}".format('hidden'), model.hidden.weight.data, (e + 1) * steps_per_epoch)

        scheduler.step()

        if (e % 500 == 0 and e > 0) or e == epoch - 1:
            torch.save(model.state_dict(), 'model/lstm_51-{}_13_warmup100_1000_val0001_lr001_b8.pt'.format(e))

            model.eval()
            with torch.no_grad():
                train_board(model, data_loader, e, writer)
                eval_board(model, val_data_loader, e, writer)

    writer.flush()
    writer.close()


if __name__ == '__main__':
    train()
