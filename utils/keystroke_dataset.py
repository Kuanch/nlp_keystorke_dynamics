import torch


class KeyStrokeDataset(torch.utils.data.Dataset):
    """
    A Map-style datasets.
    """
    def __init__(self, time_stamp, typist, transform=None):
        self.x = time_stamp
        self.label = typist
        self.transform = transform

    def __len__(self):
        # for sampler
        return len(self.label)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        _x = self.x[idx, :]
        _label = self.label[idx]
        sample = {'x': _x, 'label': _label}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __iter__(self):
        """
        For a Iterable-style datasets.
        """
        raise NotImplementedError
