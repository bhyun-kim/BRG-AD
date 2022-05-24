from .base_dataset import BaseADDataset
from torch.utils.data import DataLoader


class TorchDataset(BaseADDataset):
    """
    TorchDataset class
    """

    def __init__(self, root: str):
        super().__init__(root)

    def loaders(self, batch_size, shuffle_train=True, shuffle_test=False, num_workers = 0):
        train_loader = DataLoader(dataset=self.train_set, batch_size=batch_size, shuffle=shuffle_train,
                                  num_workers=num_workers, drop_last=True)
        test_loader = DataLoader(dataset=self.test_set, batch_size=batch_size, shuffle=shuffle_test,
                                 num_workers=num_workers, drop_last=False)
        return train_loader, test_loader