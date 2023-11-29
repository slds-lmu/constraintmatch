import torch
import torchvision
import os
import numpy as np
import logging
import PIL.ImageEnhance
import PIL.ImageDraw


logger = logging.getLogger(__name__)

PARAMETER_MAX = 10

from torchvision import transforms
from sscc.data.images import ImageDataset

class CIFAR20(ImageDataset):
    """Cifar 100 dataset using the 20 superclasses only
    Following the mapping of gansbeke:
    https://github.com/wvangansbeke/Unsupervised-Classification/blob/master/data/cifar.py
    """
    base_folder = 'cifar20'

    def __init__(self,
                 root,
                 part,
                 val_size,
                 num_constraints,
                 num_constraints_val,
                 k,
                 seed=1337,
                 transform=None,
                 label_type: str='constraint',
                 download=True,
                 fold=0,
                 **kwargs):
        super(CIFAR20, self).__init__(root, part=part,
                                      val_size=val_size,
                                      num_constraints=num_constraints,
                                      num_constraints_val=num_constraints_val,
                                      k=k,
                                      seed=seed,
                                      transform=transform,
                                      label_type=label_type,
                                      download=download,
                                      fold=fold)

        self.fold = fold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset_path = os.path.join(self.root, self.base_folder)
        self.label_type = label_type
        self.num_constraints = num_constraints
        self.num_constraints_val = num_constraints_val
        if download:
            self.download()
        self.x, self.y, self.c = self.load_dataset(part=self.part, fold=self.fold)

    def __len__(self):
        """as the __getitem__() must work differently for train and val/test,
        the __len__() must be specified such that the indeces are only sampled from the desired sample space.
        For train: __len__() corresponds to constraints df (C_train.csv)
        For val/test: __len() corresponds to the total num of obs available.
        """
        if self.label_type == 'constraint':
            return len(self.c)
        else:
            return self.x.shape[0]

    def download(self):

        if not self.should_download():
            # same process for train and val dataset, test is "labeled"
            if self.label_type == 'constraint':
                _, y, _ = self.load_dataset(part=self.part, fold=self.fold)
                c_df = self.build_constraints(y=y, seed=self.seed+self.fold)
                c_df.to_csv(f"{self.dataset_path}/fold_{self.fold}/C_{self.part}.csv")
                return
            return

        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                    ])
        trainset = torchvision.datasets.CIFAR100(root=self.dataset_path,
                                                 train=True,
                                                 download=True,
                                                 transform=transform)
        testset = torchvision.datasets.CIFAR100(root=self.dataset_path,
                                                train=False,
                                                download=True,
                                                transform=transform)

        # norm to [0, 1]
        X_train = trainset.data.swapaxes(2, 3).swapaxes(1, 2) / 255.
        Y_train = np.array(trainset.targets)
        X_test = testset.data.swapaxes(2, 3).swapaxes(1, 2) / 255.

        # map cifar100 labels to the 20 superclasses
        Y_train = np.array(trainset.targets)
        for id, y in enumerate(Y_train):
            Y_train[id] = _cifar100_to_cifar20(y)

        Y_test = np.array(testset.targets)
        for id, y in enumerate(Y_test):
            Y_test[id] = _cifar100_to_cifar20(y)

        self._split_and_save(x_test=X_test,
                             x_train=X_train,
                             y_test=Y_test,
                             y_train=Y_train)

        os.system('rm {}/cifar-100-python.tar.gz; rm -rf {}/cifar-100-python'.format(self.dataset_path, self.dataset_path))


def _cifar100_to_cifar20(target):
    """translate cf100 to 20 superclasses

    Args:
        target (int): target

    Returns:
        int: superclass label
    """
    _dict = \
        {0: 4,
        1: 1,
        2: 14,
        3: 8,
        4: 0,
        5: 6,
        6: 7,
        7: 7,
        8: 18,
        9: 3,
        10: 3,
        11: 14,
        12: 9,
        13: 18,
        14: 7,
        15: 11,
        16: 3,
        17: 9,
        18: 7,
        19: 11,
        20: 6,
        21: 11,
        22: 5,
        23: 10,
        24: 7,
        25: 6,
        26: 13,
        27: 15,
        28: 3,
        29: 15,
        30: 0,
        31: 11,
        32: 1,
        33: 10,
        34: 12,
        35: 14,
        36: 16,
        37: 9,
        38: 11,
        39: 5,
        40: 5,
        41: 19,
        42: 8,
        43: 8,
        44: 15,
        45: 13,
        46: 14,
        47: 17,
        48: 18,
        49: 10,
        50: 16,
        51: 4,
        52: 17,
        53: 4,
        54: 2,
        55: 0,
        56: 17,
        57: 4,
        58: 18,
        59: 17,
        60: 10,
        61: 3,
        62: 2,
        63: 12,
        64: 12,
        65: 16,
        66: 12,
        67: 1,
        68: 9,
        69: 19,
        70: 2,
        71: 10,
        72: 0,
        73: 1,
        74: 16,
        75: 12,
        76: 9,
        77: 13,
        78: 15,
        79: 13,
        80: 16,
        81: 19,
        82: 2,
        83: 4,
        84: 6,
        85: 19,
        86: 5,
        87: 5,
        88: 8,
        89: 19,
        90: 18,
        91: 1,
        92: 2,
        93: 15,
        94: 6,
        95: 0,
        96: 17,
        97: 8,
        98: 14,
        99: 13}

    return _dict[target]