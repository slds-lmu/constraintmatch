from imghdr import tests
import torch
import torchvision
import os
import numpy as np
import pdb
import logging
import PIL.ImageOps
import PIL.ImageEnhance
import PIL.ImageDraw
from torchvision.transforms.transforms import ToPILImage, ToTensor
from matplotlib import pyplot as plt

from sscc.data.randaugment import RandAugmentSTL10

logger = logging.getLogger(__name__)

PARAMETER_MAX = 10

from torchvision import transforms
from torch.utils import data
from sscc.data.images import ImageDataset

class STL10(ImageDataset):
    """
    """
    base_folder = 'stl10'

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
        super(STL10, self).__init__(root, 
                                      part=part,
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
                c_df = self.build_constraints(y=y, seed=self.seed, fold=self.fold)
                c_df.to_csv(f"{self.dataset_path}/fold_{self.fold}/C_{self.part}.csv")
                return
            return

        trainset = torchvision.datasets.STL10(root=self.dataset_path,
                                              split='train',
                                              download=True,
                                             )
        testset = torchvision.datasets.STL10(root=self.dataset_path,
                                             split='test',
                                             download=True,
                                             )

        # norm to [0, 1]
        X_train = trainset.data / 255.
        Y_train = np.array(trainset.labels)

        X_test = testset.data / 255.
        Y_test = np.array(testset.labels)

        self._split_and_save(x_test=X_test,
                             x_train=X_train,
                             y_test=Y_test,
                             y_train=Y_train)

        os.system('rm {}/stl10_binary.tar.gz'.format(self.dataset_path))

# stl10 training data stats
normalize_stl10 = normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

transforms_stl10_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop(96, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize_stl10,
])

transforms_stl10_weak = transforms.Compose([
    transforms.CenterCrop(size=(96, 96)),
    transforms.ToPILImage(),
    transforms.RandomCrop(96, padding=12),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize_stl10,
])

transforms_stl10_strong = transforms.Compose([
                                    transforms.CenterCrop(size=(96, 96)),
                                    transforms.ToPILImage(),
                                    RandAugmentSTL10(n=2, m=10),
                                    transforms.ToTensor(),
                                    normalize_stl10,
                                    ])

transforms_stl10_test = transforms.Compose([transforms.CenterCrop(size=(96, 96)),
                                            normalize_stl10
                                            ])
