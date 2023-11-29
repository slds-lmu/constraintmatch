# Credit: Wouter v Gansbeke
# List of augmentations based on randaugment
import torch
from tqdm import tqdm
import os
import numpy as np
import logging
import PIL.ImageOps
import PIL.ImageEnhance
import PIL.ImageDraw
import numpy as np
import pandas as pd
from torchvision import transforms
from PIL import Image
from glob import glob
from sklearn.model_selection import train_test_split

import random
import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw
import numpy as np
import torch


logger = logging.getLogger(__name__)

PARAMETER_MAX = 10

from torchvision import transforms
from sscc.data.images import ImageDataset

class IMAGENET10(ImageDataset):
    """
    """
    base_folder = 'imagenet10'

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
        super(IMAGENET10, self).__init__(root, 
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
        self._resize = transforms.Resize(256)
        self._totensor = transforms.ToTensor()
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
            return len(self.x)

    def __getitem__(self, index):
        """the iterator over indices work differently for train and val/test data.
        For constraint: we only want to iterate over the pre-specified constraints.
                   There one sample is one constraint consisting of two observations (as defined in C_train.csv)
                   The final constraint matrix is then built in data.uitls.constrained_collate_fn()
        For constraint: we want to iterate over the whole data.
                   For this, one sample corresponds to one observation.
                   The constraints are then built in data.utils.supervised_collate_fn()
        """
        if self.label_type == 'constraint':
            cons_info = self.c.iloc[index, :]
            i, j = cons_info['i'], cons_info['j']
            x_i = self.load_image_as_tensor(path=self.x[i]).to(self.device)
            x_j = self.load_image_as_tensor(path=self.x[j]).to(self.device)
            c_ij = cons_info['c_ij']
            y_i, y_j = cons_info['y_i'], cons_info['y_j']

            y_i = torch.tensor(y_i).to(self.device)
            y_j = torch.tensor(y_j).to(self.device)

            if self.transform:
                x_i = self.transform(x_i)
                x_j = self.transform(x_j)

            return x_i, x_j, y_i, y_j, c_ij
        else:
            x = self.load_image_as_tensor(path=self.x[index]).to(self.device)
            y = self.y[index]
            y = torch.tensor(y).to(self.device)

            if self.transform:
                x = self.transform(x)

            return x, y

    def load_image_as_tensor(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        return self._totensor(self._resize(img))

    def download(self):

        if not self.should_download():
            # same process for train and val dataset, test is "labeled"
            if self.label_type == 'constraint':
                _, y, _ = self.load_dataset(part=self.part, fold=self.fold)
                c_df = self.build_constraints(y=y, seed=self.seed+self.fold)
                c_df.to_csv(f"{self.dataset_path}/fold_{self.fold}/C_{self.part}.csv")
                return
            return

        ### training 
        trainpath = os.path.join(self.dataset_path, 'train10')
        subdirs = os.listdir(trainpath)
        subdirs.sort()

        # Gather the files (sorted)
        train_imgs = []
        for classidx, subdir in enumerate(subdirs):
            files = sorted(glob(os.path.join(trainpath, subdir, '*.JPEG')))
            for f in files:
                train_imgs.append((f, classidx)) 

        # stupid: loop over all 64k images and read labels
        Y_train = []
        path_train = []
        for idx in tqdm(range(len(train_imgs))):
            path = train_imgs[idx][0]
            y = train_imgs[idx][1]
            path_train.append(path)
            Y_train.append(y)
        Y_train = np.asarray(Y_train)

        ### Test 
        valpath = os.path.join(self.dataset_path, 'val10')
        subdirs = os.listdir(valpath)
        subdirs.sort()
        
        # Gather the files (sorted)
        test_imgs = []
        for classidx, subdir in enumerate(subdirs):
            files = sorted(glob(os.path.join(valpath, subdir, '*.JPEG')))
            for f in files:
                test_imgs.append((f, classidx)) 

        # stupid: loop over all 64k images and read labels
        Y_test = []
        path_test = []
        print(f'loading {len(test_imgs)} test imagenet jpegs')
        for idx in tqdm(range(len(test_imgs))):
            path = test_imgs[idx][0]
            y = test_imgs[idx][1]
            path_test.append(path)
            Y_test.append(y)
        Y_test = np.asarray(Y_test)

        self._split_and_save(img_test=path_test,
                             img_train=path_train,
                             y_test=Y_test,
                             y_train=Y_train)

    def _split_and_save(self, img_test, img_train, y_test, y_train):
        """
        overwritten for imagenet50, can not store them as ndarrays

        helper method to split train/test data into train/val/test data and store them as .npy arrays
        samples constraints from the labeled datasets on the fly

        Input: 
            * labeled train dataset (X, Y)
            * labeled test dataset (X, Y)

        Output:
            * labeled train dataset (X, Y, C)
            * labeled val dataset (X, Y, C)
            * labeled test dataset (X, Y, C)
        
        Where C refers to the constraint-matrix
        """
        folds = 5
        for fold in range(folds):
            # reset seed
            np.random.seed(self.seed)

            dataset_path = os.path.join(self.root, self.base_folder, f'fold_{fold}')
            os.mkdir(dataset_path)
            y_train_fold, y_val_fold = train_test_split(y_train, test_size=self.val_size, random_state=self.seed, stratify=y_train)
            img_train_fold, img_val_fold = train_test_split(img_train, test_size=self.val_size, random_state=self.seed, stratify=y_train)
            
            # build constraints
            c_df_train = self.build_constraints(y_train_fold, seed=self.seed, fold=fold)
            c_df_val = self.build_constraints(y_val_fold, seed=self.seed, fold=fold)
            c_df_test = self.build_constraints(y_test, seed=self.seed, fold=fold)

            # store sampled constraints
            c_df_train.to_csv(f"{dataset_path}/C_train.csv")
            c_df_val.to_csv(f"{dataset_path}/C_val.csv")
            c_df_test.to_csv(f"{dataset_path}/C_test.csv")

            # store split labels as .npy array
            np.save(file=f"{dataset_path}/Y_train.npy", arr=y_train_fold)
            np.save(file=f"{dataset_path}/Y_val.npy", arr=y_val_fold)
            np.save(file=f"{dataset_path}/Y_test.npy", arr=y_test)

            # store lists with data paths
            train_paths = open(f"{dataset_path}/paths_train.txt", "w")
            for path in img_train_fold: train_paths.write(path + '\n')
            val_paths = open(f"{dataset_path}/paths_val.txt", "w")
            for path in img_val_fold: val_paths.write(path + '\n')
            test_paths = open(f"{dataset_path}/paths_test.txt", "w")
            for path in img_test: test_paths.write(path + '\n')

    def load_dataset(self, fold, part='train'):

        path = os.path.join(self.root, self.base_folder, f'fold_{fold}')

        with open(f'{path}/paths_{part}.txt', 'r') as file: 
            x = file.read().splitlines()        
        y = np.load(file=f"{path}/Y_{part}.npy")
        y = y.astype(int)
        assert len(x) == len(y)
        constraints = pd.read_csv(f"{path}/C_{part}.csv")

        return x, y, constraints

    def should_download(self) -> bool:
        if not os.path.exists(f'{self.dataset_path}/fold_0'):
            return True
        else:
            return False



random_mirror = True

def ShearX(img, v):
    if random_mirror and random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))

def ShearY(img, v):
    if random_mirror and random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))

def Identity(img, v):
    return img

def TranslateX(img, v):
    if random_mirror and random.random() > 0.5:
        v = -v
    v = v * img.size[0]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))

def TranslateY(img, v):
    if random_mirror and random.random() > 0.5:
        v = -v
    v = v * img.size[1]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))

def TranslateXAbs(img, v):
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))

def TranslateYAbs(img, v):
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))

def Rotate(img, v):
    if random_mirror and random.random() > 0.5:
        v = -v
    return img.rotate(v)

def AutoContrast(img, _):
    return PIL.ImageOps.autocontrast(img)

def Invert(img, _):
    return PIL.ImageOps.invert(img)

def Equalize(img, _):
    return PIL.ImageOps.equalize(img)

def Solarize(img, v):
    return PIL.ImageOps.solarize(img, v)

def Posterize(img, v):
    v = int(v)
    return PIL.ImageOps.posterize(img, v)

def Contrast(img, v):
    return PIL.ImageEnhance.Contrast(img).enhance(v)

def Color(img, v):
    return PIL.ImageEnhance.Color(img).enhance(v)

def Brightness(img, v):
    return PIL.ImageEnhance.Brightness(img).enhance(v)

def Sharpness(img, v):
    return PIL.ImageEnhance.Sharpness(img).enhance(v)

def augment_list():
    l = [
        (Identity, 0, 1),  
        (AutoContrast, 0, 1),
        (Equalize, 0, 1), 
        (Rotate, -30, 30),
        (Solarize, 0, 256),
        (Color, 0.05, 0.95),
        (Contrast, 0.05, 0.95),
        (Brightness, 0.05, 0.95),
        (Sharpness, 0.05, 0.95),
        (ShearX, -0.1, 0.1),
        (TranslateX, -0.1, 0.1),
        (TranslateY, -0.1, 0.1),
        (Posterize, 4, 8),
        (ShearY, -0.1, 0.1),
    ]
    return l


augment_dict = {fn.__name__: (fn, v1, v2) for fn, v1, v2 in augment_list()}

class Augment:
    def __init__(self, n):
        self.n = n
        self.augment_list = augment_list()

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)
        for op, minval, maxval in ops:
            val = (random.random()) * float(maxval - minval) + minval
            img = op(img, val)

        return img

def get_augment(name):
    return augment_dict[name]

def apply_augment(img, name, level):
    augment_fn, low, high = get_augment(name)
    return augment_fn(img.copy(), level * (high - low) + low)

class Cutout(object):
    def __init__(self, n_holes, length, random=False):
        self.n_holes = n_holes
        self.length = length
        self.random = random

    def __call__(self, img):
        h = img.size(1)
        w = img.size(2)
        length = random.randint(1, self.length)
        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - length // 2, 0, h)
            y2 = np.clip(y + length // 2, 0, h)
            x1 = np.clip(x - length // 2, 0, w)
            x2 = np.clip(x + length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


normalize_imagenet10 = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

transforms_imagenet10_test = transforms.Compose([transforms.ToPILImage(),
                                                 transforms.CenterCrop(224),
                                                 transforms.ToTensor(), 
                                                 normalize_imagenet10]
                                                )

transforms_imagenet10_train = transforms.Compose([
                                                  transforms.ToPILImage(),
                                                  transforms.CenterCrop(224),
                                                  transforms.RandomCrop(224),
                                                  transforms.RandomHorizontalFlip(),
                                                  transforms.ToTensor(),
                                                  normalize_imagenet10]
                                                )
                                           
transforms_imagenet10_weak = transforms.Compose([
                                               transforms.ToPILImage(),
                                               transforms.CenterCrop(224),
                                               transforms.RandomCrop(224),
                                               transforms.RandomHorizontalFlip(),
                                               transforms.ToTensor(),
                                               normalize_imagenet10]
                                               )

transforms_imagenet10_strong = transforms.Compose([transforms.ToPILImage(),
                                                   transforms.RandomHorizontalFlip(),
                                                   transforms.CenterCrop(224),
                                                   transforms.RandomCrop(224),
                                                   Augment(4),
                                                   transforms.ToTensor(),
                                                   normalize_imagenet10,
                                                   Cutout(n_holes = 1, length = 75, random = True)]
                                                )