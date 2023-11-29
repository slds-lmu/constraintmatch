import numpy as np
import pandas as pd
import torch
import os
from PIL import Image

from torch.utils import data
from sklearn.model_selection import train_test_split
from torchvision import transforms

class ImageDataset(data.Dataset):
    """Superclass for the image datasets

    Contains the `build_constraints()` method that samples constraints from originally labeled datasets
    """    
    def __init__(self,
                 root: str,
                 part: str,
                 val_size: int,
                 num_constraints: int,
                 num_constraints_val: int,
                 k: int,
                 seed: int=1337,
                 transform=None,
                 label_type: str='constraint',
                 download: bool=True,
                 **kwargs):
        """Image Data Base Class

        Args:
            root (str): the root data path
            part (str): train/val/test dataset 
            val_size (int): size of the validation data set (individual sample level)
            num_constraints (int): amount of constraints to be sampled for the train dataset
            num_constraints_val (int): amount of constraints to be sampled for the val dataset
            k (int): the amount of neighbors per constraint to be sampled. Controls the underlying constraint graph: NULL means random sampling, k=50 means that one randomly selected data point has pairwise connections to 50 other data points (equiv. to a very dense graph)
            seed (int, optional): Seed for constraint sampling to stay reproducible. Defaults to 1337.
            transform ([type], optional): pytorch transform procedure. Defaults to None.
            label_type (str, optional): Get labels as 'constraint' (xi, xj, cij) or 'label' (xi, yi). Defaults to 'constraint'.
            download (bool, optional): should data be downloaded. Defaults to True.
        """               
        if isinstance(root, torch._six.string_classes):
            root = os.path.expanduser(root)
        self.root = root
        self.part = part
        self.transform = transform
        self.val_size = val_size
        self.k = k
        self.num_constraints = num_constraints
        self.num_constraints_val = num_constraints_val
        self.label_type = label_type
        self.seed = seed

        assert self.label_type in ['constraint', 'label'], 'Label type has to be "constraint" or "label"'

    @property
    def size(self):
        if self.label_type == 'constraint':
            return self.c.shape[0]
        else:
            return self.x.shape[0]

    @property
    def num_classes(self):
        return len(np.unique(self.y))

    def __len__(self):
        if self.label_type == 'constraint':
            return self.c.shape[0]
        else:
            return self.x.shape[0]

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
            c_ij = cons_info['c_ij']
            y_i, y_j = cons_info['y_i'], cons_info['y_j']

            x_i = torch.tensor(self.x[i]).to(torch.float32).to(self.device)
            x_j = torch.tensor(self.x[j]).to(torch.float32).to(self.device)
            y_i = torch.tensor(y_i).to(self.device)
            y_j = torch.tensor(y_j).to(self.device)

            if self.transform:
                x_i = self.transform(x_i)
                x_j = self.transform(x_j)

            return x_i, x_j, y_i, y_j, c_ij
        else:
            x = self.x[index]
            y = self.y[index]

            x = torch.tensor(x).to(torch.float32).to(self.device)
            y = torch.tensor(y).to(self.device)

            if self.transform:
                x = self.transform(x)

            return x, y

    def load_dataset(self, fold, part='train'):

        path = os.path.join(self.root, self.base_folder, f'fold_{fold}')

        x = np.load(file=f"{path}/X_{part}.npy").astype('float32')
        y = np.load(file=f"{path}/Y_{part}.npy").astype('int')
        y = y.astype(int)

        assert len(x) == len(y)
        constraints = pd.read_csv(f"{path}/C_{part}.csv")

        return x, y, constraints

    def _split_and_save(self, x_test, x_train, y_test, y_train):
        """
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

            x_train_fold, x_val_fold, y_train_fold, y_val_fold = train_test_split(x_train, y_train,
                                                                                  test_size=self.val_size,
                                                                                  random_state=self.seed,
                                                                                  stratify=y_train)
            
            # build constraints
            c_df_train = self.build_constraints(y_train_fold, seed=self.seed, fold=fold)
            c_df_val = self.build_constraints(y_val_fold, seed=self.seed, fold=fold)
            c_df_test = self.build_constraints(y_test, seed=self.seed, fold=fold)

            # store sampled constraints
            c_df_train.to_csv(f"{dataset_path}/C_train.csv")
            c_df_val.to_csv(f"{dataset_path}/C_val.csv")
            c_df_test.to_csv(f"{dataset_path}/C_test.csv")

            # store split data as .npy array
            np.save(file=f"{dataset_path}/X_train.npy", arr=x_train_fold)
            np.save(file=f"{dataset_path}/X_val.npy", arr=x_val_fold)
            np.save(file=f"{dataset_path}/X_test.npy", arr=x_test)
            np.save(file=f"{dataset_path}/Y_train.npy", arr=y_train_fold)
            np.save(file=f"{dataset_path}/Y_val.npy", arr=y_val_fold)
            np.save(file=f"{dataset_path}/Y_test.npy", arr=y_test)

    def should_download(self) -> bool:
        if not os.path.exists(self.dataset_path):
            os.makedirs(self.dataset_path)
            return True
        else:
            return False

    def build_constraints(self, y: np.ndarray, seed: int=0, fold: int=999) -> np.ndarray:
        """Samples random pairwise constraints.

        self.k controls the amount of neigbors per selected sample to control connectedness of the graph
            - k = NULL: randomly sampled pairs, semi-connected graph
            - k = 1: very dense, no connected parts
            - k = 1000: very connected, 1000 constraints connected to 1 single data sample

        Args:
            y (np.ndarray): vector of class labels
            seed (int, optional): reproc. seed. Defaults to 0.

        Returns:
            (np.ndarray): the constraint matrix C
        """
        if self.part == 'train': 
            num_constraints = self.num_constraints
        elif self.part == 'val':
            num_constraints = self.num_constraints_val

        if self.k:
            assert len(y) * self.k >= num_constraints, f"too few obs: {len(y)} for required num_constraints {num_constraints} given parameter k: {self.k}"

        np.random.seed(seed+fold)
        print(f'=== SEED {seed+fold}, FOLD {fold}')
        idx_sample_basis = np.arange(0, len(y))

        ml_ind1, ml_ind2 = [], []
        cl_ind1, cl_ind2 = [], []

        while num_constraints > 0:
            if self.k:
                # select k partners per sample
                tmp1 = np.random.choice(a=idx_sample_basis, size=1)[0]
                # make sure no idx1 is sampled twice
                idx_sample_basis = idx_sample_basis[idx_sample_basis != tmp1]

                k_cnt = 0
                # now select k partners for that sample tmp1
                # high k -> many partners, low k -> few to 1 partner
                while k_cnt < self.k:
                    tmp2 = np.random.choice(a=idx_sample_basis, size=1)[0]
                    if tmp1 == tmp2:
                        # not a valid choice => a constraint with itself is meaningless
                        continue
                    if y[tmp1] == y[tmp2]:
                        ml_ind1.append(tmp1)
                        ml_ind2.append(tmp2)
                    else:
                        cl_ind1.append(tmp1)
                        cl_ind2.append(tmp2)

                    num_constraints -= 1
                    k_cnt += 1
            else:
                # randomly select two samples from the labeled dataset
                tmp1 = np.random.randint(0, len(y) - 1)
                tmp2 = np.random.randint(0, len(y) - 1)
                if tmp1 == tmp2:
                    # not a valid choice => a constraints with itself is meaningless
                    continue
                if y[tmp1] == y[tmp2]:
                    # Must link constraint
                    ml_ind1.append(tmp1)
                    ml_ind2.append(tmp2)
                else:
                    # Cannot link constraint
                    cl_ind1.append(tmp1)
                    cl_ind2.append(tmp2)

                num_constraints -= 1

        ml_ind1, ml_ind2, cl_ind1, cl_ind2 = np.array(ml_ind1), np.array(ml_ind2), np.array(cl_ind1), np.array(cl_ind2)
        # if self.part == 'train':
            # apply transitivity closure of ML and entailment of CL
            # fills the underlying constraint graph and makes sure we use all information that we have
            # ml_ind1, ml_ind2, cl_ind1, cl_ind2 = self.transitive_closure(ml_ind1, ml_ind2, cl_ind1, cl_ind2, len(y))
            # pass 

        total_constraints = ml_ind1.shape[0] + cl_ind1.shape[0]

        constraints_i = np.hstack((ml_ind1, cl_ind1))
        constraints_j = np.hstack((ml_ind2, cl_ind2))

        c_df = pd.DataFrame(index=np.arange(total_constraints),
                            columns=['idx', 'part', 'i', 'j', 'y_i', 'y_j', 'c_ij'])

        c_df['idx'] = np.arange(total_constraints)
        c_df['part'] = self.part
        c_df['i'] = constraints_i
        c_df['j'] = constraints_j
        c_df['y_i'] = y[constraints_i]
        c_df['y_j'] = y[constraints_j]
        c_df['c_ij'] = np.where(c_df['y_i'] == c_df['y_j'], 1, -1)

        if self.part == 'train':
            nc_print = self.num_constraints 
        elif self.part == 'val':
            nc_print = self.num_constraints_val 
        else:
            nc_print = 999

        print(f'\nI sampled {nc_print} constraints with k={self.k} for part {self.part}\nresulting in {len(c_df)} constraints after TC/CE calculation\n')

        return c_df

    # credit to https://github.com/blueocean92/deep_constrained_clustering
    def transitive_closure(self, ml_ind1, ml_ind2, cl_ind1, cl_ind2, n):
        """
        This function calculate the total transtive closure for must-links and the full entailment
        for cannot-links.

        # Arguments
            ml_ind1, ml_ind2 = instances within a pair of must-link constraints
            cl_ind1, cl_ind2 = instances within a pair of cannot-link constraints
            n = total training instance number

        # Return
            transitive closure (must-links)
            entailment of cannot-links
        """
        ml_graph = dict()
        cl_graph = dict()
        for i in range(n):
            ml_graph[i] = set()
            cl_graph[i] = set()

        def add_both(d, i, j):
            d[i].add(j)
            d[j].add(i)

        for (i, j) in zip(ml_ind1, ml_ind2):
            add_both(ml_graph, i, j)

        def dfs(i, graph, visited, component):
            visited[i] = True
            for j in graph[i]:
                if not visited[j]:
                    dfs(j, graph, visited, component)
            component.append(i)

        visited = [False] * n
        for i in range(n):
            if not visited[i]:
                component = []
                dfs(i, ml_graph, visited, component)
                for x1 in component:
                    for x2 in component:
                        if x1 != x2:
                            ml_graph[x1].add(x2)
        for (i, j) in zip(cl_ind1, cl_ind2):
            add_both(cl_graph, i, j)
            for y in ml_graph[j]:
                add_both(cl_graph, i, y)
            for x in ml_graph[i]:
                add_both(cl_graph, x, j)
                for y in ml_graph[j]:
                    add_both(cl_graph, x, y)
        ml_res_set = set()
        cl_res_set = set()
        for i in ml_graph:
            for j in ml_graph[i]:
                if j != i and j in cl_graph[i]:
                    raise Exception('inconsistent constraints between %d and %d' % (i, j))
                if i <= j:
                    ml_res_set.add((i, j))
                else:
                    ml_res_set.add((j, i))
        for i in cl_graph:
            for j in cl_graph[i]:
                if i <= j:
                    cl_res_set.add((i, j))
                else:
                    cl_res_set.add((j, i))
        ml_res1, ml_res2 = [], []
        cl_res1, cl_res2 = [], []
        for (x, y) in ml_res_set:
            ml_res1.append(x)
            ml_res2.append(y)
        for (x, y) in cl_res_set:
            cl_res1.append(x)
            cl_res2.append(y)

        return np.array(ml_res1), np.array(ml_res2), np.array(cl_res1), np.array(cl_res2)


class ConstraintMatchData(data.Dataset):
    """
    CM Data: dataset that consists solely of unconstrained samples (e.g. those that are not part of any
    constraint)
    """
    def __init__(self,
                 data,
                 weak_transform,
                 strong_transform,
                 **kwargs):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.c = data.c
        self.labelled_idxs = np.unique(np.concatenate((self.c['i'], self.c['j'])))
        self.unlabelled_idxs = np.delete(np.arange(len(data.x)), self.labelled_idxs)

        # # select only unlabelled samples
        # self.x = data.x[self.unlabelled_idxs]
        # self.y = data.y[self.unlabelled_idxs]
        # select ALL samples: constrained samples as well used in CCM!

        self.x = data.x
        self.y = data.y
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform
        # catch case where we read imagenet data from disk 
        self.datatype = 'list' if isinstance(data.x, list) else 'tensor'
        self._resize = transforms.Resize((256, 256))
        self._totensor = transforms.ToTensor()

    @property
    def size(self):
        return len(self.x)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        if self.datatype == 'tensor':
            img = torch.tensor(self.x[index]).to(torch.float32).to(self.device)
        elif self.datatype == 'list':
            img = self.load_image_as_tensor(path=self.x[index]).to(self.device)

        img_weak = self.weak_transform(img)
        img_strong = self.strong_transform(img)
        y = torch.tensor(self.y[index]).to(torch.int64).to(self.device)

        return img_weak, img_strong, y

    def load_image_as_tensor(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        return self._totensor(self._resize(img))