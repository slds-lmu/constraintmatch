import torch
import pdb
import numpy as np
from matplotlib import pyplot as plt
from torchvision import transforms
import os
import shutil


def plot_xi(xi, fname: str = 'foo.png'):
    """[summary]

    Args:
        xi ([type]): [description]
        fname ([type]): [description]
    """
    dirname = fname.split('/')[0]
    if not os.path.exists(dirname): os.makedirs(dirname)
    # re-normalize images
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    for dim in [0, 1, 2]:
        xi[dim] = xi[dim] * stds[dim] + means[dim]
        
    plt.close()
    plt.figure(figsize=(5, 5))
    plt.imshow(xi)
    plt.title('Xi')
    plt.tight_layout()
    plt.savefig(fname)

def plot_ci(xi, xj, xzi, xzj, cij: int = 99, fname: str = 'foo.png'):
    """[summary]

    Args:
        xi ([type]): [description]
        xj ([type]): [description]
        xz ([type]): [description]
        fname ([type]): [description]
    """
    dirname = fname.split('/')[0]
    if not os.path.exists(dirname): os.makedirs(dirname)
    # re-normalize images
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    for x in [xi, xj, xzi, xzj]:
        for dim in [0, 1, 2]:
            x[dim] = x[dim] * stds[dim] + means[dim]
        
    plt.close()
    plt.figure(figsize=(20, 5))
    plt.subplot(1,4,1)
    plt.imshow(transforms.ToPILImage()(xi))
    plt.title('Xi')

    plt.subplot(1,4,2)
    plt.imshow(transforms.ToPILImage()(xzi))
    plt.title('Xzi')

    plt.subplot(1,4,3)
    plt.imshow(transforms.ToPILImage()(xzj))
    plt.title('Xzj')

    plt.subplot(1,4,4)
    plt.imshow(transforms.ToPILImage()(xj))
    plt.title('Xj')

    plt.suptitle(f'Constraint {cij}')
    plt.tight_layout()
    plt.savefig(fname)