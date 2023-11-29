import torch

from torch.utils.data.dataloader import default_collate
from sscc.data.cifar10 import CIFAR10, transforms_cifar10_train, transforms_cifar10_test
from sscc.data.cifar20 import CIFAR20
from sscc.data.imagenetDogs import IMAGENETDOGS, transforms_imagenetDogs_train, transforms_imagenetDogs_test
from sscc.data.imagenet10 import IMAGENET10, transforms_imagenet10_train, transforms_imagenet10_test
from sscc.data.stl10 import STL10, transforms_stl10_train, transforms_stl10_test

def constrained_collate_fn(batch, data_collate=default_collate):
    """
    Collates the constrained samples only
    one sample of a batch consists of 5 elements: 
        1) xi
        2) xj
        3) yi
        4) yj
        5) cij
    """
    # from timeit import default_timer as timer; start = timer()
    transposed_data = list(zip(*batch))
    data = [data_collate(b) for b in transposed_data]

    # stack images x_i and x_j
    images = torch.cat((data[0], data[1]), dim=0)
    # stack labels y_i and y_j
    labels = torch.cat((data[2], data[3]), dim=0)
    constraints = data[4]
    # rearrange pre-specified constraints to make them trainable
    train_target, eval_target = prepare_task_target(labels, constraints)
    return {'images': images.float(), 'train_target': train_target, 'eval_target': eval_target}


def supervised_collate_fn(batch, data_collate=default_collate):
    """
    Collates supervised, labeled samples
    """
    transposed_data = list(zip(*batch))
    data = [data_collate(b) for b in transposed_data]

    train_target, eval_target = prepare_supervised_task_target(target=data[1])

    return {'images': data[0].float(), 'train_target': train_target, 'eval_target': eval_target}


def constraint_match_collate_fn(batch, data_collate=default_collate):
    """
    Collates the unconstrained samples
    """
    transposed_data = list(zip(*batch))
    data = [data_collate(b) for b in transposed_data]

    return {'weakly_aug': data[0].float(), 'strongly_aug': data[1].float(), 'y': data[2]}


def prepare_supervised_task_target(target):
    """
    """
    train_target = Class2Simi(x=target)
    eval_target = target

    return train_target.detach(), eval_target.detach()


def prepare_task_target(target, constraints):
    """
    """
    train_target = Constraints2Simi(x=target, constraints=constraints)
    eval_target = target

    return train_target.detach(), eval_target.detach()


def Constraints2Simi(x, constraints, mask=None):
    """
    """
    n = int(x.shape[0]/2)
    trivial_constraints = torch.eye(n) * 0
    prespecified_constraints = torch.diag(constraints)
    out = torch.vstack([torch.hstack([trivial_constraints, prespecified_constraints]), torch.hstack([prespecified_constraints, trivial_constraints])])
    out = out.float()

    if mask is None:
        out = out.view(-1)
    else:
        mask = mask.detach()
        out = out[mask]
    return out


def Class2Simi(x, mask=None):
    # Convert class label to pairwise similarity
    n=x.nelement()
    assert (n-x.ndimension()+1) == n, 'Dimension of Label is not correct'
    expand1 = x.view(-1,1).expand(n,n)
    expand2 = x.view(1,-1).expand(n,n)
    out = expand1 - expand2
    out[out!=0] = -1 #dissimilar pair: label=-1
    out[out==0] = 1 #Similar pair: label=1
    out = out.float()

    if mask is None:
        out = out.view(-1)
    else:
        mask = mask.detach()
        out = out[mask]
    return out


def Prob2Simi(y, mask=None):
    # Convert max probs to confidence scores for ccm2
    n=y.nelement()
    assert (n-y.ndimension()+1) == n, 'Dimension of Label is not correct'
    expand1 = y.view(-1,1).expand(n,n)
    expand2 = y.view(1,-1).expand(n,n)
    confs = expand1 * expand2

    if mask is None:
        return confs.view(-1)
    else:
        mask = mask.detach()
        return confs[mask]


def Dot2Simi(y, mask=None):
    # Convert yi * yj to confidence scores for ccm3
    confs = y.mm(y.t())

    if mask is None:
        return confs.view(-1)
    else:
        mask = mask.detach()
        return confs[mask]


def IJS2Simi(y):
    """Calculate inverse jensen shannon distance across all element combinations in y

    js=srtq{[D(p||m) + D(q||m)]/2} where m = (p+q)/2 \in [0; 1]

    ijs = 1 - js

    see https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.jensenshannon.html

    Args:
        y (torch.Tensor): softmax prediction vector in shape (n, classes)

    Returns:
        torch.Tensor: IJS scores for each yi,yj combination with shape (n**2). [0; 1] => [CL, ML]
    """    
    def _kldiv2(p, q):
        # helper function for KL with
        # add increment for stability 
        p += 1e-35
        q += 1e-35
        # use torch.log() to use base e, acc. to wikipedia etc. base 2 is the real stuff
        # compare with scipy's jensenshannon() function
        return (p * (torch.log2(p) - torch.log2(q))).sum(1)

    # js's for all combinations
    yi, yj = PairEnum(y)
    m = (yi + yj) / 2 
    # js=srtq{[D(p||m) + D(q||m)]/2} where m = (p+q)/2
    js = ((_kldiv2(yi, m) + _kldiv2(yj, m)) / 2)**0.5
    return 1.0 - js


def PairEnum(x,mask=None):
    # Enumerate all pairs of feature in x
    assert x.ndimension() == 2, 'Input dimension must be 2'
    x1 = x.repeat(x.size(0),1)
    x2 = x.repeat(1,x.size(0)).view(-1,x.size(1))
    if mask is not None:
        xmask = mask.view(-1,1).repeat(1,x.size(1))
        #dim 0: #sample, dim 1:#feature
        x1 = x1[xmask].view(-1,x.size(1))
        x2 = x2[xmask].view(-1,x.size(1))
    return x1,x2


def get_data(root, params, log_params, part, label_type):
    """
    """
    if params['dataset'] == 'cifar10':
        data = CIFAR10(root=root,
                       part=part,
                       val_size=params['val_size'],
                       num_constraints=params['num_constraints'],
                       num_constraints_val=params['num_constraints_val'],
                       k=params['k'],
                       transform=transforms_cifar10_train if part=='train' else transforms_cifar10_test,
                       label_type=label_type,
                       fold=params['fold'],
                       seed=log_params['manual_seed'])
    elif params['dataset'] == 'cifar20':
        data = CIFAR20(root=root,
                       part=part,
                       val_size=params['val_size'],
                       num_constraints=params['num_constraints'],
                       num_constraints_val=params['num_constraints_val'],
                       k=params['k'],
                       transform=transforms_cifar10_train if part=='train' else transforms_cifar10_test,
                       label_type=label_type,
                       fold=params['fold'],
                       seed=log_params['manual_seed'])
    if params['dataset'] == 'stl10':
        data = STL10(root=root,
                     part=part,
                     val_size=params['val_size'],
                     num_constraints=params['num_constraints'],
                     num_constraints_val=params['num_constraints_val'],
                     k=params['k'],
                     transform=transforms_stl10_train if part=='train' else transforms_stl10_test,
                     label_type=label_type,
                     fold=params['fold'],
                     seed=log_params['manual_seed'])
    if params['dataset'] == 'imagenetDogs':
        data = IMAGENETDOGS(root=root,
                            part=part,
                            val_size=params['val_size'],
                            num_constraints=params['num_constraints'],
                            num_constraints_val=params['num_constraints_val'],
                            k=params['k'],
                            transform=transforms_imagenetDogs_train if part=='train' else transforms_imagenetDogs_test,
                            label_type=label_type,
                            fold=params['fold'],
                            seed=log_params['manual_seed'])
    if params['dataset'] == 'imagenet10':
        data = IMAGENET10(root=root,
                          part=part,
                          val_size=params['val_size'],
                          num_constraints=params['num_constraints'],
                          num_constraints_val=params['num_constraints_val'],
                          k=params['k'],
                          transform=transforms_imagenet10_train if part=='train' else transforms_imagenet10_test,
                          label_type=label_type,
                          fold=params['fold'],
                          seed=log_params['manual_seed'])
    return data


true_k_dict = {'cifar10': 10, 
               'cifar20': 20, 
               'stl10': 10,
               'imagenet10': 10,
               'imagenetDogs': 15}