import torch
import os
import tempfile
import yaml

from pytorch_lightning.loggers import MLFlowLogger

from sscc.models.constrained import Constrained
from sscc.models.pseudolabel import PseudoLabel
from sscc.models.constraintmatch import ConstraintMatch
from sscc.architectures.resnets import resnet18, ClusteringModel, resnet34

models = {'constrained': Constrained,
          'pseudolabel': PseudoLabel,
          'constraintmatch': ConstraintMatch}

def update_config(config, args):
    for name, value in vars(args).items():
        if value is None:
            continue
        for key in config.keys():
            for subkey in config[key].keys():
                if subkey == name:
                    config[key][subkey] = value
                elif config[key][subkey].__class__ == dict:
                    for subsubkey in config[key][subkey]:
                        if subsubkey == name:
                            config[key][subkey][subsubkey] = value
    return config


def save_dict_as_yaml_mlflow(data: dict, logger: MLFlowLogger, filename: str = 'config.yaml'): 
    """Store any dict in mlflow as an .yaml artifact

    Args:
        data (dict): input file, e.g. config 
        logger (MLFlowLogger): pytorch lightning mlflow logger (could be extended)
        filename (str): name for storage in mlflow artifacts
    """    
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, filename)
        with open(path, 'w') as file:
            yaml.dump(data, file, default_flow_style=False)
            logger.experiment.log_artifact(local_path=path, run_id=logger.run_id)

def parse_architecture_config(config):
    arch_params = config.get('model_params').get('architecture')
    architecture = get_scan_resnet(dataset=config['exp_params']['dataset'], **arch_params)
    return architecture

def parse_model_config(config):
    model_param = config.get('model_params')
    model = models[model_param['model']]

    architecture = parse_architecture_config(config)
    model_instance = model(model=architecture, loss=model_param['loss'], val_loss=model_param['val_loss'])

    return model_instance

def get_scan_resnet(num_classes: int=10, freeze: bool=False, pretrained: bool=False, dataset: str ='cifar10', **kwargs):
    if dataset in ['imagenetDogs', 'imagenet10', 'tinyimagenet']:
        backbone = resnet34(dataset=dataset) 
    else:
        backbone = resnet18(dataset=dataset) 

    # Setup
    model = ClusteringModel(backbone=backbone, nclusters=num_classes, nheads=1)
    # Load pretrained weights    
    if pretrained:
        if num_classes == 20 and dataset == 'cifar20':
            pretrain_path=f'./data/model_weights/selflabel_cifar-20.pth.tar'
            state_dict = torch.load(pretrain_path, map_location='cpu')
            model.load_state_dict(state_dict)
        elif num_classes == 100 and dataset == 'cifar20':
            pretrain_path=f'./data/model_weights/selflabel_cifar-20_100.pth.tar'
            state_dict = torch.load(pretrain_path, map_location='cpu')
            model.load_state_dict(state_dict)            
        elif num_classes == 10 and dataset == 'cifar10':
            pretrain_path=f'./data/model_weights/selflabel_cifar-10.pth.tar'
            state_dict = torch.load(pretrain_path, map_location='cpu')
            model.load_state_dict(state_dict)
        elif num_classes == 50 and dataset == 'cifar10':
            pretrain_path=f'./data/model_weights/selflabel_cifar-10_50.pth.tar'
            state_dict = torch.load(pretrain_path, map_location='cpu')
            model.load_state_dict(state_dict)
        elif num_classes == 10 and dataset == 'stl10':
            pretrain_path=f'./data/model_weights/selflabel_stl-10.pth.tar'
            state_dict = torch.load(pretrain_path, map_location='cpu')
            model.load_state_dict(state_dict)            
        elif num_classes == 10 and dataset == 'imagenet10':
            pretrain_path=f'./data/model_weights/selflabel_imagenet10.pth.tar'
            state_dict = torch.load(pretrain_path, map_location='cpu')
            model.load_state_dict(state_dict)    
        elif num_classes == 15 and dataset == 'imagenetDogs':
            pretrain_path=f'./data/model_weights/selflabel_imagenetdog.pth.tar'
            state_dict = torch.load(pretrain_path, map_location='cpu')
            model.load_state_dict(state_dict)    
        else: 
            print('No model weights for this dataset/ num_classes combination')
        
        # freeze all layers but the head for the linear evaluation protocol
        if freeze:
            for name, param in model.named_parameters():
                if not name.startswith('cluster_head'): param.requires_grad = False
    return model