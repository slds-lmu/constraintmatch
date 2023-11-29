#!/usr/bin/env python
import yaml
import argparse
import torch 
import numpy as np
import time
import torch.backends.cudnn as cudnn

from sscc.experiments import Experiment
from sscc.utils import save_dict_as_yaml_mlflow
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from sscc.utils import *

torch.set_default_dtype(torch.float32)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c',
                        dest='filename',
                        metavar='FILE',
                        help='path to config file',
                        default='configs/fmnist_supervised.yaml')
    parser.add_argument('--constraint_interpolation', type=str, default=None,
                        help="CI method, values: image/ feature/ False aka None")
    parser.add_argument('--num_classes', type=int, default=None,
                        help='amount of a priori classes')                        
    parser.add_argument('--learning_rate', type=float, default=None,
                        help="learning rate")
    parser.add_argument('--momentum', type=float, default=None,
                        help="momentum rate")
    parser.add_argument('--weight_decay', type=float, default=None,
                        help="weight_decay")
    parser.add_argument('--batch_size', type=int, default=None,
                        help="batch size")
    parser.add_argument('--batch_size_ul', type=int, default=None,
                        help="batch size")                        
    parser.add_argument('--scheduler', type=str, default=None,
                        help="lr scheduler [None, cosine, ...]")
    parser.add_argument('--arch', type=str, default=None,
                        help="different architectures to choose from: [lenet,vgg, ...]")
    parser.add_argument('--num_constraints', type=int, default=None,
                        help='number of total constraints for training')
    parser.add_argument('--num_constraints_val', type=int, default=None,
                        help='number of total constraints for validation')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='name of the experiment')
    parser.add_argument('--run_name', type=str, default=None,
                        help='name of the run')
    parser.add_argument('--manual_seed', type=int, default=None,
                        help='seed for training')
    parser.add_argument('--dataset', type=str, default=None,
                        help='dataset')
    parser.add_argument('--k', type=int, default=None,
                        help="amount of neighbors/ samples, controls connectedness")
    parser.add_argument('--lambda_unsup', type=float, default=None,
                        help="unsupervised loss weight for constraintmatch")
    parser.add_argument('--threshold', type=float, default=None,
                        help="threshold for unsup. loss weight for constraintmatch")
    parser.add_argument('--fold', type=int, default=None,
                        help='seed for training')
    parser.add_argument('--plot', type=int, default=None,
                        help="threshold for unsup. loss weight for constraintmatch")
    parser.add_argument('--num_workers', type=int, default=None,
                        help="number of epochs to train")
    parser.add_argument('--val_every_n_epoch', type=int, default=None,
                        help='speed up training evaluating less')
    parser.add_argument('--loss_type', type=str, default=None,
                        help="controls the different CCM variants {ccm_3, ccm_4, ccm_5}")
    args = parser.parse_args()
    return args

def run_experiment(args):
    # torch.multiprocessing.set_start_method('spawn')
    time.sleep(1)
    with open(args.filename, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    # update config
    config = update_config(config=config, args=args)
    # compile model
    model = parse_model_config(config)
    # instantiate logger
    mlflow_logger = MLFlowLogger(experiment_name=config['logging_params']['experiment_name'])
    # for reproducibility
    torch.manual_seed(config['logging_params']['manual_seed'])
    np.random.seed(config['logging_params']['manual_seed'])
    cudnn.deterministic = True
    cudnn.benchmark = True

    # log all mlflow params
    for k, single_config in config.items():
        if k != 'search_space':
            mlflow_logger.log_hyperparams(params=single_config)

    # store the config
    save_dict_as_yaml_mlflow(data=config, logger=mlflow_logger)

    experiment = Experiment(model,
                            params=config['exp_params'],
                            log_params=config['logging_params'],
                            trainer_params=config['trainer_params'],
                            run_name=config['logging_params']['run_name'],
                            experiment_name=config['logging_params']['experiment_name'])

    runner = Trainer(reload_dataloaders_every_epoch=False,
                     min_epochs=1,
                     precision=16,
                     log_every_n_steps=10,
                     checkpoint_callback=True,
                     logger=mlflow_logger,
                     check_val_every_n_epoch=1 if not 'val_every_n_epoch' in config['exp_params'] else config['exp_params']['val_every_n_epoch'],
                     num_sanity_val_steps=5,
                     fast_dev_run=False,
                     multiple_trainloader_mode='min_size',
                     callbacks=[LearningRateMonitor(logging_interval='step')],
                     **config['trainer_params']
                     )

    runner.fit(experiment)
    runner.test()

if __name__ == "__main__":
    args = parse_args()
    run_experiment(args)