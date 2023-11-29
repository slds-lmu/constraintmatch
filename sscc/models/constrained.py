from tqdm import tqdm
import torch.nn as nn
import torch
import numpy as np
import tempfile
import os 
import pandas as pd

from torch.nn import functional as F
from typing import Any
from sscc.losses import loss_dict
from sscc.metrics import Evaluator, compute_calibration_metrics
from sscc.data.utils import PairEnum


class Constrained(nn.Module):
    """
    """
    def __init__(self, model, loss, val_loss):
        super(Constrained, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device).to(torch.float32)
        self.criterion = loss_dict[loss['loss']](**loss).to(self.device)
        self.val_criterion = loss_dict[val_loss['loss']](**loss).to(self.device)
        self.cnt = 0

    def forward(self, batch, **kwargs):
        """Normal forward pass
        train_target = pairwise constraints;
        len(train_target) = batch_size**2
        eval_target = actual labeles used for evaluation
        """
        images = batch['images'].to(self.device)
        logits = self.model(images)
        out = F.softmax(logits, dim=1)

        return {'out': out}
    
    def loss_function(self, outs, batch, train=True, **kwargs):
        """
        """
        out = outs['out']
        train_target = batch['train_target'].to(self.device)
        prob1, prob2 = PairEnum(out)
        if self.model.training:
            loss = self.criterion(prob1, prob2, simi=train_target)
        else: 
            loss = self.val_criterion(prob1, prob2, simi=train_target)
        return {'loss': loss}

    def training_step(self, batch, params, batch_idx=0, step=0, **kwargs):
        """Single training step as called in the experiment template

        combines model.forward() and model.loss_function()

        Args:
            batch ([type]): [description]
            batch_idx ([type]): [description]
        """
        self.cnt += 1
        outs = self.forward(batch=batch)
        loss = self.loss_function(outs=outs, batch=batch, **params)
        return loss

    def evaluate(self, eval_dataloader: Any, confusion: Any=Evaluator(10), part: str='val', logger: Any=None, true_k: int=10):
        """Evaluate model on any dataloader during training and testings

        Args:
            eval_dataloader (Any): data loader for evaluation
            confusion (Any, optional): the evaluator object as definsed in sscc.metrics. Defaults to Evaluator(10).
            part (str, optional): train/test/val . Defaults to 'val'.
            logger (Any, optional): any pl logger. Defaults to None.
            true_k (int, optional): true clusters, important for confusion matrix plotting. Defaults to 10.

        Returns:
            None
        """   
        y = []
        pred = []
        losses = []
        n_samples = 0 
        n_constraints = 0
        print(f'Evaluating {part} dataset ...')
        for i, batch in tqdm(enumerate(eval_dataloader)):
            with torch.no_grad(): outs = self.forward(batch=batch)
            n_samples += len(batch['images'])
            n_constraints += sum(batch['train_target'] != 0.0)
            # add constrained loss calculation here, no gradients needed
            with torch.no_grad(): losses.append(self.loss_function(outs, batch)['loss'].item())
            confusion.add(outs['out'].detach().cpu(), batch['eval_target'])
            y.extend(batch['eval_target'].detach().cpu().numpy())
            pred.append(outs['out'].detach().cpu().numpy())

        c_1 = sum(eval_dataloader.dataset.c['c_ij'] == 1)
        c_minus1 = sum(eval_dataloader.dataset.c['c_ij'] == -1)
        total_c = len(eval_dataloader.dataset.c['c_ij'])
        print(f'\nDL: {part}, ML/(CL+ML): {round(c_1/(c_1 + c_minus1), 4)}, total #C {total_c}')

        pred = np.concatenate(pred, axis=0)
        orig_idx, assigned_idx = confusion.optimal_assignment(confusion.k)
        pred = pred[:, assigned_idx]
        calib_metrics = compute_calibration_metrics(true_labels=y, confidences=pred, num_bins=15)
        pred_nent = np.mean(np.sum(a=-pred * np.log(pred), axis=1) / np.log(pred.shape[1]))

        print('\n')
        print(f"{part} ==> clustering scores: {confusion.clusterscores()}")
        print(f"{part} ==> accuracy: {confusion.acc()}")
        print(f"{part} ==> loss: {sum(losses)/len(losses)}")
        print(f"{part} ==> ece: {calib_metrics['expected_calibration_error']}")
        # store in dictionary for mlflow logging
        eval_results = {f'{part}_acc': confusion.acc()}
        eval_results.update({f'{part}_{key}': value for key, value in confusion.clusterscores().items()})
        eval_results.update({f'{part}_loss': sum(losses)/len(losses)})
        eval_results.update({f'{part}_avg_accuracy': calib_metrics['avg_accuracy'],
                             f'{part}_avg_confidence': calib_metrics['avg_confidence'],
                             f'{part}_expected_calibration_error': calib_metrics['expected_calibration_error'],
                             f'{part}_max_calibration_error': calib_metrics['max_calibration_error'],
                             f'{part}_overconfidence_error': calib_metrics['overconfidence_error'],
                             f'{part}_pred_nent': pred_nent})

        if part != 'train':
            epoch_string=f'{self.epoch}'.zfill(4)
            logger.experiment.log_figure(figure=confusion.plot_confmat(title=f'{part} set, epoch {self.epoch}',
                                                                       true_k=true_k),
                                         artifact_file=f'confmat_{part}_{epoch_string}.png',
                                         run_id=logger.run_id)

        if part in ['test', 'all']:
            final_results = {f'yhat_p_{cl}': pred[:, cl] for cl in range(pred.shape[1])}
            final_results['y'] = y
            final_results['yhat'] = np.argmax(pred, 1)
            final_results = pd.DataFrame(final_results)
            with tempfile.TemporaryDirectory() as tmp_dir:
                storage_path = os.path.join(tmp_dir, f'{part}_preds.csv')
                final_results.to_csv(storage_path)
                logger.experiment.log_artifact(local_path=storage_path, run_id=logger.run_id)


        return eval_results

