from tqdm import tqdm
import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import os 
import pandas as pd
import random

from typing import Any
from torch.nn import functional as F
from sscc.losses import loss_dict
from sscc.metrics import Evaluator, compute_calibration_metrics
from sscc.data.utils import PairEnum, IJS2Simi
from scipy.special import entr


class ConstraintMatch(nn.Module):
    """Constrained ConstraintMatch Model 
    """
    def __init__(self, model, loss, val_loss):
        super(ConstraintMatch, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device).to(torch.float32)
        self.criterion = loss_dict[loss['loss']](**loss).to(self.device)
        self.val_criterion = loss_dict[val_loss['loss']](**loss).to(self.device)

    def forward(self, batch, **kwargs):          
        if len(batch) == 3:
            images = batch['images'].to(self.device)
            logits = self.model(images)
            out = F.softmax(logits, dim=1)
            return {'out': out}
        else:
            # forward pass for constrained observations
            images = batch['supervised_train']['images'].to(self.device) 
            logits = self.model(images)
            out_supervised = F.softmax(logits, dim=1)

            # forward pass for weakly augmented unconstrained data
            img_weak = batch['cm_train']['weakly_aug'].to(self.device)
            logits = self.model(img_weak)
            out_weak = F.softmax(logits, dim=1)

            # forward pass for strongly augmented unconstrained data
            img_strong = batch['cm_train']['strongly_aug'].to(self.device)
            logits = self.model(img_strong)
            # we use the softmax probs to create pseudo-constraints
            out_strong = F.softmax(logits, dim=1)

            return {'out': out_supervised, 'out_weak': out_weak, 'out_strong': out_strong}


    def loss_function(self, outs, batch, loss_type: str ='ccm_4', lambda_unsup: float = 0.1, threshold: float = 0.5, **kwargs):
        """Loss function functionality, a little over the top here
        """
        if len(batch) == 3:
            # catch the constrained only case 
            out = outs['out']
            train_target = batch['train_target'].to(self.device)
            prob1, prob2 = PairEnum(out)
            loss = self.criterion(prob1=prob1, prob2=prob2, simi=train_target)

            return {'loss': loss}
        else:
            out = outs['out']
            out_weak = outs['out_weak']
            out_strong = outs['out_strong']
            train_target = batch['supervised_train']['train_target']
           # 1) supervised loss
            prob1, prob2 = PairEnum(out)
            loss_sup = self.criterion(prob1=prob1, prob2=prob2, simi=train_target)

            ### CCM4: PCs (JS distance) from PLs (selected via informativeness)
            # Pseudo Label Selection: normalized entropy based
            # Pseudo Constraining: IJSD resulting in soft constraints
            if loss_type == 'ccm_4':

                # i) select informative pseudo labels pl that have low level of normalized entropy
                out_weak.detach_()
                # normalized entropy that we use for PL selection
                nentropy = self._normalized_entropy(y=out_weak)
                out_weak_informative = out_weak[nentropy < threshold]
                out_strong_informative = out_strong[nentropy < threshold]

                # ii) calculate pseudo constraints from them 
                # calculate soft PC labels via Inverse Jensen Shannon Distance 
                # we do not need/ use hard binary constraints for this approach
                ijs_targets = IJS2Simi(y=out_weak_informative.detach())

                # iii) track pl stats
                #FIXME: again catch for STL10 Large where we do not have ground truth for the unconstrained samples
                if -1 in batch['cm_train']['y']:
                    pl_conf_acc = 0     
                    pl_acc = 0
                else:
                    k = out_weak.shape[1]
                    confusion = Evaluator(k=k)
                    confusion.add(output=out_weak, target=batch['cm_train']['y'])
                    confusion.optimal_assignment(confusion.k)
                    pl_acc = confusion.acc()
                    if int(sum(nentropy < threshold)) > 1:
                        # cluster assignment for quality of CONFIDENT ONLY weak pseudo labels
                        confusion = Evaluator(k=k)
                        confusion.add(output=out_weak[nentropy < threshold], target=batch['cm_train']['y'][nentropy < threshold])
                        confusion.optimal_assignment(confusion.k)
                        pl_conf_acc = confusion.acc()
                    else: 
                        pl_conf_acc = 0     

                # 2) unsupervised loss via pseudo-constraints 
                prob1_saug, prob2_saug = PairEnum(out_strong_informative)
                # this is the MCL loss with IJS as soft targets
                loss_unsup = self.criterion(prob1=prob1_saug, prob2=prob2_saug, simi=ijs_targets, pseudo_targets=True)
                if loss_unsup.isnan(): loss_unsup = torch.zeros_like(loss_unsup)
                loss = loss_sup + lambda_unsup * loss_unsup

                return {'loss': loss,
                        'loss_sup': loss_sup,
                        'loss_unsup': loss_unsup * lambda_unsup, 
                        'pl_ratio': sum(nentropy < threshold) / len(nentropy),
                        'pl_acc': torch.tensor(pl_acc),
                        'pl_conf_acc': torch.tensor(pl_conf_acc),
                        'nent_mean': nentropy.mean().item(),
                        'nent_std': nentropy.std().item(),
                        'pl_soft_labels_mean': ijs_targets.mean().item(),
                        'pc_ml_ratio': sum(ijs_targets >= 0.5) / len(ijs_targets) if len(ijs_targets) > 0.0 else torch.tensor(0.0),
                        }
            elif loss_type.startswith('ccm_4_swap'):

                # i) select informative pseudo labels pl that have low level of normalized entropy
                out_weak.detach_()
                # normalized entropy that we use for PL selection

                # swap and confuse the predictions of nswap weakly augmented samples
                #FIXME: super cumbersome rn but works, improve later 
                # noise ratio (of swapped samples), per nswap sample we randomly change the position of two 
                # predicted cluster assignments (i.e. swap clusters) of the weakly annotated sample (serves as pseudo-label later!)
                noise_ratio = float(loss_type[-3:])
                nswap = int(noise_ratio*len(out_weak))
                swap_idxs = random.sample(population=range(0, len(out_weak)), k=nswap)
                for swap_idx in swap_idxs:
                    swap_i, swap_j = random.sample(population=range(0, out_weak.shape[1]), k=2)
                    new_i, new_j =  out_weak[swap_idx][swap_i].item(), out_weak[swap_idx][swap_j].item()
                    out_weak[swap_idx][swap_j], out_weak[swap_idx][swap_i] = new_i, new_j

                nentropy = self._normalized_entropy(y=out_weak)
                out_weak_informative = out_weak[nentropy < threshold]
                out_strong_informative = out_strong[nentropy < threshold]

                # ii) calculate pseudo constraints from them 
                # calculate soft PC labels via Inverse Jensen Shannon Distance 
                # we do not need/ use hard binary constraints for this approach
                # now including the swapping
                ijs_targets = IJS2Simi(y=out_weak_informative.detach())

                # iii) track pl stats
                #FIXME: again catch for STL10 Large where we do not have ground truth for the unconstrained samples
                if -1 in batch['cm_train']['y']:
                    pl_conf_acc = 0     
                    pl_acc = 0
                else:
                    k = out_weak.shape[1]
                    confusion = Evaluator(k=k)
                    confusion.add(output=out_weak, target=batch['cm_train']['y'])
                    confusion.optimal_assignment(confusion.k)
                    pl_acc = confusion.acc()
                    if int(sum(nentropy < threshold)) > 1:
                        # cluster assignment for quality of CONFIDENT ONLY weak pseudo labels
                        confusion = Evaluator(k=k)
                        confusion.add(output=out_weak[nentropy < threshold], target=batch['cm_train']['y'][nentropy < threshold])
                        confusion.optimal_assignment(confusion.k)
                        pl_conf_acc = confusion.acc()
                    else: 
                        pl_conf_acc = 0     

                # 2) unsupervised loss via pseudo-constraints 
                prob1_saug, prob2_saug = PairEnum(out_strong_informative)
                # this is the MCL loss with IJS as soft targets
                loss_unsup = self.criterion(prob1=prob1_saug, prob2=prob2_saug, simi=ijs_targets, pseudo_targets=True)
                if loss_unsup.isnan(): loss_unsup = torch.zeros_like(loss_unsup)
                loss = loss_sup + lambda_unsup * loss_unsup

                return {'loss': loss,
                        'loss_sup': loss_sup,
                        'loss_unsup': loss_unsup * lambda_unsup, 
                        'pl_ratio': sum(nentropy < threshold) / len(nentropy),
                        'pl_acc': torch.tensor(pl_acc),
                        'pl_conf_acc': torch.tensor(pl_conf_acc),
                        'nent_mean': nentropy.mean().item(),
                        'nent_std': nentropy.std().item(),
                        'pl_soft_labels_mean': ijs_targets.mean().item(),
                        'pc_ml_ratio': sum(ijs_targets >= 0.5) / len(ijs_targets) if len(ijs_targets) > 0.0 else torch.tensor(0.0),
                        }

            ### CCM5: PCs Threshold(JS distance) from PLs (selected via informativeness)
            # Pseudo Label Selection: normalized entropy based
            # Pseudo Constraining: IJSD which is then thresholded via cl_treshold to yield hard constraints
            elif loss_type == 'ccm_5':

                # i) select informative pseudo labels pl that have low level of normalized entropy
                out_weak.detach_()
                # normalized entropy that we use for PL selection
                nentropy = self._normalized_entropy(y=out_weak)
                out_weak_informative = out_weak[nentropy < threshold]
                out_strong_informative = out_strong[nentropy < threshold]

                # ii) calculate pseudo constraints from them 
                # calculate soft PC labels via Inverse Jensen Shannon Distance and then 
                # threshold them to mine hard {1, -1} constraints
                ijs_targets = IJS2Simi(y=out_weak_informative.detach())
                # transform into PCs and threshold to ML/CLs
                pcs = torch.ones_like(ijs_targets)
                # delete self-referential MLs, all other PC MLs are close to but never == 1.0
                pcs[ijs_targets == 1.0] = 0.0
                pcs[ijs_targets < kwargs['cl_threshold']] = -1.0

                # iii) track pl stats
                k = out_weak.shape[1]
                confusion = Evaluator(k=k)
                confusion.add(output=out_weak, target=batch['cm_train']['y'])
                confusion.optimal_assignment(confusion.k)
                pl_acc = confusion.acc()
                if int(sum(nentropy < threshold)) > 1:
                    # cluster assignment for quality of CONFIDENT ONLY weak pseudo labels
                    confusion = Evaluator(k=k)
                    confusion.add(output=out_weak[nentropy < threshold], target=batch['cm_train']['y'][nentropy < threshold])
                    confusion.optimal_assignment(confusion.k)
                    pl_conf_acc = confusion.acc()
                else: 
                    pl_conf_acc = 0     

                # 2) unsupervised loss via pseudo-constraints 
                prob1_saug, prob2_saug = PairEnum(out_strong_informative)
                # this is the MCL loss with IJS as soft targets
                loss_unsup = self.criterion(prob1=prob1_saug, prob2=prob2_saug, simi=pcs)
                if loss_unsup.isnan(): loss_unsup = torch.zeros_like(loss_unsup)
                loss = loss_sup + lambda_unsup * loss_unsup

                return {'loss': loss,
                        'loss_sup': loss_sup,
                        'loss_unsup': loss_unsup * lambda_unsup, 
                        'pl_ratio': sum(nentropy < threshold) / len(nentropy),
                        'pl_acc': torch.tensor(pl_acc),
                        'pl_conf_acc': torch.tensor(pl_conf_acc),
                        'nent_mean': nentropy.mean().item(),
                        'nent_std': nentropy.std().item(),
                        'pl_soft_labels_mean': ijs_targets.mean().item(),
                        'pc_ml_ratio': sum(pcs == 1.0) / len(pcs) if len(pcs) > 0.0 else torch.tensor(0.0),
                        }

    def training_step(self, batch, params, batch_idx=0,  step=0, **kwargs):
        """Single training step as called in the experiment template

        combines model.forward() and model.loss_function()

        Args:
            batch ([type]): [description]
            batch_idx ([type]): [description]
        """
        outs = self.forward(batch=batch)
        loss = self.loss_function(outs=outs, batch=batch, **params)
        return loss

    def evaluate(self, eval_dataloader: Any, confusion: Any=Evaluator(10), part: str='val', logger: Any=None, true_k: int=10):
        """Evaluate model on any dataloader during training and testing

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
        # re-sort predictions based on optimal assignment
        pred = pred[:, assigned_idx]
        calib_metrics = compute_calibration_metrics(true_labels=y, confidences=pred, num_bins=15)
        # calculate normalized mean predictive entropy 
        pred_nent = np.mean(np.sum(a=-pred * np.log(pred), axis=1) / np.log(pred.shape[1]))

        # calculate hacky mvalues from it 
        # modes (prob score > threshold) in prediction matrix / amount of samples
        mvalue_0p05 = len(pred[pred>0.05]) / len(pred)
        mvalue_0p1 = len(pred[pred>0.1]) / len(pred)
        mvalue_0p2 = len(pred[pred>0.2]) / len(pred)
        mvalue_0p3 = len(pred[pred>0.3]) / len(pred)

        confusion.optimal_assignment(confusion.k)
        print('\n')
        print(f"{part} ==> clustering scores: {confusion.clusterscores()}")
        print(f"{part} ==> accuracy: {confusion.acc()}")
        print(f"{part} ==> loss: {sum(losses)/len(losses)}")
        print(f"{part} ==> ece: {calib_metrics['expected_calibration_error']}")
        # store in dictionary for mlflow logging
        eval_results = {f'{part}_acc': confusion.acc()}
        eval_results.update({f'{part}_mvalue_0p3': mvalue_0p3})
        eval_results.update({f'{part}_mvalue_0p2': mvalue_0p2})
        eval_results.update({f'{part}_mvalue_0p1': mvalue_0p1})
        eval_results.update({f'{part}_mvalue_0p05': mvalue_0p05})
        eval_results.update({f'{part}_loss': sum(losses)/len(losses)})
        eval_results.update({f'{part}_{key}': value for key, value in confusion.clusterscores().items()})
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
            # log the test predictions
            final_results = {f'yhat_p_{cl}': pred[:, cl] for cl in range(pred.shape[1])}
            final_results['y'] = y
            final_results['yhat'] = np.argmax(pred, 1)
            final_results = pd.DataFrame(final_results)
            with tempfile.TemporaryDirectory() as tmp_dir:
                storage_path = os.path.join(tmp_dir, f'{part}_preds.csv')
                final_results.to_csv(storage_path)
                logger.experiment.log_artifact(local_path=storage_path, run_id=logger.run_id)

        return eval_results

    def pl_stats(self, cm_train_gen, threshold, logger, step):
        """
        """
        outs_weak = []
        outs_strong = []
        y = []
        for i, batch in enumerate(cm_train_gen):
            out_weak = F.softmax(self.model(batch['weakly_aug'].to(self.device)), dim=1)
            out_strong = F.softmax(self.model(batch['strongly_aug'].to(self.device)), dim=1)
            outs_weak.append(out_weak.detach().cpu().numpy())
            outs_strong.append(out_strong.detach().cpu().numpy())
            y.extend(batch['y'].detach().cpu().numpy())

        outs_weak = np.vstack(outs_weak)
        outs_strong = np.vstack(outs_strong)


        # 1.) average entropy
        entr_weak_mean = entr(outs_weak).sum(axis=1).mean()
        entr_strong_mean = entr(outs_strong).sum(axis=1).mean()
        logger.experiment.log_metric(key='weak_entropy',
                                     value=entr_weak_mean,
                                     step=step,
                                     run_id=logger.run_id)
        logger.experiment.log_metric(key='strong_entropy',
                                     value=entr_strong_mean,
                                     step=step,
                                     run_id=logger.run_id)

        # 2.) distribution of max softmax predictions (all obs)
        max_class_weak = np.argmax(outs_weak, axis=1)
        max_class_strong = np.argmax(outs_strong, axis=1)
        max_prob_weak = np.amax(outs_weak, axis=1)
        max_prob_strong = np.amax(outs_strong, axis=1)

        fig, axs = plt.subplots(2)
        fig.suptitle(f"distribution of max. softmax pred.: epoch {step}")
        axs[0].hist(max_prob_weak, bins=1000, density=True, histtype='stepfilled')
        axs[0].set_title("Weakly augmented samples")
        axs[1].hist(max_prob_strong, bins=1000, density=True, histtype='stepfilled')
        axs[1].set_title("Strongly augmented samples")
        fig.tight_layout()
        logger.experiment.log_figure(figure=fig,
                                     artifact_file=f'max_softmax_{step}.png',
                                     run_id=logger.run_id)

        # 3.) distribution of max softmax predictions (considered obs only => passed threshold
        counts_weak = np.unique(outs_weak.argmax(1), return_counts=True)
        counts_strong = np.unique(outs_strong.argmax(1), return_counts=True)
        y_all = np.unique(y, return_counts=True)

        fig, axs = plt.subplots(3)
        fig.suptitle(f"Predictive class distribution \nnon-annotated samples in epoch {step}")

        axs[0].bar(x=counts_weak[0], height=counts_weak[1])
        axs[0].set_xticks(counts_weak[0])
        axs[0].set_title("Weakly augmented samples")

        axs[1].bar(x=counts_strong[0], height=counts_strong[1])
        axs[1].set_xticks(counts_strong[0])
        axs[1].set_title("Strongly augmented samples")

        axs[2].bar(x=y_all[0], height=y_all[1])
        axs[2].set_xticks(y_all[0])
        axs[2].set_title("True labels")

        fig.tight_layout()

        # ax.set_xticks([np.arange(len(out_strong))])
        logger.experiment.log_figure(figure=fig,
                                     artifact_file=f'pred_dist_{step}.png',
                                     run_id=logger.run_id)

        # 4.) distribution of max softmax predictions (considered obs only => passed threshold)
        idx_selected = np.where((outs_weak > threshold).sum(1) == 1)[0]
        counts_weak = (outs_weak > threshold).sum(0)
        counts_strong = (outs_strong > threshold).sum(0)
        y_selected = [y[idx] for idx in idx_selected]
        y_dist = np.unique(y_selected, return_counts=True)
        y_height = [int(y_dist[1][y_dist[0] == idx]) if idx in y_dist[0] else 0 for idx in range(len(y_all[0]))]

        fig, axs = plt.subplots(3)
        fig.suptitle(f"Predictive class distribution \nnon-annotated samples (>=threshold) in epoch {step}")

        axs[0].bar(x=np.arange(counts_weak.shape[0]), height=counts_weak)
        axs[0].set_xticks(np.arange(outs_weak.shape[1]))
        axs[0].set_title(f"Weakly augmented samples, selected {counts_weak.sum()}")

        axs[1].bar(x=np.arange(counts_strong.shape[0]), height=counts_strong)
        axs[1].set_xticks(np.arange(outs_strong.shape[1]))
        axs[1].set_title("Strongly augmented samples")

        axs[2].bar(x=np.arange(len(y_height)), height=y_height)
        axs[2].set_xticks(np.arange(outs_strong.shape[1]))
        axs[2].set_title("True labels")

        fig.tight_layout()

        logger.experiment.log_figure(figure=fig,
                                     artifact_file=f'pred_dist_thresh_{step}.png',
                                     run_id=logger.run_id)

        logger.experiment.log_metric(key='n_pseudo_labels',
                                     value=counts_weak.sum(),
                                     step=step,
                                     run_id=logger.run_id)

    def _normalized_entropy(self, y: torch.Tensor):
        # entropy is upper bounded by log(|y|), hence we normalize to [0; 1] over **C**ardinality
        return -1 * y.mul(y.log()).sum(dim=1) / torch.log(torch.tensor(y.shape[1]))
