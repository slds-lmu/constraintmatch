from __future__ import print_function
import torch
import seaborn as sn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import linear_sum_assignment as hungarian
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score, adjusted_mutual_info_score

class Evaluator(object):
    """
    column of confusion matrix: predicted index
    row of confusion matrix: target index
    """
    def __init__(self, k, normalized = False):
        super(Evaluator, self).__init__()
        self.k = k
        self.conf = torch.LongTensor(k,k)
        self.normalized = normalized
        self.match_dict = {}
        self.reset()

    def reset(self):
        self.conf.fill_(0)
        self.gt_n_cluster = None
        self.match_dict = {}

    def cuda(self):
        self.conf = self.conf.cuda()

    def add(self, output, target):
        output = output.squeeze()
        target = target.squeeze()
        assert output.size(0) == target.size(0), \
                'number of targets and outputs do not match'
        if output.ndimension()>1: #it is the raw probabilities over classes
            assert output.size(1) == self.conf.size(0), \
                'number of outputs does not match size of confusion matrix'

            _,pred = output.max(1) #find the predicted class
        else: #it is already the predicted class
            pred = output
        indices = (target*self.conf.stride(0) + pred.squeeze_().type_as(target)).type_as(self.conf)
        ones = torch.ones(1).type_as(self.conf).expand(indices.size(0))
        self._conf_flat = self.conf.view(-1)
        self._conf_flat.index_add_(0, indices, ones)

    def classIoU(self,ignore_last=False):
        confusion_tensor = self.conf
        if ignore_last:
            confusion_tensor = self.conf.narrow(0,0,self.k-1).narrow(1,0,self.k-1)
        union = confusion_tensor.sum(0).view(-1) + confusion_tensor.sum(1).view(-1) - confusion_tensor.diag().view(-1)
        acc = confusion_tensor.diag().float().view(-1).div(union.float()+1)
        return acc

    def recall(self,clsId):
        i = clsId
        TP = self.conf[i,i].sum().item()
        TPuFN = self.conf[i,:].sum().item()
        if TPuFN==0:
            return 0
        return float(TP)/TPuFN

    def precision(self,clsId):
        i = clsId
        TP = self.conf[i,i].sum().item()
        TPuFP = self.conf[:,i].sum().item()
        if TPuFP==0:
            return 0
        return float(TP)/TPuFP

    def f1score(self,clsId):
        r = self.recall(clsId)
        p = self.precision(clsId)
        if (p+r)==0:
            return 0
        return 2*float(p*r)/(p+r)

    def acc(self):
        TP = self.conf.diag().sum().item()
        total = self.conf.sum().item()
        if total==0:
            return 0
        return float(TP)/total

    def optimal_assignment(self,gt_n_cluster=None,assign=None):
        if assign is None:
            mat = -self.conf.cpu().numpy() #hungarian finds the minimum cost
            orig_idx, assigned_idx = hungarian(mat)
        self.conf = self.conf[:,assigned_idx]
        self.gt_n_cluster = gt_n_cluster
        # store the matching between original and optimal cluster assignments
        for oo, aa in zip(orig_idx, assigned_idx): self.match_dict[oo] = aa
        return orig_idx, assigned_idx

    def show(self,width=6,row_labels=None,column_labels=None):
        print("Confusion Matrix:")
        conf = self.conf
        rows = self.gt_n_cluster or conf.size(0)
        cols = conf.size(1)
        if column_labels is not None:
            print(("%" + str(width) + "s") % '', end='')
            for c in column_labels:
                print(("%" + str(width) + "s") % c, end='')
            print('')
        for i in range(0,rows):
            if row_labels is not None:
                print(("%" + str(width) + "s|") % row_labels[i], end='')
            for j in range(0,cols):
                print(("%"+str(width)+".d")%conf[i,j],end='')
            print('')

    def conf2label(self):
        conf=self.conf
        gt_classes_count=conf.sum(1).squeeze()
        n_sample = gt_classes_count.sum().item()
        gt_label = torch.zeros(n_sample)
        pred_label = torch.zeros(n_sample)
        cur_idx = 0
        for c in range(conf.size(0)):
            if gt_classes_count[c]>0:
                gt_label[cur_idx:cur_idx+gt_classes_count[c]].fill_(c)
            for p in range(conf.size(1)):
                if conf[c][p]>0:
                    pred_label[cur_idx:cur_idx+conf[c][p]].fill_(p)
                cur_idx = cur_idx + conf[c][p];
        return gt_label,pred_label

    def clusterscores(self):
        target,pred = self.conf2label()
        NMI = normalized_mutual_info_score(target,pred)
        ARI = adjusted_rand_score(target,pred)
        AMI = adjusted_mutual_info_score(target,pred)
        return {'NMI':NMI, 'ARI':ARI, 'AMI':AMI}

    def plot_confmat(self, title: str='', true_k: int=10):
        """Seaborn heatmap for confusion matrix, then propagated to the 
        mlflow logging func as plt object
        """
        df_cm = pd.DataFrame(np.asarray(self.conf),
                             index = [int(i) for i in range(self.k)],
                             columns = [int(j) for j in range(self.k)])
        # only plot realistic true classes
        df_cm = df_cm.iloc[:true_k, :]
        plt.close()
        fig, axs = plt.subplots(1)
        fig.suptitle(title)
        fig.set_size_inches(1 + int(7 * self.k / 10), int(7 * true_k / 10))
        sn.set(font_scale=1.0)
        # add original indices to the optimally matched ones on the yaxis
        mm = self.match_dict
        yticklabels = [f'{optimal} ({mm[optimal]})' for optimal in df_cm.index.to_numpy()]
        heat = sn.heatmap(df_cm, annot=True, annot_kws={"size": 10}, 
                          fmt='d', cmap="YlGnBu",
                          ax=axs, yticklabels=yticklabels)
        axs.set_xlabel('optimal prediction assignment')
        axs.set_ylabel('true optimal (original) cluster assignment')
        plt.tight_layout()
        return fig

def compute_calibration_metrics(true_labels, confidences, num_bins=30):
    """
    Implementation of calibration metrics based on binning.
    This

    Code taken from
    https://github.com/hollance/reliability-diagrams
    """
    pred_labels = confidences.argmax(axis=1)
    assert(len(confidences) == len(pred_labels))
    assert(len(confidences) == len(true_labels))
    assert(num_bins > 0)

    pred_confidences = confidences.max(axis=1)

    pred_labels = np.array(pred_labels, np.float)
    true_labels = np.array(true_labels, np.float)
    pred_confidences = np.array(pred_confidences, np.float)

    bins = np.linspace(0.0, 1.0, num_bins+1)
    indices = np.digitize(pred_confidences, bins, right=True)
    bin_accuracies = np.zeros(num_bins, dtype=np.float)
    bin_confidences = np.zeros(num_bins, dtype=np.float)
    bin_counts = np.zeros(num_bins, dtype=np.float)

    for b in range(num_bins):
        selected = np.where(indices == b + 1)[0]
        if len(selected) > 0:
            bin_accuracies[b] = np.mean(true_labels[selected] == pred_labels[selected])
            bin_confidences[b] = np.mean(pred_confidences[selected])
            bin_counts[b] = len(selected)

    avg_acc = np.sum(bin_accuracies * bin_counts) / np.sum(bin_counts)
    avg_conf = np.sum(bin_confidences * bin_counts) / np.sum(bin_counts)

    gaps = np.abs(bin_accuracies - bin_confidences)
    ece = np.sum(gaps * bin_counts) / np.sum(bin_counts)
    mce = np.max(gaps)
    oe = np.sum(np.maximum(bin_confidences - bin_accuracies, np.zeros_like(gaps)) * bin_counts) / np.sum(bin_counts)
    return {
        "accuracies": bin_accuracies,
        "confidences": bin_confidences,
        "counts": bin_counts,
        "bins": bins,
        "avg_accuracy": avg_acc,
        "avg_confidence": avg_conf,
        "expected_calibration_error": ece,
        "max_calibration_error": mce,
        "overconfidence_error": oe,
    }
