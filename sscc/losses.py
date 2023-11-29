# collection of different (pairwise) loss functions
import torch.nn as nn
import torch.nn.functional as F
import torch

from torch.cuda.amp import autocast

class KCL(nn.Module):
    # KLD-based Clustering Loss (KCL)
    def __init__(self, margin=2.0, **kwargs):
        super(KCL,self).__init__()
        self.kld = KLDiv()
        self.margin = margin

    def forward(self, prob1: torch.Tensor, prob2: torch.Tensor, simi: torch.Tensor, pseudo_targets: bool=False):
        """KL dif via hinge loss as in Hsu 2016

        Args:
            prob1 (torch.Tensor): prob vector 1
            prob2 (torch.Tensor): prob vector 1
            simi (torch.Tensor): binary constraint matrix
            pseud_targets(bool): use pseudo targets or not (default 0), important for ccm

        Returns:
            [type]: differentiable loss
        """        
        # kld calculated over all possible permutations
        # prob1.shape = prob2.shape = (batch_size**, num_classes)
        kld = self.kld(prob1,prob2)
        if not pseudo_targets: 
            # self-build hinge loss, same time + result as pt's F.hinge_embedding_loss
            # important to filter out 0s as we have sparse matrices in the CM case
            # # simi: 1->similar; -1->dissimilar; 0->unknown(ignore)
            kld_plus1 = kld[simi == 1]
            kld_minus1 = torch.clamp(input=self.margin-kld[simi == -1], min=0.0)
            return torch.mean(torch.cat((kld_plus1, kld_minus1)))
        elif pseudo_targets:
            # simi \in [0, 1], use as weights only
            kld_plus1 = kld * simi
            kld_minus1 = torch.clamp(input=self.margin-kld, min=0.0) * (1-simi)
            return torch.mean(kld_plus1 + kld_minus1)
            


class KLDiv(nn.Module):
    # Calculate KL-Divergence
    eps = 1e-35

    def forward(self, predict, target):
       assert predict.ndimension()==2,'Input dimension must be 2'
       target = target.detach()
       # KL(T||I) = \sum T(logT-logI)
       predict += eps
       target += eps
       logI = predict.log()
       logT = target.log()
       TlogTdI = target * (logT - logI)
       kld = TlogTdI.sum(1)
       return kld


class MCL(nn.Module):
    # Meta Classification Likelihood (MCL)

    eps = 1e-35 # Avoid calculating log(0). Use the small value of float16.

    def __init__(self, **kwargs):
        super(MCL, self).__init__()

    def forward(self, prob1: torch.Tensor, prob2: torch.Tensor, simi: torch.Tensor=None, soft: torch.Tensor=None):
        """MCL loss as in Hsu 2019
        as opposed to KCL also allows training with soft pairwise labels

        if simi given as target, works with {-1, 0, 1} hard constraints
        if soft given as target, works with {-1, 0, 1} soft constraints

        Args:
            prob1 (torch.Tensor): prob vector 1
            prob2 (torch.Tensor): prob vector 1
            simi (torch.Tensor): binary constraint matrix, 1->similar; -1->dissimilar; 0->unknown(ignore)
            soft (torch.Tensor, optional): soft labels matrix for the pseudo-constraints as used in ccm 3/4. Contains values in [0; 1] 
                with 0: Cannot Link, 1: Must Link. Defaults to None.

        Returns:
            [type]: differentiable loss
        """     
        P = prob1.mul(prob2)
        P = P.sum(1)
        P.mul_(simi).add_(simi.eq(-1).type_as(P))
        neglogP = -P.add_(MCL.eps).log_()
        return torch.mean(neglogP[simi != 0])


class MCLSoft(nn.Module):
    # Meta Classification Likelihood (MCL)
    eps = 1e-35 # Avoid calculating log(0). Use the small value of float16.

    def __init__(self, **kwargs):
        super(MCLSoft, self).__init__()
        
    def forward(self, prob1: torch.Tensor, prob2: torch.Tensor, simi: torch.Tensor=None):
        """MCL loss as in Hsu 2019
        as opposed to KCL also allows training with soft pairwise labels

        if simi given as target, works with {-1, 0, 1} hard constraints
        if soft given as target, works with {-1, 0, 1} soft constraints

        Args:
            prob1 (torch.Tensor): prob vector 1
            prob2 (torch.Tensor): prob vector 1
            simi (torch.Tensor): soft constraint matrix, 
                1->similar;
                -1->dissimilar;
                -0.90 -> 0.9 * CL + 0.1 * ML 
                0->unknown(ignore)

        Returns:
            [type]: differentiable loss
        """     
        Shat = prob1.mul_(prob2)
        Shat = Shat.sum(1)
        # use only non-zero constraints
        Shat_rel = Shat[simi != 0.0]
        simi_rel = simi[simi != 0.0]
        # Map CL constraints from -1 -> 0 (and soft constraints 0.9 -> 0.1)
        # as now 0 -> dissimilar, 1 -> similar
        simi_rel[simi_rel < 0.0] += 1
        l1 = Shat_rel.add_(MCLSoft.eps).log_().mul_(simi_rel)
        l2 = (1-Shat_rel).add_(MCLSoft.eps).log_().mul_(1-simi_rel)
        neglogP = -l1.add_(l2)

        return torch.mean(neglogP)


class FocalMCL(nn.Module):
    # Focal Version of the Meta Classification Likelihood (MCL)
    # essentially the Focalloss as MCL = BCE

    eps = 1e-35 # Avoid calculating log(0). Use the small value of float16.
    def __init__(self, alpha: float = 0.5, gamma: float = 0.0, **kwargs):
        super(FocalMCL, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, prob1: torch.Tensor, prob2: torch.Tensor, simi: torch.Tensor=None):
        """Focal Loss version of MCL loss as explained in https://amaarora.github.io/2020/06/29/FocalLoss.html

        Args:
            prob1 (torch.Tensor): prob vector 1
            prob2 (torch.Tensor): prob vector 1
            simi (torch.Tensor): binary constraint matrix, 1->similar; -1->dissimilar; 0->unknown(ignore)

        Returns:
            [type]: differentiable loss
        """     
        Shat = prob1.mul_(prob2)
        Shat = Shat.sum(1)

        # only non-zero constraints and map to [0,1]
        Shat = Shat[simi != 0]
        simi = simi[simi != 0]
        simi[simi == -1] = 0

        # ML 
        l1 = simi * self.alpha * (1-Shat)**self.gamma * Shat.add_(FocalMCL.eps).log_()
        # CL 
        l2 = (1-simi) * (1-self.alpha) * (Shat)**self.gamma * (1 - Shat).add_(FocalMCL.eps).log_()
        floss = -l1.add(l2)

        return torch.mean(floss)


class SoftFocalMCL(nn.Module):
    # Soft Focal Version of the Meta Classification Likelihood (MCL)
    # essentially the Focalloss as MCL = BCE 
    # also allowing soft constraints

    eps = 1e-35 # Avoid calculating log(0). Use the small value of float16.
    def __init__(self, alpha: float = 0.5, gamma: float = 0.0):
        super(SoftFocalMCL, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, prob1: torch.Tensor, prob2: torch.Tensor, simi: torch.Tensor=None):
        Shat = prob1.mul_(prob2)
        Shat = Shat.sum(1)
        # use only non-zero constraints
        Shat_rel = Shat[simi != 0.0]
        simi_rel = simi[simi != 0.0]
        # Map CL constraints from -1 -> 0 (and soft constraints 0.9 -> 0.1)
        # as now 0 -> dissimilar, 1 -> similar
        simi_rel[simi_rel < 0.0] += 1.0
        l1 = Shat_rel.add_(MCLSoft.eps).log_().mul_(simi_rel)
        l2 = (1-Shat_rel).add_(MCLSoft.eps).log_().mul_(1-simi_rel)
        neglogP = -l1.add_(l2)


        Shat = prob1.mul_(prob2)
        Shat = Shat.sum(1)

        # only non-zero constraints and map to [0,1]
        Shat = Shat[simi != 0]
        simi = simi[simi != 0]
        simi[simi < 0] += 1.0

        aa = Shat.add_(SoftFocalMCL.eps).log_().mul_(simi)

        # ML 
        l1 = self.alpha * (1-Shat)**self.gamma * Shat.add_(SoftFocalMCL.eps).log_().mul_(simi)
        # CL 
        l2 = (1-self.alpha) * (Shat)**self.gamma * (1 - Shat).add_(SoftFocalMCL.eps).log_().mul_(1-simi)
        floss = -l1.add(l2)

        return torch.mean(floss)


class MBCE(nn.Module):
    # Meta Classification Likelihood (MCL) via Binary XE
    # also handles soft CIs

    def __init__(self, **kwargs):
        super(MBCE, self).__init__()

    def forward(self, prob1: torch.Tensor, prob2: torch.Tensor, simi: torch.Tensor=None, pseudo_targets: bool=False):   
        Shat = prob1.mul(prob2)
        Shat = Shat.sum(1)

        if not pseudo_targets: 
            # use only non-zero constraints and map to [0, 1]
            # pseudo targets already fit [0, 1] mapping
            Shat = Shat[simi != 0.0]
            simi = simi[simi != 0.0].float()
            simi[simi<0.0] += 1.0
        # this is required to circumvent numerical instability via amp + log for the Shat calc
        # https://discuss.pytorch.org/t/torch-nn-bceloss-are-unsafe-to-autocast-while-working-with-cosine-similarity/117582
        with autocast(enabled=False):
            bce = F.binary_cross_entropy(input=Shat, target=simi, reduction='none')
        return torch.mean(bce)


class MBCEnt(nn.Module):
    # Meta Classification Likelihood (MCL) via Binary XE
    # includes Entropy Regularization
    # also handles soft CIs

    def __init__(self, lambda_ent: float = 0.0, **kwargs):
        super(MBCEnt, self).__init__()
        self.lambda_ent = lambda_ent

    def forward(self, prob1: torch.Tensor, prob2: torch.Tensor, simi: torch.Tensor=None, pseudo_targets: bool=False):   
        Shat = prob1.mul(prob2)
        Shat = Shat.sum(1)

        if not pseudo_targets: 
            # use only non-zero constraints and map to [0, 1]
            # pseudo targets already fit [0, 1] mapping
            Shat = Shat[simi != 0.0]
            prob1 = prob1[simi != 0.0]
            prob2 = prob2[simi != 0.0]
            simi = simi[simi != 0.0].float()
            simi[simi<0.0] += 1.0
        # this is required to circumvent numerical instability via amp + log for the Shat calc
        # https://discuss.pytorch.org/t/torch-nn-bceloss-are-unsafe-to-autocast-while-working-with-cosine-similarity/117582
        with autocast(enabled=False):
            bce = F.binary_cross_entropy(input=Shat, target=simi, reduction='none')
        if self.lambda_ent != 0.0:
            # use normalized entropy for regularization as this thing grows with log(C)
            # nope, no normalization!
            nent1 = -prob1.mul(prob1.log()).sum(1) / torch.log(torch.tensor(prob1.shape[1]))
            nent2 = -prob2.mul(prob2.log()).sum(1) / torch.log(torch.tensor(prob2.shape[1]))
            ent_reg = nent1.add(nent2).div(2)
            # and clamp to min 0
            return torch.mean(torch.clamp(bce - self.lambda_ent * ent_reg, min=0.0))
        else:
            return torch.mean(bce)


class FocalMBCE(nn.Module):
    # Focal Version of the Meta Classification Likelihood (MCL)
    # essentially the Focalloss as MCL = BCE
    # alpha: weight param for CL

    eps = 1e-35 # Avoid calculating log(0). Use the small value of float16.
    def __init__(self, alpha: float = 0.5, gamma: float = 0.0, **kwargs):
        super(FocalMBCE, self).__init__()
        self.alpha = torch.tensor([alpha, 1-alpha]).cuda()
        self.gamma = gamma

    def forward(self, prob1: torch.Tensor, prob2: torch.Tensor, simi: torch.Tensor=None, pseudo_targets: bool=False):
        """Focal Loss version of MCL loss as explained in https://amaarora.github.io/2020/06/29/FocalLoss.html

        Args:
            prob1 (torch.Tensor): prob vector 1
            prob2 (torch.Tensor): prob vector 1
            simi (torch.Tensor): binary constraint matrix, 1->similar; -1->dissimilar; 0->unknown(ignore)

        Returns:
            [type]: differentiable loss
        """     
        Shat = prob1.mul_(prob2)
        Shat = Shat.sum(1)
        if not pseudo_targets: 
            # use only non-zero constraints and map to [0, 1]
            # pseudo targets already fit [0, 1] mapping
            Shat = Shat[simi != 0.0]
            simi = simi[simi != 0.0].float()
            simi[simi<0.0] += 1.0

        with autocast(enabled=False):
            bce = F.binary_cross_entropy(input=Shat, target=simi, reduction='none')
        # still weighting ML vs. CL 
        simi = simi.type(torch.long)
        at = self.alpha.gather(0, simi.data.view(-1))
        # wild trick for pt={...
        pt = torch.exp(-bce)
        focalloss = at*(1-pt)**self.gamma * bce
        return torch.mean(focalloss)

class CE(nn.Module):
    # CE for supervised baselines
    def __init__(self, **kwargs):
        super(CE, self).__init__()

    def forward(self, yhat: torch.Tensor, y: torch.Tensor):
        with autocast(enabled=False): ce = F.cross_entropy(input=yhat, target=y, reduction='none')
        return torch.mean(ce)

loss_dict = {
    'KCL': KCL, 
    'MCL': MCL, 
    'MCLSoft': MCLSoft,
    'FocalMCL': FocalMCL,
    'SoftFocalMCL': SoftFocalMCL, 
    'MBCE': MBCE, 
    'FocalMBCE': FocalMBCE,
    'MBCEnt': MBCEnt,
    'CE': CE
}