import math
import torch
import torch.nn.functional as F

class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # Tensor [C] or float or None
        self.reduction = reduction
    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        loss = (1 - pt) ** self.gamma * ce
        if self.alpha is not None:
            if isinstance(self.alpha, torch.Tensor):
                alpha_t = self.alpha.to(logits.device)[targets]
            else:
                alpha_t = self.alpha
            loss = alpha_t * loss
        return loss.mean() if self.reduction == "mean" else loss.sum()

class AsymmetricLoss(torch.nn.Module):
    # ASL for single-label (Ridnik et al.)
    def __init__(self, gamma_pos=0.0, gamma_neg=4.0, eps=1e-8):
        super().__init__()
        self.gp = gamma_pos
        self.gn = gamma_neg
        self.eps = eps
    def forward(self, logits, targets):
        # one-hot
        num_classes = logits.size(1)
        y = F.one_hot(targets, num_classes=num_classes).float()
        x = torch.sigmoid(logits)
        xs_pos = x
        xs_neg = 1.0 - x
        # probs
        p_t = y * xs_pos + (1 - y) * xs_neg
        # asymmetric focusing
        w = (1 - p_t) ** (self.gp * y + self.gn * (1 - y))
        # BCE
        loss = - (y * torch.log(xs_pos.clamp(min=self.eps)) +
                  (1 - y) * torch.log(xs_neg.clamp(min=self.eps)))
        loss = (w * loss).sum(dim=1)
        return loss.mean()

class BalancedSoftmaxLoss(torch.nn.Module):
    # BSCE: logit에 log(prior) 더해 CE 계산
    def __init__(self, cls_counts: torch.Tensor):
        super().__init__()
        prior = cls_counts.float().clamp(min=1)
        self.log_prior = torch.log(prior / prior.sum())
    def forward(self, logits, targets):
        logits_adj = logits + self.log_prior.to(logits.device)
        return F.cross_entropy(logits_adj, targets)

class LDAMLoss(torch.nn.Module):
    # from "Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss"
    def __init__(self, cls_counts: torch.Tensor, max_m=0.5, s=30):
        super().__init__()
        m_list = 1.0 / torch.sqrt(torch.sqrt(cls_counts.float().clamp(min=1)))
        m_list = m_list * (max_m / m_list.max())
        self.m_list = m_list
        self.s = s
    def forward(self, logits, targets):
        # subtract margin on true class logit
        idx = torch.arange(logits.size(0), device=logits.device)
        logits_m = logits.clone()
        margins = self.m_list.to(logits.device)[targets]
        logits_m[idx, targets] = logits_m[idx, targets] - margins
        return F.cross_entropy(self.s * logits_m, targets)