import torch
from torch.nn import functional as F

def context_loss(output, obstacle, value, difficulty, cfg):
    return F.binary_cross_entropy_with_logits(output['obstacle_logits'], obstacle) + cfg.lambda_aux * F.mse_loss(output['value'], value) + F.cross_entropy(output['difficulty_logits'], difficulty)

def selection_loss(scores, targets, groups):
    device = scores['mode_heads'].device
    mt = torch.as_tensor(targets['mode'], device=device)
    lt = torch.as_tensor(targets['local'], device=device)
    m, p = (targets['positive_mode'], targets['positive_path'])
    group = groups[m]
    loss_mode = F.cross_entropy(scores['mode_scores'][None], torch.tensor([m], device=device))
    loss_rank = F.cross_entropy(scores['local_scores'][group][None], torch.tensor([group.index(p)], device=device))
    mh, lh = (scores['mode_heads'], scores['local_heads'])
    gain = (mh[:, 0] - mt[:, 0]).square().sum() + (lh[p, 0] - lt[p, 0]).square()
    risk = (mh[:, 1] - mt[:, 1]).square().sum() + (lh[p, 1] - lt[p, 1]).square()
    cost = (mh[:, 3] - mt[:, 3]).square().sum() + (lh[p, 2] - lt[p, 2]).square()
    cont = (lh[p, 3] - lt[p, 3]).square()
    unc = (mh[:, 2] - mt[:, 2]).square().sum()
    return loss_mode + loss_rank + gain + risk + cost + cont + unc
