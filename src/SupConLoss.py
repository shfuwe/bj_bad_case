import torch
import torch.nn as nn
import math
class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, device0,features,labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device=device0

        batch_size = features.shape[0]
        
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, labels.T).float().to(device)
        
        # compute mask_contrast
        mask_contrast=1-mask

        # compute mask_same
        diag = torch.diag(mask)
        a_diag = torch.diag_embed(diag)
        mask_same= mask - a_diag

        
        # compute logits
        logits = torch.div(torch.matmul(features, features.T),0.3)
#         diagl = torch.diag(logits)
#         logits=logits-diagl
        
        exp_logits=torch.exp(logits)
#         b = torch.zeros(batch_size, batch_size).to(device)
#         exp_logits=torch.where(exp_logits !=1, exp_logits, b)
        
    
        # compute logits_contrast_sum_exp logits_same mask_same_sum
        logits_contrast_exp = exp_logits * mask_contrast
        logits_contrast_sum_exp=logits_contrast_exp.sum(1, keepdim=True)

        logits_same=logits * mask_same
        
        mask_same_sum=mask_same.sum(1,keepdim=True)
        
        # compute mean_log_prob_pos
        mean_log_prob_pos=(logits_same-torch.log(logits_contrast_sum_exp)).sum(1,keepdim=True)/mask_same_sum#某类只有一个mask_same_sum为0
        
        # compute loss
#         loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss=-mean_log_prob_pos
        # zero = torch.zeros_like(loss)
#         print(loss)
#         loss=torch.where(loss > -99999, loss, zero)
#         loss=torch.where(loss < 99999, loss, zero)
#         print(loss)
        return loss.mean()
        
#         sum_=torch.tensor([0.0]).to(device)
#         for i in loss:
#             if not math.isinf(i) and not math.isnan(i):
#                 sum_+=i
#         return sum_
