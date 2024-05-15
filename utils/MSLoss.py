import torch
import torch.nn as nn
import torch.nn.functional as F


class MSLoss(nn.Module):
    def __init__(self, temperature=0.3, totalepoch=100, self_paced=True):
        super(MSLoss, self).__init__()
        self.temperature = temperature
        self.totalepoch = totalepoch
        self.self_paced = self_paced

    def forward(self, image_feature, text_feature, labels=None, epoch=0):

        mask = (torch.mm(labels.float(), labels.float().T) > 0).float()
        pos_mask = mask
        neg_mask = 1 - mask

        image_dot_text = torch.matmul(F.normalize(image_feature, dim=1), F.normalize(text_feature, dim=1).T)

        all_exp = torch.exp(image_dot_text / self.temperature)
        pos_exp = pos_mask * all_exp
        neg_exp = neg_mask * all_exp

        if self.self_paced:
            if epoch <= int(self.totalepoch/3):
                delta = epoch / int(self.totalepoch/3)
            else:
                delta = 1
            pos_exp *= torch.exp(-1 - image_dot_text).detach() ** (delta/4)
            neg_exp *= torch.exp(-1 + image_dot_text).detach() ** (delta)

        loss = -torch.log(pos_exp.sum(1)/(neg_exp.sum(1) + pos_exp.sum(1)))
        return loss.mean()