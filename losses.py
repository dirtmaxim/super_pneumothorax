import torch
import torch.nn as nn
from torch.nn import functional as F

class DiceCoef(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self,input, target):
        input_sigm = torch.sigmoid(input)
        iflat = input_sigm.view(-1)
        tflat = target.view(-1)
        intersection = (iflat * tflat).sum()
        return ((2.0 * intersection + self.smooth) / (iflat.sum() + tflat.sum() + self.smooth))


class FocalLoss(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma

    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), input.size()))
        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + \
            ((-max_val).exp() + (-input - max_val).exp()).log()
        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        return loss.mean()


class FocalDiceLoss(nn.Module):
    def __init__(self, alpha, gamma, smooth):
        super().__init__()
        self.alpha = alpha
        self.focal = FocalLoss(gamma)
        self.dice_coef = DiceCoef(smooth)
        
    def forward(self, input, target):
        loss = self.alpha*self.focal(input, target) - torch.log(self.dice_coef(input, target))
        return loss.mean()
    
class BCEDiceLoss(nn.Module):
    def __init__(self, alpha, smooth):
        super().__init__()
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss()
        self.dice_coef = DiceCoef(smooth)
        
    def forward(self, input, target):
        loss = self.alpha*self.bce(input, target) - torch.log(self.dice_coef(input, target))
        return loss.mean()