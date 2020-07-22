import torch
import torch.nn as nn
from torch.nn import functional as F

class DiceCompCoef(nn.Module):
    def __init__(self, threshold, min_pixels = 10):
        super().__init__()
        self.threshold = threshold
        self.min_pixels = min_pixels
    
    def forward(self,input, target):
        input_sigm = torch.sigmoid(input)
        input_sigm = (input_sigm >= self.threshold).type(torch.float32)
        iflat = input_sigm.view(-1)
        tflat = target.view(-1)
        if (iflat.sum() < self.min_pixels) and (tflat.sum() == 0):
            return 1.
        intersection = (iflat * tflat).sum()
        return ((2.0 * intersection) / (iflat.sum() + tflat.sum()))