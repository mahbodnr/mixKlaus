import torch
import torch.nn as nn
import torch.nn.functional as F

class PositiveConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        assert kwargs["bias"] is False, "Bias is not supported"
        super().__init__(*args, **kwargs)

    def forward(self, x):
        positive_weights = torch.exp(self.weight)
        return F.conv2d(x, positive_weights, padding=self.padding, bias=None, stride=self.stride)

    def get_positive_weights(self):
        positive_weights = torch.exp(self.weight)
        positive_weights = F.normalize(positive_weights, p=1, dim=1)
        return positive_weights

class PowerSoftmax(nn.Module):
    def __init__(self, power, dim):
        super().__init__()
        self.power = power
        self.dim = dim

    def forward(self, x):
        if self.power == 1:
            return F.normalize(x, p=1, dim=self.dim)
        power_x = torch.pow(x, self.power)
        return power_x / torch.sum(power_x, dim=self.dim, keepdim=True)