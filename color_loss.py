import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.stats as st


def gauss_kernel(kernlen=21, nsig=3, channels=1):
    interval = (2 * nsig + 1.) / (kernlen)
    x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kernlen + 1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw / kernel_raw.sum()
    out_filter = np.array(kernel, dtype=np.float32)
    out_filter = out_filter.reshape((kernlen, kernlen, 1, 1))
    out_filter = np.repeat(out_filter, channels, axis=2)
    out_filter = np.repeat(out_filter, channels, axis=3)
    return out_filter


class Blur(nn.Module):
    def __init__(self, nc):
        super(Blur, self).__init__()
        kernel = gauss_kernel(21, 3, nc)
        kernel = torch.from_numpy(kernel).permute(3, 2, 0, 1)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, x):
        x = F.conv2d(x, self.weight, stride=1, padding=10)
        return x


class ColorLoss(nn.Module):
    def __init__(self):
        super(ColorLoss, self).__init__()

    def forward(self, x1, x2):
        return torch.nn.MSELoss(x1, x2)

