import torch.nn as nn
import torch.nn.functional as F


class AvgPool2d(nn.Identity):
    def forward(self, x):
        return F.avg_pool2d(x, x.size()[3])


class Flatten(nn.Identity):
    def forward(self, x):
        return x.view(x.size(0), -1)
