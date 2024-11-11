import torch.nn as nn


class Add(nn.Identity):
    def forward(self, x1, x2):
        return x1 + x2
