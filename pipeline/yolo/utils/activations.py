# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Activation functions
"""

import torch
import torch.nn as nn


# SiLU https://arxiv.org/pdf/1606.08415.pdf ----------------------------------------------------------------------------
class SiLU(nn.Module):  # export-friendly version of nn.SiLU()
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)