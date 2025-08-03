import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule


class bbaOp(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.conv1 = nn.Conv2d(in_features, in_features, kernel_size=3, padding=1, groups=in_features)
        self.conv2 = nn.Conv2d(in_features, in_features, kernel_size=5, padding=2, groups=in_features)
        self.conv3 = nn.Conv2d(in_features, in_features, kernel_size=7, padding=3, groups=in_features)

        self.projector = nn.Conv2d(in_features, in_features, kernel_size=1)

    def forward(self, x):
        identity = x
        conv1_x = self.conv1(x)
        conv2_x = self.conv2(x)
        conv3_x = self.conv3(x)

        x = (conv1_x + conv2_x + conv3_x) / 3.0 + identity
        identity = x
        x = self.projector(x)
        return identity + x


class bba(BaseModule):
    def __init__(self, in_dim, factor=4, memory_slots=10):
        super().__init__()

        self.project1 = nn.Linear(in_dim, 64)
        self.nonlinear = F.gelu
        self.project2 = nn.Linear(64, in_dim)
        self.dropout = nn.Dropout(p=0.1)

        self.adapter_conv = bbaOp(64)

        self.norm = nn.LayerNorm(in_dim)
        self.gamma = nn.Parameter(torch.ones(in_dim) * 1e-6)
        self.gammax = nn.Parameter(torch.ones(in_dim))

        # === Memory Bank ===
        self.memory_slots = memory_slots
        self.register_buffer("memory_bank", torch.zeros(memory_slots, 64))  # (S, C)
        self.memory_ptr = 0
        self.memory_filled = 0

    def forward(self, x, hw_shapes=None):
        identity = x
        x = self.norm(x) * self.gamma + x * self.gammax

        project1 = self.project1(x)  # (B, N, 64)
        b, n, c = project1.shape
        h, w = hw_shapes

        # Reshape to image format for conv
        project1_img = project1.reshape(b, h, w, c).permute(0, 3, 1, 2)  # (B, C, H, W)
        project1_img = self.adapter_conv(project1_img)
        project1 = project1_img.permute(0, 2, 3, 1).reshape(b, n, c)  # (B, N, C)

        # === Memory Fusion ===
        if self.memory_filled > 0:
            memory_avg = self.memory_bank[:self.memory_filled].mean(dim=0, keepdim=True)  # (1, C)
            project1 = project1 + memory_avg.unsqueeze(0)  # broadcast to (B, N, C)

        # === Update Memory Bank ===
        with torch.no_grad():
            batch_mean = project1.mean(dim=1)  # (B, C)
            for vec in batch_mean:
                self.memory_bank[self.memory_ptr] = vec
                self.memory_ptr = (self.memory_ptr + 1) % self.memory_slots
                self.memory_filled = min(self.memory_filled + 1, self.memory_slots)

        nonlinear = self.nonlinear(project1)
        nonlinear = self.dropout(nonlinear)
        project2 = self.project2(nonlinear)

        return identity + project2
