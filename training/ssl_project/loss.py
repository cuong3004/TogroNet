import torch
import torch.nn as nn


class BarlowTwinsLoss(nn.Module):
    def __init__(self, lambda_coeff=5e-3):
        super().__init__()
        self.lambda_coeff = lambda_coeff

    def off_diagonal(self, x):
        n, m = x.shape
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, z1, z2):
        eps = 1e-9
        batch_size = z1.size(0)

        z1 = (z1 - z1.mean(0)) / (z1.std(0, unbiased=False) + eps)
        z2 = (z2 - z2.mean(0)) / (z2.std(0, unbiased=False) + eps)

        c = torch.matmul(z1.T, z2) / batch_size

        on_diag = torch.diagonal(c).add(-1).pow(2).sum()
        off_diag = self.off_diagonal(c).pow(2).sum()

        return on_diag + self.lambda_coeff * off_diag