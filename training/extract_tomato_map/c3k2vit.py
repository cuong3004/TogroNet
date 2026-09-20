"""C3k2ViT — khoi CSP hai nhanh (kieu C3k2 cua YOLO26) trong do nhanh xu ly sau
Bottleneck duoc noi them mot khoi attention toan cuc, chi phi tuyen tinh theo
O(N) (lay cam hung tu MobileViTv2), thay cho self-attention da dau O(N^2) chuan
trong PSABlock/C2PSA cua Ultralytics. Muc tieu: giu kha nang mo hinh hoa quan he
toan cuc cua attention nhung giam chi phi tinh toan de phu hop trien khai tren
thiet bi bien (Raspberry Pi).

Phu thuoc (khoi co ban, khong liet lai o day):
    - Conv        : ultralytics.nn.modules.conv.Conv        (Conv2d + BN + activation)
    - Bottleneck  : ultralytics.nn.modules.block.Bottleneck  (cap conv residual chuan)
"""

import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import Bottleneck


class LinearViTAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, attn_ratio: float = 0.5):
        super().__init__()
        self.W_i = nn.Conv2d(dim, 1, 1)     
        self.W_k = nn.Conv2d(dim, dim, 1)   
        self.W_v = nn.Conv2d(dim, dim, 1)   
        self.proj = Conv(dim, dim, 1, act=False)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        cs = self.W_i(x).view(B, 1, -1)
        cs = torch.softmax(cs, dim=-1)  

        k = self.W_k(x).view(B, C, -1)
        v = self.act(self.W_v(x)).view(B, C, -1)

        context = (k * cs).sum(dim=-1, keepdim=True)  
        out = (v * context).view(B, C, H, W)

        return self.proj(out)


class ViTBlock(nn.Module):
    def __init__(self, c: int, attn_ratio: float = 0.5, num_heads: int = 4, shortcut: bool = True):
        super().__init__()
        self.attn = LinearViTAttention(c, num_heads=num_heads, attn_ratio=attn_ratio)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class C3k2ViT(nn.Module):
    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        e: float = 0.5,
        g: int = 1,
        shortcut: bool = True,
        attn_ratio: float = 0.5,
    ):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            nn.Sequential(
                Bottleneck(self.c, self.c, shortcut, g),
                ViTBlock(self.c, attn_ratio=attn_ratio, num_heads=max(self.c // 64, 1), shortcut=shortcut),
            )
            for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


if __name__ == "__main__":
    block = C3k2ViT(c1=256, c2=256, n=2)
    x = torch.randn(1, 256, 40, 40)
    y = block(x)
    n_params = sum(p.numel() for p in block.parameters())
    print(f"input : {tuple(x.shape)}")
    print(f"output: {tuple(y.shape)}")
    print(f"params: {n_params:,}")
