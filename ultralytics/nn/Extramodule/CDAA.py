import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.block import Conv


class CDA_attention(nn.Module):
    """
    Cell-Density-Adaptive Attention
    1. 用梯度能量图近似细胞密度（无标注）
    2. 密度图指导注意力温度缩放 → 高密度区更尖锐，低密度更平滑
    3. 全程可微，端到端训练.
    """

    def __init__(self, channels, kernel=7):
        super().__init__()
        self.kernel = kernel
        self.norm = nn.LayerNorm(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.out = nn.Conv2d(channels, channels, 1)
        # 密度估计：拉普拉斯能量 + 1×1 映射
        self.density_mlp = nn.Sequential(nn.Conv2d(1, 8, 3, 1, 1), nn.ReLU(), nn.Conv2d(8, 1, 3, 1, 1), nn.Sigmoid())

    def forward(self, x):
        B, C, H, W = x.shape
        # 1. 快速密度图：梯度能量
        gray = x.mean(dim=1, keepdim=True)
        lap = F.conv2d(
            gray, torch.tensor([[[[0, -1, 0], [-1, 4, -1], [0, -1, 0]]]], device=x.device, dtype=x.dtype), padding=1
        )
        density = self.density_mlp(lap.abs())  # [B,1,H,W]

        # 2. 标准 self-attention
        q, k, v = self.qkv(x).chunk(3, dim=1)
        q = q.view(B, C, -1).permute(0, 2, 1)  # B,HW,C
        k = k.view(B, C, -1)  # B,C,HW
        v = v.view(B, C, -1).permute(0, 2, 1)  # B,HW,C
        attn = (q @ k) * (C**-0.5)  # B,HW,HW

        # 3. 密度自适应温度
        density_flat = density.view(B, 1, -1)  # B,1,HW
        temperature = 1 + 2 * (1 - density_flat)  # 高密度→T→1（尖锐），低密度→T→3（平滑）
        attn = F.softmax(attn / temperature, dim=-1)

        out = (attn @ v).permute(0, 2, 1).view(B, C, H, W)
        return self.out(out) + x


class CDAA(nn.Module):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.m = nn.Sequential(*(CDA_attention(self.c) for _ in range(n)))

    def forward(self, x):
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))


if __name__ == "__main__":
    x = torch.randn(1, 3, 224, 224)
    model = CDAA(3, 3)
    y = model(x)
    print(y.shape)
    print(torch.__version__)
