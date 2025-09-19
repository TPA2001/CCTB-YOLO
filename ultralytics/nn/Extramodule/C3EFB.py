import torch
import torch.nn as nn
from einops import rearrange


class EfficientAttention(nn.Module):

    def __init__(self, channels, c2=None, num_heads=1, factor=16, bias=False):
        super(EfficientAttention, self).__init__()
        self.groups = factor  # 分组数量
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)  # Softmax层
        self.agp = nn.AdaptiveAvgPool2d((1, 1))  # 全局平均池化
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # 沿高度方向的平均池化
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # 沿宽度方向的平均池化
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)  # 分组归一化
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)  # 1x1卷积
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1, groups=channels // self.groups)  # 3x3深度卷积
        self.num_heads = num_heads  # 注意力头数量
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=bias)  # QKV投影
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=bias)  # 输出投影
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))  # 温度参数

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # 分组后的输入
        x_h = self.pool_h(group_x)  # 高度方向的池化
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)  # 宽度方向的池化
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))  # 池化后的特征融合
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())  # 分组归一化
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))  # 全局注意力权重


        qkv = self.qkv(x)  # QKV投影
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)  # 归一化Q
        k = torch.nn.functional.normalize(k, dim=-1)  # 归一化K

        attn = (q @ k.transpose(-2, -1)) * self.temperature  # 计算注意力分数
        attn = attn.softmax(dim=-1)  # Softmax归一化
        out = (attn @ v)  # 注意力加权求和
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)  # 输出投影
        x12 = out.reshape(b * self.groups, -1, h*w)
        weights = (torch.matmul(x11, x12)).reshape(b * self.groups, 1, h, w)  # 权重计算
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)+out  # 加权特征与输出特征相加

def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    default_act = nn.SiLU()  # 默认激活函数

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)  # 卷积层
        self.bn = nn.BatchNorm2d(c2)  # 批量归一化
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()  # 激活函数

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))  # 前向传播

    def forward_fuse(self, x):
        return self.act(self.conv(x))  # 融合前向传播


class Bottleneck(nn.Module):

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.25):
        super().__init__()
        c_ = int(c2 * e)  # 隐藏通道数
        self.cv1 = Conv(c1, c_, k[0], 1)  # 第一个卷积
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)  # 第二个卷积
        self.add = shortcut and c1 == c2  # 是否添加残差连接

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))  # 前向传播


class C3EFB(nn.Module):

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.25):
        super().__init__()
        self.c = int(c2 * e) 
        self.cv1 = Conv(c1, 2 * self.c, 1, 1) 
        self.efficient_attention = EfficientAttention(2 * self.c, factor=16) 
        self.cv2 = Conv((2 + n) * self.c, c2, 1) 
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)) 

    def forward(self, x):
        y = list(self.efficient_attention(self.cv1(x)).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)  
        return self.cv2(torch.cat(y, 1)) 

    def forward_split(self, x):
        y = list(self.efficient_attention(self.cv1(x)).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    x = torch.randn(1, 512, 16, 16)  # 输入张量
    model = C3EFB(512, 256)  # 创建模型
    print(model(x).shape)  # 输出张量形状

    # 计算并打印模型的参数量
    total_params = count_parameters(model)
    print(f"模型的总参数量: {total_params}")