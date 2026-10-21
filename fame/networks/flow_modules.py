import math
import torch
from torch import nn
from torch.nn import functional as F

def mlp(dims):
    layers = []
    for i, (a, b) in enumerate(zip(dims, dims[1:])):
        layers.append(nn.Linear(a, b))
        if i < len(dims) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)

class SinusoidalTimeEmbedding(nn.Module):

    def __init__(self, dim=256):
        super().__init__()
        self.register_buffer('freq', torch.exp(-math.log(10000) * torch.arange(dim // 2) / (dim // 2 - 1)))

    def forward(self, t):
        phase = t[:, None] * self.freq[None]
        return torch.cat([phase.sin(), phase.cos()], -1)

class CFMAnchorProposer(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        h = cfg.hidden
        self.time = nn.Sequential(SinusoidalTimeEmbedding(h), nn.Linear(h, h), nn.ReLU())
        self.anchor = nn.Sequential(nn.Linear(4, h), nn.ReLU())
        self.context = nn.Sequential(nn.Linear(cfg.latent, h), nn.ReLU())
        self.velocity = mlp([3 * h, h, h, 4])

    def forward(self, x, t, z):
        return self.velocity(torch.cat([self.anchor(x), self.time(t), self.context(z)], -1))

    def flow_matching_loss(self, z, target):
        x0 = torch.randn_like(target)
        t = torch.rand(len(target), device=target.device)
        xt = (1 - t[:, None]) * x0 + t[:, None] * target
        return ((self(xt, t, z) - (target - x0)) ** 2).sum(-1).mean()

    def sample(self, z, k=None):
        k = self.cfg.candidates if k is None else k
        context = z.repeat_interleave(k, 0)
        x = torch.randn(len(context), 4, device=z.device) * self.cfg.temperature
        for i in range(self.cfg.flow_steps):
            t = x.new_full((len(x),), i / self.cfg.flow_steps)
            x = x + self(x, t, context) / self.cfg.flow_steps
        direction = F.normalize(x[:, 2:4], dim=-1, eps=1e-08)
        default = torch.zeros_like(direction)
        default[:, 1] = 1
        direction = torch.where(x[:, 2:4].norm(dim=-1, keepdim=True) > 1e-08, direction, default)
        return torch.cat([x[:, :2].clamp(-1, 1), direction], -1).reshape(len(z), k, 4)

    def diversity_loss(self, anchors):
        k = anchors.shape[1]
        if k < 2:
            return anchors.sum() * 0
        squared = (anchors[:, :, None] - anchors[:, None, :]).square().sum(-1)
        mask = ~torch.eye(k, dtype=torch.bool, device=anchors.device)
        return torch.exp(-squared / self.cfg.sigma ** 2)[:, mask].mean()

class PathEncoder(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.net = mlp([cfg.path_samples * 4, 256, 128])

    def forward(self, paths):
        return self.net(paths.flatten(1))

class HierarchicalSelector(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.path = PathEncoder(cfg)
        self.ctx = mlp([cfg.latent, 128, 128])
        self.mode_norm = nn.LayerNorm(6)
        self.path_norm = nn.LayerNorm(6)
        self.mode_net = mlp([262, 256, 256, 5])
        self.local_net = mlp([268, 256, 256, 4])
        self.raw_weights = nn.Parameter(torch.zeros(5))

    def mode(self, z, representatives, statistics):
        x = torch.cat([self.ctx(z).expand(len(representatives), -1), self.path(representatives), self.mode_norm(statistics)], -1)
        heads = self.mode_net(x)
        signs = heads.new_tensor([1, -1, -1, -1, 1])
        scores = (heads * F.softplus(self.raw_weights) * signs).sum(-1)
        return (heads, scores)

    def local(self, z, paths, descriptors, statistics):
        x = torch.cat([self.ctx(z).expand(len(paths), -1), self.path(paths), self.path_norm(descriptors), self.mode_norm(statistics)], -1)
        heads = self.local_net(x)
        r, c, h = self.cfg.local_weights
        return (heads, heads @ heads.new_tensor([1, -r, -c, -h]))
