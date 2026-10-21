import torch
from torch import nn
from .mapping_backbone import MappingBackbone
from .flow_modules import CFMAnchorProposer, HierarchicalSelector, mlp

class FAME(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = MappingBackbone()
        self.context = nn.Sequential(nn.Linear(1024, cfg.latent), nn.LayerNorm(cfg.latent), nn.ReLU())
        self.proposer = CFMAnchorProposer(cfg)
        self.selector = HierarchicalSelector(cfg)
        self.difficulty = mlp([cfg.latent, 128, 4])

    def forward(self, state):
        value, obstacle, features = self.backbone(state, return_features=True)
        z = self.context(features.mean((-2, -1)))
        return {'z': z, 'value': value, 'obstacle_logits': torch.logit(obstacle.clamp(1e-06, 1 - 1e-06)), 'difficulty_logits': self.difficulty(z)}
