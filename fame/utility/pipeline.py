from dataclasses import dataclass
import numpy as np
import torch
from .geometry import astar, descriptors, cluster_modes, path_tensor

@dataclass
class Candidates:
    paths: list
    features: np.ndarray
    groups: list
    representatives: list
    statistics: np.ndarray
    distances: np.ndarray

def propose(model, output, history, previous, cfg):
    with torch.no_grad():
        anchors = model.proposer.sample(output['z'])[0].cpu().numpy()
    obstacle = output['obstacle_logits'][0, 0].sigmoid().detach().cpu().numpy()
    value = output['value'][0].detach().cpu().numpy()
    paths = []
    for anchor in anchors:
        path = astar(obstacle, anchor, cfg)
        if path is not None and len(path) > 1:
            paths.append(path)
    features = np.array([descriptors(p, obstacle, value, history, previous, cfg) for p in paths], np.float32).reshape(-1, 6)
    groups, reps, statistics, distances = cluster_modes(paths, features, cfg)
    return Candidates(paths, features, groups, reps, statistics, distances)

def score_candidates(model, z, candidates, cfg):
    if not candidates.paths:
        return None

    def tensor(x):
        return torch.as_tensor(x, dtype=z.dtype, device=z.device)
    paths = tensor(np.stack([path_tensor(p, cfg) for p in candidates.paths]))
    stats = tensor(candidates.statistics)
    mode_heads, mode_scores = model.selector.mode(z, paths[candidates.representatives], stats)
    owner = np.zeros(len(paths), int)
    for m, group in enumerate(candidates.groups):
        owner[group] = m
    local_heads, local_scores = model.selector.local(z, paths, tensor(candidates.features), stats[torch.as_tensor(owner, device=z.device)])
    return dict(mode_heads=mode_heads, mode_scores=mode_scores, local_heads=local_heads, local_scores=local_scores)

def choose(candidates, scores):
    if scores is None:
        return None
    mode = int(scores['mode_scores'].argmax())
    group = candidates.groups[mode]
    best = group[int(scores['local_scores'][group].argmax())]
    return candidates.paths[best]
