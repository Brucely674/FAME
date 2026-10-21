import numpy as np
import torch
from .geometry import astar, to_grid

def action_path(path, cfg):
    if len(path) < 2:
        return path.copy()
    result = [path[0]]
    distance = 0.0
    for i in range(1, len(path)):
        distance += np.linalg.norm(path[i, :2] - path[i - 1, :2])
        corner = i + 1 < len(path) and abs(np.arctan2(np.sin(path[i, 2] - path[i - 1, 2]), np.cos(path[i, 2] - path[i - 1, 2]))) > 0.1
        if distance >= cfg.movement_m or corner or i == len(path) - 1:
            result.append(path[i])
            distance = 0.0
    return np.asarray(result)

class CoverageOracle:

    def __init__(self, scene):
        self.scene = scene
        self.origin = scene.pose.clone()
        self.covered = scene.coverage_mask()
        self.cache = {}

    @torch.no_grad()
    def gain(self, path):
        key = np.asarray(path, dtype=np.float32).tobytes()
        if key in self.cache:
            return self.cache[key]
        poses = self.scene.path_poses(action_path(path, self.scene.cfg), self.origin)
        covered = self.covered.copy()
        old = self.origin
        for target in poses[1:]:
            if self.scene.blocked(old, target):
                self.cache[key] = 0.0
                return 0.0
            delta = (target[4] - old[4] + 180) % 360 - 180
            for step in range(1, self.scene.params.n_interpolation_steps + 1):
                ratio = step / self.scene.params.n_interpolation_steps
                pose = old + ratio * (target - old)
                pose[4] = old[4] + ratio * delta
                devices = [self.scene.device.index or 0] if self.scene.device.type == 'cuda' else []
                with torch.random.fork_rng(devices=devices):
                    torch.manual_seed(self.scene.cfg.seed)
                    points, _ = self.scene.render(pose)
                covered |= self.scene.coverage_mask(points)
            old = target
        gain = float((covered & ~self.covered).mean())
        self.cache[key] = gain
        return gain

    def bank(self, obstacle, rng):
        cfg = self.scene.cfg
        free = np.argwhere(obstacle <= cfg.obstacle_threshold)
        if not len(free):
            return ([], np.empty((0, 4), np.float32), np.empty(0, np.float32))
        indices = rng.choice(len(free), min(cfg.oracle_samples, len(free)), replace=False)
        paths, anchors, gains = ([], [], [])
        for index in indices:
            row, col = free[index]
            theta = rng.uniform(-np.pi, np.pi)
            anchor = np.array([2 * col / cfg.image_size - 1, 2 * row / cfg.image_size - 1, np.sin(theta), np.cos(theta)], np.float32)
            path = astar(obstacle, anchor, cfg)
            if path is None or len(path) < 2:
                continue
            paths.append(path)
            anchors.append(anchor)
            gains.append(self.gain(path))
        if not paths:
            return ([], np.empty((0, 4), np.float32), np.empty(0, np.float32))
        order = np.argsort(gains)[::-1][:cfg.oracle_keep]
        return ([paths[i] for i in order], np.asarray(anchors)[order], np.asarray(gains, np.float32)[order])

def sparse_value_targets(anchors, gains):
    values = np.zeros((8, 64, 64), np.float32)
    mask = np.zeros_like(values)
    for anchor, gain in zip(anchors, gains):
        col, row = np.rint((anchor[:2] + 1) * 32).astype(int).clip(0, 63)
        theta = np.arctan2(anchor[2], anchor[3])
        azimuth = np.arctan2(-np.cos(theta), -np.sin(theta))
        heading = int(np.rint(azimuth % (2 * np.pi) / (2 * np.pi) * 8)) % 8
        values[heading, row, col] = max(values[heading, row, col], gain)
        mask[heading, row, col] = 1
    return (values, mask)
