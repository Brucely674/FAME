import numpy as np
import torch
from fame.utility.geometry import to_grid, value_at
from fame.utility.pipeline import propose, score_candidates, choose
from fame.utility.oracle import action_path

class FAMEPlanner:

    def __init__(self, model, cfg):
        self.model, self.cfg = (model, cfg)
        self.reset()

    def reset(self):
        self.committed = []
        self.last_target = None
        self.reference_gain = None
        self.blocked_locations = []

    @torch.no_grad()
    def analyze(self, scene):
        self.model.eval()
        state = scene.state()
        output = self.model(state)
        probabilities = output['obstacle_logits'].sigmoid()
        probabilities[state[:, 4:5] > 0] = 0
        center = scene.pose[[0, 2]].detach().cpu().numpy()
        for point in self.blocked_locations:
            local = -(point - center) / scene.scale
            x, y = to_grid(local, self.cfg)
            if 0 <= x < self.cfg.image_size and 0 <= y < self.cfg.image_size:
                probabilities[0, 0, max(0, y - 1):y + 2, max(0, x - 1):x + 2] = 1
        output['obstacle_logits'] = torch.logit(probabilities.clamp(1e-06, 1 - 1e-06))
        previous = scene.poses_to_local(self.committed)
        history = state[0, 4].cpu().numpy()
        history = history / max(1.0, float(history.max()))
        return (state, output, previous, history)

    def should_replan(self, scene, output, previous):
        if previous is None:
            return True
        uv = to_grid(previous[:, :2], self.cfg)
        if ((uv < 0) | (uv >= self.cfg.image_size)).any():
            return True
        obstacle = output['obstacle_logits'][0, 0].sigmoid().cpu().numpy()
        if (obstacle[uv[:, 1], uv[:, 0]] > self.cfg.obstacle_threshold).any():
            return True
        if self.last_target is not None:
            deviation = float(torch.linalg.vector_norm(scene.pose[:3] - self.last_target[:3])) / scene.scale
            if deviation > self.cfg.deviation_m:
                return True
        value = output['value'][0].cpu().numpy()
        gain = value_at(value, previous[-1, :2], previous[-1, 2], self.cfg)
        return self.reference_gain is not None and self.reference_gain > 0 and (gain < self.reference_gain * self.cfg.gain_drop_ratio)

    def commit(self, scene, path, output):
        sampled = action_path(path, self.cfg)[1:self.cfg.commitment + 1]
        self.committed = scene.path_poses(sampled)
        if len(sampled):
            value = output['value'][0].detach().cpu().numpy()
            self.reference_gain = value_at(value, sampled[-1, :2], sampled[-1, 2], self.cfg)

    @torch.no_grad()
    def next_pose(self, scene):
        state, output, previous, history = self.analyze(scene)
        if self.should_replan(scene, output, previous):
            candidates = propose(self.model, output, history, previous, self.cfg)
            scores = score_candidates(self.model, output['z'], candidates, self.cfg)
            selected = choose(candidates, scores)
            if selected is None:
                self.committed = []
                target = scene.pose.clone()
                target[4] = (target[4] + 45) % 360
                self.last_target = target.clone()
                return target
            self.commit(scene, selected, output)
        if not self.committed:
            return None
        self.last_target = self.committed.pop(0)
        return self.last_target.clone()

    def collision(self, scene, target):
        midpoint = ((scene.pose[:3] + target[:3]) / 2)[[0, 2]].detach().cpu().numpy()
        self.blocked_locations.append(midpoint)
        self.committed = []
