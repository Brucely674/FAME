from pathlib import Path
import json
import random
import numpy as np
import torch
from fame.networks.fame_model import FAME
from fame.utility.simulator import MappingScene
from .fame_planning import FAMEPlanner

def load_fame(checkpoint, cfg, device):
    model = FAME(cfg).to(device)
    data = torch.load(checkpoint, map_location=device, weights_only=True)
    for field in ('latent', 'hidden', 'anchor_dim', 'path_samples'):
        if field in data.get('config', {}) and data['config'][field] != getattr(cfg, field):
            raise ValueError(f'Checkpoint shape configuration mismatch: {field}')
    model.load_state_dict(data['model_state_dict'], strict=True)
    return model.eval()

@torch.no_grad()
def run_episode(scene, model, cfg, start, budget):
    scene.reset(start)
    planner = FAMEPlanner(model, cfg)
    coverage = [scene.metrics()['coverage']]
    for step in range(budget):
        target = planner.next_pose(scene)
        if target is None:
            break
        if not scene.execute(target):
            planner.collision(scene, target)
        coverage.append(scene.metrics()['coverage'])
    coverage.extend([coverage[-1]] * (budget + 1 - len(coverage)))
    auc = sum(((a + b) / 2 for a, b in zip(coverage, coverage[1:]))) / max(1, budget)
    result = scene.metrics()
    result.update(auc=auc, coverage_curve=coverage, collisions=scene.collisions, poses=torch.stack(scene.trajectory).cpu().tolist())
    return result

def run_test_fame(params, cfg, checkpoint, output_dir, device='cuda:0', budget=None, save_cloud=False):
    if not params.test_scenes:
        raise ValueError('test_scenes must explicitly identify scenes')
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model = load_fame(checkpoint, cfg, device)
    budget = (100, 200, 400, 500)[cfg.difficulty_id] if budget is None else budget
    records = []
    for name in params.test_scenes:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        scene = MappingScene(params, cfg, name, device)
        for start in range(cfg.eval_starts):
            seed = cfg.seed + start
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            result = run_episode(scene, model, cfg, start, budget)
            result.update(scene=name, start=start, seed=seed)
            records.append(result)
            if save_cloud:
                np.savez_compressed(output / f'{name}_{start}.npz', points=scene.points.cpu().numpy() / scene.scale, colors=scene.colors.cpu().numpy())
        del scene
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    summary = {key: {'mean': float(np.mean([r[key] for r in records])), 'std': float(np.std([r[key] for r in records]))} for key in ('coverage', 'auc', 'completion_percent', 'completion_cm')}
    (output / 'results.json').write_text(json.dumps({'summary': summary, 'episodes': records}, indent=2))
    return summary
