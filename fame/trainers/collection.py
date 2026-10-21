import numpy as np
import torch
from fame.utility.simulator import MappingScene
from fame.utility.oracle import CoverageOracle, sparse_value_targets
from fame.utility.pipeline import propose, score_candidates, choose
from fame.utility.replay import pack_candidates
from fame.testers.fame_planning import FAMEPlanner

def collect_training_data(model, params, cfg, scene_names, replay, iteration, device, selector_only=False):
    rng = np.random.default_rng(cfg.seed + iteration)
    trajectories = cfg.train_trajectories_first if iteration == 0 else cfg.train_trajectories_later
    written = 0
    model.eval()
    for name in scene_names:
        scene = MappingScene(params, cfg, name, device)
        for rollout in range(trajectories):
            scene.reset(int(rng.integers(len(scene.starts))))
            planner = FAMEPlanner(model, cfg)
            for step in range(cfg.train_steps):
                state, output, previous, history = planner.analyze(scene)
                candidates = propose(model, output, history, previous, cfg)
                oracle = CoverageOracle(scene)
                bank_paths, anchors, bank_gains = ([], np.empty((0, 4), np.float32), np.empty(0, np.float32))
                if step % cfg.decision_stride == 0:
                    obstacle = scene.obstacle_target()
                    bank_paths, anchors, bank_gains = oracle.bank(obstacle, rng)
                    if len(anchors) and (not selector_only or candidates.paths):
                        gains = [oracle.gain(path) for path in candidates.paths]
                        values, mask = sparse_value_targets(anchors, bank_gains)
                        record = dict(state=state[0].cpu().numpy(), obstacle=obstacle[None], value=values, value_mask=mask, anchors=anchors, difficulty=np.array(cfg.difficulty_id, np.int64))
                        record.update(pack_candidates(candidates, gains))
                        replay.write(record)
                        written += 1
                if iteration < cfg.warmup_iterations and bank_paths and (not selector_only):
                    path = bank_paths[int(rng.integers(min(3, len(bank_paths))))]
                    planner.commit(scene, path, output)
                    target = planner.committed.pop(0) if planner.committed else None
                    planner.last_target = target
                else:
                    target = planner.next_pose(scene)
                if target is None:
                    break
                if not scene.execute(target):
                    planner.collision(scene, target)
        del scene
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return written
