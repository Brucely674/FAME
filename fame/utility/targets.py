import numpy as np

def selection_targets(candidates, gains, difficulty_entropy, aux_estimates, cfg):
    local = np.column_stack([gains, candidates.features[:, 1], candidates.features[:, 2], candidates.features[:, 4]])
    modes = []
    for m, group in enumerate(candidates.groups):
        rep = candidates.representatives[m]
        stat = candidates.statistics[m]
        uncertainty = stat[3] + cfg.uncertainty_beta * difficulty_entropy * abs(stat[1] - aux_estimates[m])
        length = np.linalg.norm(np.diff(candidates.paths[rep][:, :2], axis=0), axis=1).sum() / cfg.extent_m
        cost = cfg.cost_alpha * length + (1 - cfg.cost_alpha) * candidates.features[rep, 4]
        other = [r for r in candidates.representatives if r != rep]
        diversity = min((candidates.distances[rep, r] for r in other)) if other else 0.0
        modes.append([max(gains[group]), stat[2], uncertainty, cost, diversity])
    positive_path = int(np.argmax(gains))
    positive_mode = next((m for m, g in enumerate(candidates.groups) if positive_path in g))
    return dict(mode=np.asarray(modes, np.float32), local=local.astype(np.float32), positive_path=positive_path, positive_mode=positive_mode)
