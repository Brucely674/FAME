import heapq
import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt

def to_grid(xy, cfg):
    return np.rint((np.asarray(xy) / cfg.extent_m + 0.5) * cfg.image_size).astype(int)

def from_grid(xy, cfg):
    return (np.asarray(xy) / cfg.image_size - 0.5) * cfg.extent_m

def astar(obstacle, anchor, cfg):
    n = cfg.image_size
    start = tuple(to_grid([0.0, 0.0], cfg))
    goal = tuple(to_grid(np.asarray(anchor[:2]) * cfg.extent_m / 2, cfg))
    blocked = obstacle > cfg.obstacle_threshold

    def free(p):
        x, y = p
        return 0 <= x < n and 0 <= y < n and (not blocked[y, x])
    if not free(start) or not free(goal):
        return None
    costs, parent = ({start: 0.0}, {})
    queue = [(float(np.linalg.norm(np.subtract(goal, start))), 0.0, start)]
    while queue:
        _, cost, p = heapq.heappop(queue)
        if cost > costs[p]:
            continue
        if p == goal:
            route = [p]
            while p in parent:
                p = parent[p]
                route.append(p)
            xy = from_grid(np.array(route[::-1]), cfg)
            theta = np.zeros(len(xy))
            if len(xy) > 1:
                delta = np.diff(xy, axis=0)
                theta[:-1] = np.arctan2(delta[:, 1], delta[:, 0])
            theta[-1] = np.arctan2(anchor[2], anchor[3])
            return np.column_stack([xy, theta])
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            q = (p[0] + dx, p[1] + dy)
            if not free(q):
                continue
            if dx and dy and (not free((p[0] + dx, p[1])) or not free((p[0], p[1] + dy))):
                continue
            new = cost + np.hypot(dx, dy)
            if new < costs.get(q, np.inf):
                costs[q], parent[q] = (new, p)
                heapq.heappush(queue, (new + np.linalg.norm(np.subtract(goal, q)), new, q))
    return None

def resample(path, n):
    if len(path) == 1:
        return np.repeat(path, n, axis=0)
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1))]
    if arc[-1] < 1e-09:
        return np.repeat(path[-1:], n, axis=0)
    keep = np.r_[np.diff(arc) > 1e-09, True]
    arc, p = (arc[keep], path[keep])
    query = np.linspace(0, arc[-1], n)
    return np.column_stack([np.interp(query, arc, p[:, 0]), np.interp(query, arc, p[:, 1]), np.interp(query, arc, np.unwrap(p[:, 2]))])

def path_tensor(path, cfg):
    p = resample(path, cfg.path_samples)
    return np.column_stack([p[:, :2] / (cfg.extent_m / 2), np.sin(p[:, 2]), np.cos(p[:, 2])]).astype(np.float32)

def corridor(path, cfg):
    mask = np.zeros((cfg.image_size, cfg.image_size), bool)
    uv = to_grid(path[:, :2], cfg).clip(0, cfg.image_size - 1)
    mask[uv[:, 1], uv[:, 0]] = True
    return binary_dilation(mask, iterations=cfg.dilation)

def path_distance(a, b, cfg):
    p, q = (resample(a, cfg.path_samples), resample(b, cfg.path_samples))
    delta = np.linalg.norm(p[:, :2] - q[:, :2], axis=1) / (cfg.extent_m / 2)
    end = delta[-1] + cfg.endpoint_angle_weight * (1 - np.cos(p[-1, 2] - q[-1, 2]))
    ma, mb = (corridor(a, cfg), corridor(b, cfg))
    topo = 1 - (ma & mb).sum() / max(1, (ma | mb).sum())
    return float(np.dot(cfg.distance_weights, [end, delta.mean(), delta[:cfg.prefix_samples].mean(), topo]))

def continuity(path, previous, cfg):
    if previous is None or len(previous) == 0:
        return 0.0
    p, q = (resample(path[:cfg.commitment], cfg.commitment), resample(previous[:cfg.commitment], cfg.commitment))
    alpha = cfg.continuity_alpha
    return alpha * np.linalg.norm(p[:, :2] - q[:, :2], axis=1).mean() / (cfg.extent_m / 2) + (1 - alpha) * (1 - np.cos(p[:, 2] - q[:, 2])).mean()

def value_at(value, xy, theta, cfg):
    u, v = np.rint((np.asarray(xy) / cfg.extent_m + 0.5) * 64).astype(int).clip(0, 63)
    azimuth = np.arctan2(-np.cos(theta), -np.sin(theta))
    heading = int(np.rint(azimuth % (2 * np.pi) / (2 * np.pi) * 8)) % 8
    return float(value[heading, v, u])

def descriptors(path, obstacle, value, history, previous, cfg):
    uv = to_grid(path[:, :2], cfg).clip(0, cfg.image_size - 1)
    x, y = uv.T
    p = obstacle[y, x].clip(1e-06, 1 - 1e-06)
    blocked = obstacle > cfg.obstacle_threshold
    clearance = distance_transform_edt(~blocked) * cfg.extent_m / cfg.image_size
    nearest = clearance[y, x].min() if blocked.any() else cfg.extent_m
    risk = cfg.risk_alpha * p.mean() + (1 - cfg.risk_alpha) / (nearest + 1e-06)
    turns = np.abs(np.arctan2(np.sin(np.diff(path[:, 2])), np.cos(np.diff(path[:, 2]))))
    cost = cfg.cost_alpha * len(path) / cfg.max_path_cells + (1 - cfg.cost_alpha) * (turns.mean() / np.pi if len(turns) else 0)
    return np.array([value_at(value, path[-1, :2], path[-1, 2], cfg), risk, cost, 1 - history[y, x].mean(), continuity(path, previous, cfg), (-p * np.log(p) - (1 - p) * np.log(1 - p)).mean()], np.float32)

def cluster_modes(paths, features, cfg):
    count = len(paths)
    if not count:
        return ([], [], np.empty((0, 6), np.float32), np.empty((0, 0)))
    distance = np.array([[path_distance(a, b, cfg) for b in paths] for a in paths])
    groups = [[i] for i in range(count)]
    while len(groups) > 1:
        score, i, j = min(((distance[np.ix_(a, b)].mean(), i, j) for i, a in enumerate(groups) for j, b in enumerate(groups) if i < j))
        if score > cfg.mode_threshold:
            break
        groups[i] += groups.pop(j)
    representatives, stats = ([], [])
    for members in groups:
        sub = distance[np.ix_(members, members)]
        representative = members[int(sub.sum(1).argmin())]
        representatives.append(representative)
        f = features[members]
        stats.append([len(members), f[:, 0].mean(), f[:, 1].mean(), f[:, 0].var(), f[:, 1].var(), distance[members, representative].mean()])
    return (groups, representatives, np.asarray(stats, np.float32), distance)
