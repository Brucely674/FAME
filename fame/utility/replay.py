from pathlib import Path
import uuid
import numpy as np
from .pipeline import Candidates

class ReplayBuffer:

    def __init__(self, root, capacity=50000):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.capacity = capacity

    def write(self, record, split='train'):
        folder = self.root / split
        folder.mkdir(exist_ok=True)
        path = folder / (uuid.uuid4().hex + '.npz')
        np.savez_compressed(path, **record)
        if split == 'train':
            paths = sorted(folder.glob('*.npz'), key=lambda p: p.stat().st_mtime_ns)
            for obsolete in paths[:-self.capacity]:
                obsolete.unlink()
        return path

    def files(self, split):
        return sorted((self.root / split).glob('*.npz'))

    def holdout(self, count, rng):
        files = self.files('train')
        if len(files) <= count:
            raise ValueError(f'Collected {len(files)} states; more than {count} are required for the validation holdout')
        folder = self.root / 'validation'
        folder.mkdir(exist_ok=True)
        for index in rng.choice(len(files), count, replace=False):
            files[index].rename(folder / files[index].name)

def pack_candidates(candidates, gains):
    lengths = np.array([len(p) for p in candidates.paths], np.int64)
    owner = np.zeros(len(lengths), np.int64)
    for i, group in enumerate(candidates.groups):
        owner[group] = i
    return dict(paths=np.concatenate(candidates.paths) if len(lengths) else np.empty((0, 3)), offsets=np.r_[0, np.cumsum(lengths)], features=candidates.features, owner=owner, representatives=np.array(candidates.representatives, np.int64), statistics=candidates.statistics, distances=candidates.distances, gains=np.asarray(gains, np.float32))

def unpack_candidates(record):
    offsets = record['offsets']
    paths = [record['paths'][a:b] for a, b in zip(offsets[:-1], offsets[1:])]
    groups = [np.flatnonzero(record['owner'] == m).tolist() for m in range(len(record['representatives']))]
    return Candidates(paths, record['features'], groups, record['representatives'].tolist(), record['statistics'], record['distances'])

def read_record(path):
    with np.load(path, allow_pickle=False) as file:
        return {key: file[key] for key in file.files}
