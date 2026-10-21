import argparse
from dataclasses import fields, replace
from pathlib import Path
import runpy
from types import SimpleNamespace
from fame.utility.config import Config

def load_configuration(filename):
    namespace = runpy.run_path(filename)
    simulator = namespace.get('SIMULATOR', {})
    values = namespace.get('MODEL', {})
    known = {field.name for field in fields(Config)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f'Unknown model configuration: {sorted(unknown)}')
    return (SimpleNamespace(**simulator), replace(Config(), **values))

def parser():
    root = argparse.ArgumentParser(prog='fame')
    commands = root.add_subparsers(dest='command', required=True)
    default = str(Path(__file__).resolve().parents[1] / 'configs' / 'default.py')
    for name in ('train', 'test', 'infer'):
        command = commands.add_parser(name)
        command.add_argument('--config', default=default)
        command.add_argument('--data-path', required=True)
        command.add_argument('--scenes', nargs='+', required=True)
        command.add_argument('--output', required=True)
        command.add_argument('--device', default='cuda:0')
        command.add_argument('--difficulty', choices=['simple', 'normal', 'hard', 'insane'], default='simple')
        command.add_argument('--seed', type=int, default=0)
        if name == 'train':
            command.add_argument('--resume')
            command.add_argument('--iterations', type=int)
        else:
            command.add_argument('--checkpoint', required=True)
            command.add_argument('--budget', type=int)
            command.add_argument('--starts', type=int, default=5)
    return root

def main(argv=None):
    args = parser().parse_args(argv)
    params, cfg = load_configuration(args.config)
    params.data_path = str(Path(args.data_path).resolve())
    cfg = replace(cfg, seed=args.seed, difficulty_id=['simple', 'normal', 'hard', 'insane'].index(args.difficulty))
    if args.command == 'train':
        params.train_scenes = args.scenes
        if args.iterations is not None:
            if not 1 <= args.iterations <= 15:
                raise ValueError('iterations must be in [1, 15]')
            cfg = replace(cfg, collection_iterations=args.iterations)
        from fame.trainers.train_fame_model import run_training_fame
        run_training_fame(params, cfg, args.output, args.device, args.resume)
    else:
        if args.starts < 1 or (args.budget is not None and args.budget < 1):
            raise ValueError('starts and budget must be positive')
        params.test_scenes = args.scenes
        cfg = replace(cfg, eval_starts=args.starts)
        from fame.testers.evaluate import run_test_fame
        summary = run_test_fame(params, cfg, args.checkpoint, args.output, args.device, args.budget, save_cloud=args.command == 'infer')
        print(summary)
