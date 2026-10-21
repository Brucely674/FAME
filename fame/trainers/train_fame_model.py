from pathlib import Path
import json
import random
import numpy as np
import torch
from torch.nn import functional as F
from fame.networks.fame_model import FAME
from fame.utility.replay import ReplayBuffer, read_record, unpack_candidates
from fame.utility.geometry import value_at
from fame.utility.pipeline import score_candidates
from fame.utility.targets import selection_targets
from fame.utility.losses import selection_loss
from .collection import collect_training_data

def configure_stage(model, stage):
    allowed = {'context': {'backbone', 'context', 'difficulty'}, 'anchor': {'proposer'}, 'selector': {'selector'}}
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(stage == 'joint' or name.split('.')[0] in allowed[stage])

def stage_mode(model, stage, training):
    model.train(training)
    if stage in ('anchor', 'selector'):
        model.backbone.eval()
        model.context.eval()
        model.difficulty.eval()
    if stage == 'selector':
        model.proposer.eval()

def compute_loss(model, record, stage, cfg, device, rng):
    tensor = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)
    output = model(tensor(record['state'])[None])
    z = output['z']
    total = z.sum() * 0
    if stage in ('context', 'joint'):
        obs = tensor(record['obstacle'])[None]
        values = tensor(record['value'])[None]
        mask = tensor(record['value_mask'])[None]
        value_loss = ((output['value'] - values).square() * mask).sum() / mask.sum().clamp_min(1)
        difficulty = torch.as_tensor(record['difficulty'], dtype=torch.long, device=device).reshape(1)
        total = total + F.binary_cross_entropy_with_logits(output['obstacle_logits'], obs)
        total = total + cfg.lambda_aux * value_loss + F.cross_entropy(output['difficulty_logits'], difficulty)
    if stage in ('anchor', 'joint'):
        target = tensor(record['anchors'][rng.integers(len(record['anchors']))])[None]
        total = total + model.proposer.flow_matching_loss(z, target)
        total = total + cfg.lambda_div * model.proposer.diversity_loss(model.proposer.sample(z))
    if stage in ('selector', 'joint'):
        candidates = unpack_candidates(record)
        if not candidates.paths:
            return None if stage == 'selector' else total
        probs = output['difficulty_logits'].softmax(-1)
        entropy = float(-(probs * probs.clamp_min(1e-08).log()).sum().detach())
        value = output['value'][0].detach().cpu().numpy()
        estimates = [np.mean([value_at(value, p[:2], p[2], cfg) for p in candidates.paths[r]]) for r in candidates.representatives]
        targets = selection_targets(candidates, record['gains'], entropy, estimates, cfg)
        scores = score_candidates(model, z, candidates, cfg)
        total = total + selection_loss(scores, targets, candidates.groups)
    return total

def run_epoch(model, paths, stage, optimizer, cfg, device, training, rng):
    stage_mode(model, stage, training)
    if training:
        optimizer.zero_grad(set_to_none=True)
    loss_sum, count, pending = (0.0, 0, 0)

    def update():
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(pending)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    with torch.set_grad_enabled(training):
        for index in rng.permutation(len(paths)):
            record = read_record(paths[index])
            loss = compute_loss(model, record, stage, cfg, device, rng)
            if loss is None:
                continue
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite {stage} loss in {paths[index]}')
            loss_sum += float(loss.detach())
            count += 1
            if training:
                loss.backward()
                pending += 1
                if pending == cfg.effective_batch:
                    update()
                    pending = 0
    if training and pending:
        update()
    if not count:
        raise ValueError(f'No valid {stage} samples in replay')
    return loss_sum / count

def train_stage(model, replay, stage, epochs, cfg, device, output, iteration):
    if epochs <= 0:
        return
    configure_stage(model, stage)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    best, stale = (float('inf'), 0)
    checkpoint = output / f'{stage}_best.pth'
    rng = np.random.default_rng(cfg.seed + iteration)
    rows = []
    for epoch in range(epochs):
        train_loss = run_epoch(model, replay.files('train'), stage, optimizer, cfg, device, True, rng)
        devices = [torch.device(device).index or 0] if str(device).startswith('cuda') else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(cfg.seed)
            validation = run_epoch(model, replay.files('validation'), stage, optimizer, cfg, device, False, np.random.default_rng(cfg.seed))
        rows.append(dict(epoch=epoch, train_loss=train_loss, validation_loss=validation))
        if validation < best:
            best, stale = (validation, 0)
            torch.save(dict(model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(), config=vars(cfg), epoch=epoch, iteration=iteration, stage=stage, validation_loss=validation), checkpoint)
        else:
            stale += 1
        if stale >= cfg.patience:
            break
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(saved['model_state_dict'])
    (output / f'{stage}_log.json').write_text(json.dumps(rows, indent=2))

def run_training_fame(params, cfg, output_dir, device='cuda:0', resume=None):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if not params.train_scenes:
        raise ValueError('train_scenes must explicitly identify the training split')
    if set(params.train_scenes) & set(params.test_scenes):
        raise ValueError('Training and test scenes overlap')
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    replay = ReplayBuffer(output / 'replay', cfg.replay_capacity)
    model = FAME(cfg).to(device)
    start = 0
    if resume:
        saved = torch.load(resume, map_location=device, weights_only=True)
        if saved['config'] != vars(cfg):
            raise ValueError('Resume configuration differs from checkpoint')
        model.load_state_dict(saved['model_state_dict'])
        start = saved['iteration'] + 1
    for iteration in range(start, cfg.collection_iterations):
        folder = output / f'iteration_{iteration:02d}'
        folder.mkdir(exist_ok=True)
        collect_training_data(model, params, cfg, params.train_scenes, replay, iteration, device)
        if not replay.files('validation'):
            replay.holdout(cfg.validation_size, np.random.default_rng(cfg.seed))
        train_stage(model, replay, 'context', cfg.context_epochs, cfg, device, folder, iteration)
        train_stage(model, replay, 'anchor', cfg.anchor_epochs, cfg, device, folder, iteration)
        selector_replay = ReplayBuffer(folder / 'selector_replay', cfg.replay_capacity)
        collect_training_data(model, params, cfg, params.train_scenes, selector_replay, iteration, device, selector_only=True)
        if not selector_replay.files('validation'):
            selector_replay.holdout(cfg.validation_size, np.random.default_rng(cfg.seed))
        train_stage(model, selector_replay, 'selector', cfg.selector_epochs, cfg, device, folder, iteration)
        train_stage(model, selector_replay, 'joint', cfg.joint_epochs, cfg, device, folder, iteration)
        torch.save(dict(model_state_dict=model.state_dict(), config=vars(cfg), iteration=iteration), output / 'latest.pth')
