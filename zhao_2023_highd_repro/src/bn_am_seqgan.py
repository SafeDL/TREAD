from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .common import finite_summary, output_dir, save_json, set_seed
from .generator import (
    _canonical_sequences,
    _duration_rmse,
    _monotonic_rate,
    _regularize_sequence,
    _states_from_sequence,
)


class BNLSTMCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.linear = nn.Linear(input_dim + hidden_dim, 4 * hidden_dim)
        self.gate_bn = nn.BatchNorm1d(4 * hidden_dim)
        self.cell_bn = nn.BatchNorm1d(hidden_dim)

    def _bn(self, layer: nn.BatchNorm1d, values: torch.Tensor) -> torch.Tensor:
        if self.training and values.shape[0] < 2:
            return values
        return layer(values)

    def forward(
        self,
        values: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h, c = state
        gates = self._bn(self.gate_bn, self.linear(torch.cat([values, h], dim=-1)))
        f, i, g, o = gates.chunk(4, dim=-1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(self._bn(self.cell_bn, c))
        return h, c


class BNSeqGenerator(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, seq_len: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.bos_token = vocab_size
        self.seq_len = seq_len
        self.embedding = nn.Embedding(vocab_size + 1, embed_dim)
        self.input_bn = nn.BatchNorm1d(embed_dim)
        self.cell = BNLSTMCell(embed_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def _embed(self, token: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(token)
        if self.training and emb.shape[0] < 2:
            return emb
        return self.input_bn(emb)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, steps = tokens.shape
        prev = torch.full((batch,), self.bos_token, dtype=torch.long, device=tokens.device)
        h = torch.zeros(batch, self.cell.hidden_dim, device=tokens.device)
        c = torch.zeros_like(h)
        logits = []
        for step in range(steps):
            h, c = self.cell(self._embed(prev), (h, c))
            logits.append(self.output(h))
            prev = tokens[:, step]
        return torch.stack(logits, dim=1)

    def sample(
        self,
        batch: int,
        device: torch.device,
        temperature: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prev = torch.full((batch,), self.bos_token, dtype=torch.long, device=device)
        h = torch.zeros(batch, self.cell.hidden_dim, device=device)
        c = torch.zeros_like(h)
        tokens = []
        log_probs = []
        for _ in range(self.seq_len):
            h, c = self.cell(self._embed(prev), (h, c))
            logits = self.output(h) / max(float(temperature), 1.0e-3)
            dist = torch.distributions.Categorical(logits=logits)
            prev = dist.sample()
            tokens.append(prev)
            log_probs.append(dist.log_prob(prev))
        return torch.stack(tokens, dim=1), torch.stack(log_probs, dim=1)

    def complete_from_prefix(
        self,
        prefix: torch.Tensor,
        device: torch.device,
        temperature: float,
    ) -> torch.Tensor:
        batch, prefix_len = prefix.shape
        prev = torch.full((batch,), self.bos_token, dtype=torch.long, device=device)
        h = torch.zeros(batch, self.cell.hidden_dim, device=device)
        c = torch.zeros_like(h)
        out = []
        for step in range(self.seq_len):
            h, c = self.cell(self._embed(prev), (h, c))
            if step < prefix_len:
                prev = prefix[:, step]
            else:
                logits = self.output(h) / max(float(temperature), 1.0e-3)
                prev = torch.distributions.Categorical(logits=logits).sample()
            out.append(prev)
        return torch.stack(out, dim=1)


class PlainSeqGenerator(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, seq_len: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.bos_token = vocab_size
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size + 1, embed_dim)
        self.cell = nn.LSTMCell(embed_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, steps = tokens.shape
        prev = torch.full((batch,), self.bos_token, dtype=torch.long, device=tokens.device)
        h = torch.zeros(batch, self.hidden_dim, device=tokens.device)
        c = torch.zeros_like(h)
        logits = []
        for step in range(steps):
            h, c = self.cell(self.embedding(prev), (h, c))
            logits.append(self.output(h))
            prev = tokens[:, step]
        return torch.stack(logits, dim=1)

    def sample(
        self,
        batch: int,
        device: torch.device,
        temperature: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prev = torch.full((batch,), self.bos_token, dtype=torch.long, device=device)
        h = torch.zeros(batch, self.hidden_dim, device=device)
        c = torch.zeros_like(h)
        tokens = []
        log_probs = []
        for _ in range(self.seq_len):
            h, c = self.cell(self.embedding(prev), (h, c))
            logits = self.output(h) / max(float(temperature), 1.0e-3)
            dist = torch.distributions.Categorical(logits=logits)
            prev = dist.sample()
            tokens.append(prev)
            log_probs.append(dist.log_prob(prev))
        return torch.stack(tokens, dim=1), torch.stack(log_probs, dim=1)

    def complete_from_prefix(
        self,
        prefix: torch.Tensor,
        device: torch.device,
        temperature: float,
    ) -> torch.Tensor:
        batch, prefix_len = prefix.shape
        prev = torch.full((batch,), self.bos_token, dtype=torch.long, device=device)
        h = torch.zeros(batch, self.hidden_dim, device=device)
        c = torch.zeros_like(h)
        out = []
        for step in range(self.seq_len):
            h, c = self.cell(self.embedding(prev), (h, c))
            if step < prefix_len:
                prev = prefix[:, step]
            else:
                logits = self.output(h) / max(float(temperature), 1.0e-3)
                prev = torch.distributions.Categorical(logits=logits).sample()
            out.append(prev)
        return torch.stack(out, dim=1)


class SelfAttention(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        self.scale = math.sqrt(float(dim))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        q, k, v = self.qkv(values).chunk(3, dim=-1)
        score = torch.matmul(q, k.transpose(1, 2)) / self.scale
        return self.out(torch.matmul(torch.softmax(score, dim=-1), v))


class AMDiscriminator(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, conv_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pre_attention = SelfAttention(embed_dim)
        self.conv = nn.Conv1d(embed_dim, conv_dim, kernel_size=3, padding=1)
        self.conv_bn = nn.BatchNorm1d(conv_dim)
        self.post_attention = SelfAttention(conv_dim)
        self.classifier = nn.Linear(conv_dim, 1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embedding(tokens)
        x = self.pre_attention(x)
        x = self.conv_bn(self.conv(x.transpose(1, 2))).transpose(1, 2)
        x = self.post_attention(F.relu(x))
        pooled = torch.amax(x, dim=1)
        return self.classifier(pooled).squeeze(-1)


class PlainCNNDiscriminator(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, conv_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.conv = nn.Conv1d(embed_dim, conv_dim, kernel_size=3, padding=1)
        self.classifier = nn.Linear(conv_dim, 1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embedding(tokens).transpose(1, 2)
        x = F.relu(self.conv(x)).transpose(1, 2)
        pooled = torch.amax(x, dim=1)
        return self.classifier(pooled).squeeze(-1)


class RankDiscriminator(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, conv_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.conv = nn.Conv1d(embed_dim, conv_dim, kernel_size=3, padding=1)
        self.proj = nn.Linear(conv_dim, conv_dim)
        self.scale = math.sqrt(float(conv_dim))

    def encode(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embedding(tokens).transpose(1, 2)
        x = F.relu(self.conv(x)).transpose(1, 2)
        x = torch.amax(x, dim=1)
        return F.normalize(self.proj(x), dim=-1)

    def score(self, tokens: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        ref = self.encode(reference).mean(dim=0, keepdim=True)
        ref = F.normalize(ref, dim=-1)
        return torch.sum(self.encode(tokens) * ref, dim=-1) * self.scale


def _device(cfg: dict) -> torch.device:
    use_cuda = bool(cfg["seqgan"].get("use_cuda", True))
    return torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")


def _edges(values: np.ndarray, bins: int, lo: float | None = None) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    low = float(np.min(arr) if lo is None else lo)
    high = float(np.max(arr))
    pad = max((high - low) * 0.05, 1.0e-3)
    return np.linspace(low, high + pad, int(bins) + 1, dtype=np.float32)


def _centers(edges: np.ndarray) -> np.ndarray:
    return ((edges[:-1] + edges[1:]) * 0.5).astype(np.float32)


def _encode(sequences: np.ndarray, y_edges: np.ndarray, vx_edges: np.ndarray) -> np.ndarray:
    y_bins = len(y_edges) - 1
    vx_bins = len(vx_edges) - 1
    y_idx = np.clip(np.digitize(sequences[:, :, 0], y_edges) - 1, 0, y_bins - 1)
    vx_idx = np.clip(np.digitize(sequences[:, :, 1], vx_edges) - 1, 0, vx_bins - 1)
    return (y_idx * vx_bins + vx_idx).astype(np.int64)


def _decode(tokens: np.ndarray, y_edges: np.ndarray, vx_edges: np.ndarray) -> np.ndarray:
    vx_bins = len(vx_edges) - 1
    y = _centers(y_edges)[tokens // vx_bins]
    vx = _centers(vx_edges)[tokens % vx_bins]
    return np.stack([y, vx], axis=-1).astype(np.float32)


def _batches(tokens: torch.Tensor, batch_size: int, shuffle: bool = True) -> list[torch.Tensor]:
    order = torch.randperm(tokens.shape[0], device=tokens.device) if shuffle else torch.arange(tokens.shape[0], device=tokens.device)
    return [tokens[order[i:i + batch_size]] for i in range(0, len(order), batch_size) if len(order[i:i + batch_size]) > 1]


def _pretrain_generator(
    generator: BNSeqGenerator,
    tokens: torch.Tensor,
    cfg: dict,
) -> list[float]:
    opt = torch.optim.Adam(generator.parameters(), lr=float(cfg["seqgan"]["generator_lr"]))
    losses = []
    for _ in range(int(cfg["seqgan"]["pretrain_epochs"])):
        epoch = []
        generator.train()
        for batch in _batches(tokens, int(cfg["seqgan"]["batch_size"])):
            logits = generator(batch)
            loss = F.cross_entropy(logits.reshape(-1, generator.vocab_size), batch.reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch.append(float(loss.detach().cpu()))
        losses.append(float(np.mean(epoch)))
    return losses


def _train_discriminator_epoch(
    discriminator: AMDiscriminator,
    generator: BNSeqGenerator,
    tokens: torch.Tensor,
    opt: torch.optim.Optimizer,
    cfg: dict,
    device: torch.device,
) -> float:
    batch_size = int(cfg["seqgan"]["batch_size"])
    temperature = float(cfg["seqgan"]["temperature"])
    losses = []
    discriminator.train()
    generator.eval()
    for real in _batches(tokens, batch_size):
        with torch.no_grad():
            fake, _ = generator.sample(real.shape[0], device, temperature)
        logit = torch.cat([discriminator(real), discriminator(fake)], dim=0)
        label = torch.cat([torch.ones(real.shape[0], device=device), torch.zeros(fake.shape[0], device=device)])
        loss = F.binary_cross_entropy_with_logits(logit, label)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def _rollout_rewards(
    sequence: torch.Tensor,
    rollout_policy: BNSeqGenerator,
    discriminator: AMDiscriminator,
    cfg: dict,
    device: torch.device,
) -> torch.Tensor:
    rollout_policy.eval()
    discriminator.eval()
    batch, steps = sequence.shape
    rollout_count = int(cfg["seqgan"]["rollout_count"])
    temperature = float(cfg["seqgan"]["temperature"])
    rewards = []
    with torch.no_grad():
        for step in range(steps):
            prefix = sequence[:, :step + 1]
            if step == steps - 1:
                score = torch.sigmoid(discriminator(sequence))
            else:
                expanded = prefix.repeat_interleave(rollout_count, dim=0)
                completed = rollout_policy.complete_from_prefix(expanded, device, temperature)
                score = torch.sigmoid(discriminator(completed)).reshape(batch, rollout_count).mean(dim=1)
            rewards.append(score)
    return torch.stack(rewards, dim=1)


def _adversarial_train(
    generator: BNSeqGenerator,
    discriminator: AMDiscriminator,
    tokens: torch.Tensor,
    cfg: dict,
    device: torch.device,
) -> tuple[list[float], list[float], list[float]]:
    g_opt = torch.optim.Adam(generator.parameters(), lr=float(cfg["seqgan"]["generator_lr"]) * 0.5)
    d_opt = torch.optim.Adam(discriminator.parameters(), lr=float(cfg["seqgan"]["discriminator_lr"]))
    rollout_policy = copy.deepcopy(generator).to(device)
    rollout_policy.eval()
    g_losses = []
    d_losses = []
    reward_means = []
    batch_size = int(cfg["seqgan"]["batch_size"])
    temperature = float(cfg["seqgan"]["temperature"])
    for _ in range(int(cfg["seqgan"]["adversarial_epochs"])):
        for _ in range(int(cfg["seqgan"]["generator_steps"])):
            generator.train()
            discriminator.eval()
            fake, log_prob = generator.sample(batch_size, device, temperature)
            reward = _rollout_rewards(fake, rollout_policy, discriminator, cfg, device)
            loss = -(log_prob * reward).sum(dim=1).mean()
            g_opt.zero_grad()
            loss.backward()
            g_opt.step()
            g_losses.append(float(loss.detach().cpu()))
            reward_means.append(float(reward.mean().detach().cpu()))
        for _ in range(int(cfg["seqgan"]["discriminator_steps"])):
            d_losses.append(_train_discriminator_epoch(discriminator, generator, tokens, d_opt, cfg, device))
        rollout_policy.load_state_dict(generator.state_dict())
    return g_losses, d_losses, reward_means


def _rank_batches(tokens: torch.Tensor, batch_size: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    order = torch.randperm(tokens.shape[0], device=tokens.device)
    batches = []
    for i in range(0, len(order), batch_size):
        idx = order[i:i + batch_size]
        if len(idx) < 2:
            continue
        ref_idx = torch.roll(idx, shifts=1)
        batches.append((tokens[idx], tokens[ref_idx]))
    return batches


def _train_ranker_epoch(
    ranker: RankDiscriminator,
    generator: PlainSeqGenerator,
    tokens: torch.Tensor,
    opt: torch.optim.Optimizer,
    cfg: dict,
    device: torch.device,
) -> float:
    batch_size = int(cfg["seqgan"]["batch_size"])
    temperature = float(cfg["seqgan"]["temperature"])
    losses = []
    ranker.train()
    generator.eval()
    for real, reference in _rank_batches(tokens, batch_size):
        with torch.no_grad():
            fake, _ = generator.sample(real.shape[0], device, temperature)
        real_score = ranker.score(real, reference)
        fake_score = ranker.score(fake, reference)
        loss = F.softplus(-(real_score - fake_score)).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def _rank_rollout_rewards(
    sequence: torch.Tensor,
    rollout_policy: PlainSeqGenerator,
    ranker: RankDiscriminator,
    reference: torch.Tensor,
    cfg: dict,
    device: torch.device,
) -> torch.Tensor:
    rollout_policy.eval()
    ranker.eval()
    batch, steps = sequence.shape
    rollout_count = int(cfg["seqgan"]["rollout_count"])
    temperature = float(cfg["seqgan"]["temperature"])
    rewards = []
    with torch.no_grad():
        for step in range(steps):
            prefix = sequence[:, :step + 1]
            if step == steps - 1:
                score = ranker.score(sequence, reference)
            else:
                expanded = prefix.repeat_interleave(rollout_count, dim=0)
                completed = rollout_policy.complete_from_prefix(expanded, device, temperature)
                ref = reference.repeat_interleave(max(rollout_count, 1), dim=0)
                if ref.shape[0] != completed.shape[0]:
                    ref = reference.repeat((math.ceil(completed.shape[0] / reference.shape[0]), 1))[:completed.shape[0]]
                score = ranker.score(completed, ref).reshape(batch, rollout_count).mean(dim=1)
            rewards.append(torch.sigmoid(score))
    return torch.stack(rewards, dim=1)


def _rankgan_adversarial_train(
    generator: PlainSeqGenerator,
    ranker: RankDiscriminator,
    tokens: torch.Tensor,
    cfg: dict,
    device: torch.device,
) -> tuple[list[float], list[float], list[float]]:
    g_opt = torch.optim.Adam(generator.parameters(), lr=float(cfg["seqgan"]["generator_lr"]) * 0.5)
    r_opt = torch.optim.Adam(ranker.parameters(), lr=float(cfg["seqgan"]["discriminator_lr"]))
    rollout_policy = copy.deepcopy(generator).to(device)
    rollout_policy.eval()
    g_losses = []
    r_losses = []
    reward_means = []
    batch_size = int(cfg["seqgan"]["batch_size"])
    temperature = float(cfg["seqgan"]["temperature"])
    for _ in range(int(cfg["seqgan"]["adversarial_epochs"])):
        for _ in range(int(cfg["seqgan"]["generator_steps"])):
            generator.train()
            ranker.eval()
            fake, log_prob = generator.sample(batch_size, device, temperature)
            ref_idx = torch.randint(0, tokens.shape[0], (batch_size,), device=device)
            reference = tokens[ref_idx]
            reward = _rank_rollout_rewards(fake, rollout_policy, ranker, reference, cfg, device)
            loss = -(log_prob * reward).sum(dim=1).mean()
            g_opt.zero_grad()
            loss.backward()
            g_opt.step()
            g_losses.append(float(loss.detach().cpu()))
            reward_means.append(float(reward.mean().detach().cpu()))
        for _ in range(int(cfg["seqgan"]["discriminator_steps"])):
            r_losses.append(_train_ranker_epoch(ranker, generator, tokens, r_opt, cfg, device))
        rollout_policy.load_state_dict(generator.state_dict())
    return g_losses, r_losses, reward_means


def _teacher_nll(generator: BNSeqGenerator, tokens: torch.Tensor) -> float:
    generator.eval()
    with torch.no_grad():
        logits = generator(tokens)
        loss = F.cross_entropy(logits.reshape(-1, generator.vocab_size), tokens.reshape(-1))
    return float(loss.detach().cpu())


def _sample_outputs(
    generator: BNSeqGenerator,
    signs: np.ndarray,
    lateral: np.ndarray,
    conditions: np.ndarray,
    y_edges: np.ndarray,
    vx_edges: np.ndarray,
    config: dict,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gen_cfg = config["generator"]
    rng = np.random.default_rng(int(gen_cfg["seed"]) + 300)
    count = int(gen_cfg["sample_count"])
    template_idx = rng.integers(0, len(conditions), size=count)
    durations = conditions[template_idx, 0].copy()
    durations += rng.normal(0.0, 0.05, size=count).astype(np.float32)
    durations = np.clip(
        durations,
        float(config["data"]["min_lane_change_seconds"]),
        float(config["data"]["max_lane_change_seconds"]),
    )
    sampled_lateral = lateral[template_idx].copy()
    sampled_lateral += rng.normal(0.0, float(gen_cfg.get("lateral_noise", 0.04)), size=count).astype(np.float32)
    sampled_lateral = np.clip(sampled_lateral, 2.8, 4.5)
    sampled_signs = signs[template_idx]

    generator.eval()
    token_chunks = []
    with torch.no_grad():
        for start in range(0, count, int(config["seqgan"]["batch_size"])):
            batch = min(int(config["seqgan"]["batch_size"]), count - start)
            sampled, _ = generator.sample(batch, device, float(config["seqgan"]["temperature"]))
            token_chunks.append(sampled.cpu().numpy())
    tokens = np.concatenate(token_chunks, axis=0)
    trajectories = []
    paper_sequences = []
    for idx in range(count):
        seq = _decode(tokens[idx], y_edges, vx_edges)
        seq = _regularize_sequence(
            seq,
            float(durations[idx]),
            float(sampled_lateral[idx]),
            float(gen_cfg.get("speed_noise", 0.25)),
            rng,
        )
        paper_sequences.append(seq)
        trajectories.append(_states_from_sequence(seq, float(durations[idx]), float(sampled_signs[idx])))
    out_conditions = conditions[template_idx].copy()
    out_conditions[:, 0] = durations
    out_conditions[:, 5] = sampled_lateral * sampled_signs
    return (
        np.stack(trajectories).astype(np.float32),
        np.stack(paper_sequences).astype(np.float32),
        out_conditions.astype(np.float32),
    )


def train_bn_am_seqgan(config: dict) -> dict[str, Path]:
    out = output_dir(config)
    data_path = out / "emergency_cutin_dataset.npz"
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    set_seed(int(config["generator"]["seed"]))
    data = np.load(data_path, allow_pickle=True)
    sequences = data["paper_sequences"].astype(np.float32)
    conditions = data["conditions"].astype(np.float32)
    canonical, signs, lateral = _canonical_sequences(sequences)
    y_edges = _edges(canonical[:, :, 0], int(config["seqgan"]["y_bins"]), lo=0.0)
    vx_edges = _edges(canonical[:, :, 1], int(config["seqgan"]["vx_bins"]))
    token_np = _encode(canonical, y_edges, vx_edges)
    device = _device(config)
    tokens = torch.as_tensor(token_np, dtype=torch.long, device=device)
    vocab_size = int(config["seqgan"]["y_bins"]) * int(config["seqgan"]["vx_bins"])
    generator = BNSeqGenerator(
        vocab_size,
        int(config["seqgan"]["embedding_dim"]),
        int(config["seqgan"]["hidden_dim"]),
        sequences.shape[1],
    ).to(device)
    discriminator = AMDiscriminator(
        vocab_size,
        int(config["seqgan"]["embedding_dim"]),
        int(config["seqgan"]["discriminator_dim"]),
    ).to(device)
    pretrain_losses = _pretrain_generator(generator, tokens, config)
    d_opt = torch.optim.Adam(discriminator.parameters(), lr=float(config["seqgan"]["discriminator_lr"]))
    disc_pretrain_losses = [
        _train_discriminator_epoch(discriminator, generator, tokens, d_opt, config, device)
        for _ in range(int(config["seqgan"]["discriminator_pretrain_epochs"]))
    ]
    adv_g_losses, adv_d_losses, reward_means = _adversarial_train(generator, discriminator, tokens, config, device)
    trajectories, paper_sequences, out_conditions = _sample_outputs(
        generator,
        signs,
        lateral,
        conditions,
        y_edges,
        vx_edges,
        config,
        device,
    )
    model_path = out / "bn_am_seqgan.pt"
    torch.save(
        {
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "y_edges": y_edges,
            "vx_edges": vx_edges,
            "config": config["seqgan"],
            "pretrain_losses": pretrain_losses,
            "disc_pretrain_losses": disc_pretrain_losses,
            "adv_generator_losses": adv_g_losses,
            "adv_discriminator_losses": adv_d_losses,
            "rollout_reward_means": reward_means,
        },
        model_path,
    )
    np.savez_compressed(
        out / "generated_trajectories.npz",
        trajectories=trajectories,
        paper_sequences=paper_sequences,
        conditions=out_conditions,
        method=np.asarray(["bn_am_seqgan"]),
    )
    metrics = {
        "model": "bn_am_seqgan",
        "generated_count": int(len(trajectories)),
        "sequence_points": int(sequences.shape[1]),
        "state": "(lateral_y, longitudinal_vx)",
        "vocab_size": int(vocab_size),
        "teacher_forcing_nll": _teacher_nll(generator, tokens),
        "pretrain_loss_last": float(pretrain_losses[-1]),
        "discriminator_loss_last": float((adv_d_losses or disc_pretrain_losses)[-1]),
        "policy_gradient_loss_last": float(adv_g_losses[-1]) if adv_g_losses else float("nan"),
        "rollout_reward_mean_last": float(reward_means[-1]) if reward_means else float("nan"),
        "rollout_count": int(config["seqgan"]["rollout_count"]),
        "duration_distribution_rmse": _duration_rmse(conditions[:, 0], out_conditions[:, 0]),
        "real_duration_s": finite_summary(conditions[:, 0]),
        "generated_duration_s": finite_summary(out_conditions[:, 0]),
        "real_final_lateral_m": finite_summary(conditions[:, 5]),
        "generated_final_lateral_m": finite_summary(out_conditions[:, 5]),
        "monotonic_lateral_rate": _monotonic_rate(trajectories),
        "x_monotonic_rate": float(np.mean(np.all(np.diff(trajectories[:, :, 0], axis=1) >= -1.0e-3, axis=1))),
    }
    save_json(out / "generation_metrics.json", metrics)
    save_json(
        out / "bn_am_seqgan_stats.json",
        {
            "architecture": {
                "generator": "batch-normalized LSTM token policy",
                "discriminator": "attention-CNN-attention classifier",
                "training": "MLE pretrain + discriminator BCE + SeqGAN Monte Carlo rollout policy-gradient reward",
            },
            "device": str(device),
            "samples": int(sequences.shape[0]),
            "tokens": {
                "y_bins": int(config["seqgan"]["y_bins"]),
                "vx_bins": int(config["seqgan"]["vx_bins"]),
                "vocab_size": int(vocab_size),
            },
            "loss": {
                "pretrain": pretrain_losses,
                "discriminator_pretrain": disc_pretrain_losses,
                "adversarial_generator": adv_g_losses,
                "adversarial_discriminator": adv_d_losses,
                "rollout_reward_mean": reward_means,
            },
        },
    )
    result = {
        "bn_am_seqgan_model": model_path,
        "generated": out / "generated_trajectories.npz",
        "generation_metrics": out / "generation_metrics.json",
        "bn_am_seqgan_stats": out / "bn_am_seqgan_stats.json",
    }
    if bool(config["seqgan"].get("train_vanilla_baseline", False)):
        result.update(
            train_vanilla_seqgan_baseline(
                config,
                tokens,
                signs,
                lateral,
                conditions,
                y_edges,
                vx_edges,
                vocab_size,
                sequences.shape[1],
                device,
            )
        )
    if bool(config["seqgan"].get("train_rankgan_baseline", False)):
        result.update(
            train_rankgan_baseline(
                config,
                tokens,
                signs,
                lateral,
                conditions,
                y_edges,
                vx_edges,
                vocab_size,
                sequences.shape[1],
                device,
            )
        )
    return result


def train_vanilla_seqgan_baseline(
    config: dict,
    tokens: torch.Tensor,
    signs: np.ndarray,
    lateral: np.ndarray,
    conditions: np.ndarray,
    y_edges: np.ndarray,
    vx_edges: np.ndarray,
    vocab_size: int,
    seq_len: int,
    device: torch.device,
) -> dict[str, Path]:
    set_seed(int(config["generator"]["seed"]) + 77)
    out = output_dir(config)
    generator = PlainSeqGenerator(
        vocab_size,
        int(config["seqgan"]["embedding_dim"]),
        int(config["seqgan"]["hidden_dim"]),
        seq_len,
    ).to(device)
    discriminator = PlainCNNDiscriminator(
        vocab_size,
        int(config["seqgan"]["embedding_dim"]),
        int(config["seqgan"]["discriminator_dim"]),
    ).to(device)
    pretrain_losses = _pretrain_generator(generator, tokens, config)
    d_opt = torch.optim.Adam(discriminator.parameters(), lr=float(config["seqgan"]["discriminator_lr"]))
    disc_pretrain_losses = [
        _train_discriminator_epoch(discriminator, generator, tokens, d_opt, config, device)
        for _ in range(int(config["seqgan"]["discriminator_pretrain_epochs"]))
    ]
    adv_g_losses, adv_d_losses, reward_means = _adversarial_train(generator, discriminator, tokens, config, device)
    trajectories, paper_sequences, out_conditions = _sample_outputs(
        generator,
        signs,
        lateral,
        conditions,
        y_edges,
        vx_edges,
        config,
        device,
    )
    model_path = out / "seqgan_baseline.pt"
    torch.save(
        {
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "y_edges": y_edges,
            "vx_edges": vx_edges,
            "config": config["seqgan"],
            "pretrain_losses": pretrain_losses,
            "disc_pretrain_losses": disc_pretrain_losses,
            "adv_generator_losses": adv_g_losses,
            "adv_discriminator_losses": adv_d_losses,
            "rollout_reward_means": reward_means,
        },
        model_path,
    )
    np.savez_compressed(
        out / "seqgan_generated_trajectories.npz",
        trajectories=trajectories,
        paper_sequences=paper_sequences,
        conditions=out_conditions,
        method=np.asarray(["seqgan_baseline"]),
    )
    metrics = {
        "model": "seqgan_baseline",
        "generated_count": int(len(trajectories)),
        "sequence_points": int(seq_len),
        "state": "(lateral_y, longitudinal_vx)",
        "vocab_size": int(vocab_size),
        "teacher_forcing_nll": _teacher_nll(generator, tokens),
        "pretrain_loss_last": float(pretrain_losses[-1]),
        "discriminator_loss_last": float((adv_d_losses or disc_pretrain_losses)[-1]),
        "policy_gradient_loss_last": float(adv_g_losses[-1]) if adv_g_losses else float("nan"),
        "rollout_reward_mean_last": float(reward_means[-1]) if reward_means else float("nan"),
        "rollout_count": int(config["seqgan"]["rollout_count"]),
        "duration_distribution_rmse": _duration_rmse(conditions[:, 0], out_conditions[:, 0]),
        "real_duration_s": finite_summary(conditions[:, 0]),
        "generated_duration_s": finite_summary(out_conditions[:, 0]),
        "real_final_lateral_m": finite_summary(conditions[:, 5]),
        "generated_final_lateral_m": finite_summary(out_conditions[:, 5]),
        "monotonic_lateral_rate": _monotonic_rate(trajectories),
        "x_monotonic_rate": float(np.mean(np.all(np.diff(trajectories[:, :, 0], axis=1) >= -1.0e-3, axis=1))),
    }
    save_json(out / "seqgan_generation_metrics.json", metrics)
    return {
        "seqgan_baseline_model": model_path,
        "seqgan_generated": out / "seqgan_generated_trajectories.npz",
        "seqgan_generation_metrics": out / "seqgan_generation_metrics.json",
    }


def train_rankgan_baseline(
    config: dict,
    tokens: torch.Tensor,
    signs: np.ndarray,
    lateral: np.ndarray,
    conditions: np.ndarray,
    y_edges: np.ndarray,
    vx_edges: np.ndarray,
    vocab_size: int,
    seq_len: int,
    device: torch.device,
) -> dict[str, Path]:
    set_seed(int(config["generator"]["seed"]) + 131)
    out = output_dir(config)
    generator = PlainSeqGenerator(
        vocab_size,
        int(config["seqgan"]["embedding_dim"]),
        int(config["seqgan"]["hidden_dim"]),
        seq_len,
    ).to(device)
    ranker = RankDiscriminator(
        vocab_size,
        int(config["seqgan"]["embedding_dim"]),
        int(config["seqgan"]["discriminator_dim"]),
    ).to(device)
    pretrain_losses = _pretrain_generator(generator, tokens, config)
    r_opt = torch.optim.Adam(ranker.parameters(), lr=float(config["seqgan"]["discriminator_lr"]))
    ranker_pretrain_losses = [
        _train_ranker_epoch(ranker, generator, tokens, r_opt, config, device)
        for _ in range(int(config["seqgan"]["discriminator_pretrain_epochs"]))
    ]
    adv_g_losses, adv_r_losses, reward_means = _rankgan_adversarial_train(generator, ranker, tokens, config, device)
    trajectories, paper_sequences, out_conditions = _sample_outputs(
        generator,
        signs,
        lateral,
        conditions,
        y_edges,
        vx_edges,
        config,
        device,
    )
    model_path = out / "rankgan_baseline.pt"
    torch.save(
        {
            "generator": generator.state_dict(),
            "ranker": ranker.state_dict(),
            "y_edges": y_edges,
            "vx_edges": vx_edges,
            "config": config["seqgan"],
            "pretrain_losses": pretrain_losses,
            "ranker_pretrain_losses": ranker_pretrain_losses,
            "adv_generator_losses": adv_g_losses,
            "adv_ranker_losses": adv_r_losses,
            "rollout_reward_means": reward_means,
        },
        model_path,
    )
    np.savez_compressed(
        out / "rankgan_generated_trajectories.npz",
        trajectories=trajectories,
        paper_sequences=paper_sequences,
        conditions=out_conditions,
        method=np.asarray(["rankgan_baseline"]),
    )
    metrics = {
        "model": "rankgan_baseline",
        "generated_count": int(len(trajectories)),
        "sequence_points": int(seq_len),
        "state": "(lateral_y, longitudinal_vx)",
        "vocab_size": int(vocab_size),
        "teacher_forcing_nll": _teacher_nll(generator, tokens),
        "pretrain_loss_last": float(pretrain_losses[-1]),
        "ranker_loss_last": float((adv_r_losses or ranker_pretrain_losses)[-1]),
        "policy_gradient_loss_last": float(adv_g_losses[-1]) if adv_g_losses else float("nan"),
        "rollout_reward_mean_last": float(reward_means[-1]) if reward_means else float("nan"),
        "rollout_count": int(config["seqgan"]["rollout_count"]),
        "duration_distribution_rmse": _duration_rmse(conditions[:, 0], out_conditions[:, 0]),
        "real_duration_s": finite_summary(conditions[:, 0]),
        "generated_duration_s": finite_summary(out_conditions[:, 0]),
        "real_final_lateral_m": finite_summary(conditions[:, 5]),
        "generated_final_lateral_m": finite_summary(out_conditions[:, 5]),
        "monotonic_lateral_rate": _monotonic_rate(trajectories),
        "x_monotonic_rate": float(np.mean(np.all(np.diff(trajectories[:, :, 0], axis=1) >= -1.0e-3, axis=1))),
    }
    save_json(out / "rankgan_generation_metrics.json", metrics)
    return {
        "rankgan_baseline_model": model_path,
        "rankgan_generated": out / "rankgan_generated_trajectories.npz",
        "rankgan_generation_metrics": out / "rankgan_generation_metrics.json",
    }
