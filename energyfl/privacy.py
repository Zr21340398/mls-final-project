"""Research privacy mechanisms; the masking backend is NOT Flower SecAgg+."""

import math, secrets
import numpy as np


def epsilon(releases, sigma, delta):
    if releases == 0:
        return 0.0
    # Gaussian RDP: alpha/(2*sigma^2), where sigma is relative to the
    # replacement sensitivity, NOT relative to the clip bound. No amplification.
    orders = np.concatenate([np.arange(1.1, 10, 0.1), np.arange(10, 256, dtype=float)])
    return float(
        np.min(
            releases * orders / (2 * sigma * sigma) + math.log(1 / delta) / (orders - 1)
        )
    )


def clip(update, bound):
    update = np.asarray(update, dtype=np.float64)
    return update * min(1.0, bound / max(np.linalg.norm(update), 1e-12))


def masked_sum(vectors, seed):
    """Pairwise additive masking correctness simulator for active survivors.

    The orchestrator knows all seeds and vectors. This demonstrates cancellation
    only: no encryption, key exchange, secret sharing or post-mask dropout recovery.
    Simulated dropouts must be resolved BEFORE this function. Never call it SecAgg+.
    """
    masked = [v.astype(np.float64).copy() for v in vectors]
    rng = np.random.default_rng(seed)
    for i in range(len(masked)):
        for j in range(i + 1, len(masked)):
            mask = rng.uniform(-1, 1, size=masked[i].shape)
            masked[i] += mask
            masked[j] -= mask
    return np.sum(masked, axis=0)


def aggregate(updates, weights, scheduled, config, round_seed):
    p = config["privacy"]
    sim = config["simulation"]
    equal = config["train"]["weighting"] == "equal"
    vectors = [
        clip(u, p["clip_norm"])
        if p["enabled"] or p["clip_updates"]
        else u.astype(np.float64)
        for u in updates
    ]
    # Fixed denominator m means missing scheduled updates contribute zero.
    # It is crucial for a stable 2C/m replacement sensitivity under dropout.
    if equal:
        total = (
            masked_sum(vectors, round_seed)
            if sim["aggregation"] == "masking_sim"
            else np.sum(vectors, axis=0)
        )
        result = total / scheduled
    else:
        w = np.asarray(weights, dtype=float)
        w /= w.sum()
        result = np.sum([v * a for v, a in zip(vectors, w)], axis=0)
    noise_std = 0.0
    if p["enabled"]:
        noise_std = p["noise_multiplier"] * 2 * p["clip_norm"] / scheduled
        if p["deterministic_noise"]:
            # Only public-data debugging: known noise seeds invalidate a real DP claim.
            noise = np.random.default_rng(round_seed + 999).normal(
                0, noise_std, size=result.shape
            )
        else:
            rng = secrets.SystemRandom()
            noise = np.fromiter(
                (rng.gauss(0, noise_std) for _ in range(result.size)), dtype=np.float64
            ).reshape(result.shape)
        result += noise
    return result.astype(np.float32), noise_std
