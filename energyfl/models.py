"""Small sequence models: only one client batch and one model occupy the GPU."""

import random, time, os
import numpy as np
import torch
from torch import nn


class Forecaster(nn.Module):
    def __init__(self, kind="gru", hidden=32, lookback=96, features=6):
        super().__init__()
        self.kind = kind
        if kind == "gru":
            self.encoder = nn.GRU(features, hidden, batch_first=True)
            self.head = nn.Linear(hidden, 1)
        else:
            self.encoder = nn.Sequential(
                nn.Flatten(), nn.Linear(lookback * features, hidden), nn.ReLU()
            )
            self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.encoder(x)[0][:, -1] if self.kind == "gru" else self.encoder(x)
        return self.head(h).squeeze(-1)


def setup(seed, device, threads):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA requested but unavailable; use CPU smoke config or install CUDA torch on node"
        )
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    return torch.device(device)


def state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def flat(model):
    return (
        torch.nn.utils.parameters_to_vector(model.parameters())
        .detach()
        .cpu()
        .numpy()
        .copy()
    )


def assign(model, vector):
    torch.nn.utils.vector_to_parameters(
        torch.tensor(
            vector, dtype=torch.float32, device=next(model.parameters()).device
        ),
        model.parameters(),
    )


def fit(model, households, steps, batch_size, lr, seed, device):
    rng = np.random.default_rng(seed)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    # Fresh local optimizer every round makes the FedAvg update and resume state
    # explicit. Centralized sampling is proportional to each household's windows.
    weights = np.array([len(h) for h in households], dtype=float)
    weights /= weights.sum()
    losses = []
    start = time.perf_counter()
    for _ in range(steps):
        hi = rng.choice(len(households), size=batch_size, p=weights)
        xx = []
        yy = []
        for idx in np.unique(hi):
            h = households[idx]
            x, y, _ = h.batch(
                "train", rng.integers(len(h), size=int((hi == idx).sum()))
            )
            xx.append(x)
            yy.append(y)
        x = torch.from_numpy(np.concatenate(xx)).to(device)
        y = torch.from_numpy(np.concatenate(yy)).to(device)
        opt.zero_grad(set_to_none=True)
        loss = nn.functional.mse_loss(model(x), y)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        losses.append(float(loss.detach()))
    if device.type == "cuda":
        torch.cuda.synchronize()
    return {
        "train_loss_normalized": float(np.mean(losses)),
        "measured_train_s": time.perf_counter() - start,
    }


@torch.inference_mode()
def evaluate(
    model,
    households,
    split,
    device,
    batch_size=256,
    local_states=None,
    seasonal=False,
    prediction_sink=None,
):
    model.eval()
    per = []
    for h in households:
        if local_states is not None:
            model.load_state_dict(local_states[h.id])
        errors = []
        for start in range(0, len(h.arr[split]), batch_size):
            x, y, t = h.batch(
                split, np.arange(start, min(start + batch_size, len(h.arr[split])))
            )
            actual = h.inverse(y)
            if seasonal:
                pred = h.arr["raw"][t - 48] + h.arr["raw"][t - 47]
            else:
                pred = h.inverse(model(torch.from_numpy(x).to(device)).cpu().numpy())
            # All methods are evaluated on the same targets. Seasonal inputs use
            # the same causal filled series where original readings are missing.
            if seasonal:
                fallback = (h.arr["x"][t - 48, 0] + h.arr["x"][t - 47, 0]) * h.arr[
                    "std"
                ] + 2 * h.arr["mean"]
                pred = np.where(np.isfinite(pred), pred, fallback)
            if not np.isfinite(pred).all():
                raise FloatingPointError("Non-finite predictions")
            errors.extend((pred - actual).tolist())
            if prediction_sink:
                for j in range(len(t)):
                    prediction_sink(
                        [
                            h.id,
                            int(h.arr["timestamps"][t[j]]),
                            float(actual[j]),
                            float(pred[j]),
                        ]
                    )
        if errors:
            e = np.array(errors)
            per.append(
                {
                    "client": h.id,
                    "n": len(e),
                    "mae_kwh": float(np.abs(e).mean()),
                    "rmse_kwh": float(np.sqrt((e * e).mean())),
                }
            )
    if not per:
        raise ValueError(f"No valid {split} windows")
    return {
        "macro_mae_kwh": float(np.mean([x["mae_kwh"] for x in per])),
        "macro_rmse_kwh": float(np.mean([x["rmse_kwh"] for x in per])),
        "p90_household_mae_kwh": float(np.percentile([x["mae_kwh"] for x in per], 90)),
        "evaluated_clients": len(per),
        "per_household": per,
    }
