from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Make sure we can import the original CausalRCA repo modules when running from
# the workspace root (sys.path would otherwise only include this adapt dir).
_THIS_DIR = Path(__file__).resolve().parent
_CAUSALRCA_ROOT = _THIS_DIR.parent
if str(_CAUSALRCA_ROOT) not in sys.path:
    sys.path.insert(0, str(_CAUSALRCA_ROOT))

try:
    import torch
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.autograd import Variable
except Exception:  # pragma: no cover
    torch = None
    optim = None  # type: ignore
    F = None  # type: ignore
    Variable = None  # type: ignore

try:
    from sknetwork.ranking import PageRank
except Exception:  # pragma: no cover
    PageRank = None


def _pagerank_numpy(adj: np.ndarray, damping: float = 0.85, max_iter: int = 200, tol: float = 1e-10) -> np.ndarray:
    """Lightweight PageRank fallback using NumPy only.

    adj: (n, n) non-negative weighted adjacency matrix where adj[i, j] is weight of edge i->j.
    Returns: (n,) rank scores.
    """

    n = int(adj.shape[0])
    if n <= 0:
        return np.array([], dtype=float)
    if n == 1:
        return np.array([1.0], dtype=float)

    A = np.asarray(adj, dtype=np.float64)
    # PageRank expects non-negative weights; also drop any non-finite values.
    A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
    A[A < 0] = 0.0

    # Build row-stochastic transition matrix.
    row_sum = A.sum(axis=1, keepdims=True)
    M = np.divide(A, row_sum, out=np.zeros_like(A, dtype=np.float64), where=row_sum != 0)

    dangling = (row_sum.reshape(-1) == 0)
    if np.any(dangling):
        M[dangling, :] = 1.0 / n

    M = np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)

    pr = np.full(n, 1.0 / n, dtype=np.float64)
    teleport = (1.0 - float(damping)) / n
    for _ in range(int(max_iter)):
        pr_new = teleport + float(damping) * (M.T @ pr)
        pr_new = np.nan_to_num(pr_new, nan=0.0, posinf=0.0, neginf=0.0)
        s_new = float(pr_new.sum())
        if s_new > 0:
            pr_new = pr_new / s_new
        else:
            pr_new = np.full(n, 1.0 / n, dtype=np.float64)

        if np.linalg.norm(pr_new - pr, ord=1) < float(tol):
            pr = pr_new
            break
        pr = pr_new

    return pr


# Import original CausalRCA modules/config as-is.
# We keep the dependency surface small and avoid editing upstream files.
from config import CONFIG  # type: ignore
from modules import MLPDecoder, MLPEncoder, SEMDecoder, SEMEncoder  # type: ignore
from utils import (  # type: ignore
    A_connect_loss,
    A_positive_loss,
    kl_gaussian_sem,
    matrix_poly,
    nll_gaussian,
    preprocess_adj_new,
    preprocess_adj_new1,
)


@dataclass
class CausalRCAResult:
    adjacency: np.ndarray
    var_scores: Dict[str, float]
    service_scores: Dict[str, float]


def _h_A(A: Any, m: int) -> Any:
    assert torch is not None
    expm_A = matrix_poly(A * A, m)
    return torch.trace(expm_A) - m


def _stau(w: Any, tau: float) -> Any:
    assert torch is not None
    prox_plus = torch.nn.Threshold(0.0, 0.0)
    w1 = prox_plus(torch.abs(w) - tau)
    return torch.sign(w) * w1


def _update_optimizer(optimizer: Any, original_lr: float, c_A: float) -> Tuple[Any, float]:
    # Keep consistent with upstream scripts.
    MAX_LR = 1e-2
    MIN_LR = 1e-4
    estimated_lr = original_lr / (math.log10(c_A) + 1e-10)
    if estimated_lr > MAX_LR:
        lr = MAX_LR
    elif estimated_lr < MIN_LR:
        lr = MIN_LR
    else:
        lr = estimated_lr

    for group in optimizer.param_groups:
        group["lr"] = lr
    return optimizer, lr


def _extract_service_from_var(var_name: str) -> str:
    # var names are like "frontend_avg_duration" or "frontend_calls".
    # service is the prefix before first underscore.
    s = (var_name or "").strip()
    if not s:
        return ""
    if "_" in s:
        return s.split("_", 1)[0]
    return s


def run_causalrca_on_dataframe(
    df,
    *,
    epochs: int = 200,
    graph_threshold: float = 0.3,
    gamma: float = 0.25,
    eta: int = 10,
    k_max_iter: int = 100,
    max_seconds: Optional[float] = None,
    encoder_type: str = "mlp",
    decoder_type: str = "mlp",
    device: str = "cpu",
) -> CausalRCAResult:
    """Run CausalRCA causal discovery + PageRank ranking on a single time-series dataframe.

    df: pandas DataFrame with shape [T, D]. Columns are variables.

    Notes:
    - This is adapted from train_* scripts but returns structured outputs.
    - We keep defaults close to upstream (epochs=500), but expose a knob because per-uuid training can be expensive.
    """

    if torch is None:
        raise RuntimeError("torch is required to run CausalRCA. Please install a compatible torch.")
    if optim is None or Variable is None:
        raise RuntimeError("torch submodules unavailable; please reinstall torch.")

    # help static analyzers
    assert torch is not None
    assert optim is not None
    assert Variable is not None

    torch_ = torch
    Variable_ = Variable

    start_time = time.time()

    # Defensive: ensure numeric matrix
    X = df.astype("float64")
    col_names: List[str] = list(X.columns)

    data_sample_size = int(X.shape[0])
    data_variable_size = int(X.shape[1])
    if data_sample_size <= 1 or data_variable_size <= 1:
        # Not enough signal; return zeros.
        adj = np.zeros((data_variable_size, data_variable_size), dtype=float)
        return CausalRCAResult(adjacency=adj, var_scores={c: 0.0 for c in col_names}, service_scores={})

    # match CONFIG usage
    CONFIG.epochs = int(epochs)
    CONFIG.graph_threshold = float(graph_threshold)
    # CONFIG is a class-like container in the upstream repo; set attribute defensively.
    setattr(CONFIG, "cuda", False)

    # add adjacency matrix A
    adj_A = np.zeros((data_variable_size, data_variable_size))

    if encoder_type == "mlp":
        encoder = MLPEncoder(
            data_variable_size * CONFIG.x_dims,
            CONFIG.x_dims,
            CONFIG.encoder_hidden,
            int(CONFIG.z_dims),
            adj_A,
            batch_size=CONFIG.batch_size,
            do_prob=CONFIG.encoder_dropout,
            factor=(not CONFIG.no_factor),
        ).double()
    elif encoder_type == "sem":
        encoder = SEMEncoder(
            data_variable_size * CONFIG.x_dims,
            CONFIG.encoder_hidden,
            int(CONFIG.z_dims),
            adj_A,
            batch_size=CONFIG.batch_size,
            do_prob=CONFIG.encoder_dropout,
            factor=(not CONFIG.no_factor),
        ).double()
    else:
        raise ValueError(f"unsupported encoder_type={encoder_type}")

    if decoder_type == "mlp":
        decoder = MLPDecoder(
            data_variable_size * CONFIG.x_dims,
            CONFIG.z_dims,
            CONFIG.x_dims,
            encoder,
            data_variable_size=data_variable_size,
            batch_size=CONFIG.batch_size,
            n_hid=CONFIG.decoder_hidden,
            do_prob=float(CONFIG.decoder_dropout[0]) if isinstance(CONFIG.decoder_dropout, tuple) else float(CONFIG.decoder_dropout),
        ).double()
    elif decoder_type == "sem":
        decoder = SEMDecoder(
            data_variable_size * CONFIG.x_dims,
            CONFIG.z_dims,
            2,
            encoder,
            data_variable_size=data_variable_size,
            batch_size=CONFIG.batch_size,
            n_hid=CONFIG.decoder_hidden,
            do_prob=float(CONFIG.decoder_dropout[0]) if isinstance(CONFIG.decoder_dropout, tuple) else float(CONFIG.decoder_dropout),
        ).double()
    else:
        raise ValueError(f"unsupported decoder_type={decoder_type}")

    if device != "cpu":
        raise ValueError("this wrapper currently supports CPU only")

    # keep it simple; upstream supports more.
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=float(CONFIG.lr))

    # constraint params
    c_A = float(CONFIG.c_A)
    lambda_A = float(CONFIG.lambda_A)
    h_tol = float(CONFIG.h_tol)
    h_A_old = float("inf")

    # training data
    train_data = X

    def train_one_epoch(c_A_val: float, lambda_A_val: float) -> Tuple[float, np.ndarray, Any]:
        encoder.train()
        decoder.train()

        optimizer_, lr = _update_optimizer(optimizer, float(CONFIG.lr), c_A_val)
        _ = optimizer_  # silence linter

        data = train_data
        data_tensor = torch_.tensor(data.to_numpy().reshape(data_sample_size, data_variable_size, 1))
        data_tensor = Variable_(data_tensor).double()

        optimizer.zero_grad()
        enc_x, logits, origin_A, adj_A_tilt_encoder, z_gap, z_positive, myA, Wa = encoder(data_tensor)
        edges = logits
        _, output, _ = decoder(data_tensor, edges, data_variable_size * CONFIG.x_dims, origin_A, adj_A_tilt_encoder, Wa)

        target = data_tensor
        preds = output
        variance = 0.0

        loss_nll = nll_gaussian(preds, target, variance)
        loss_kl = kl_gaussian_sem(logits)
        loss = loss_kl + loss_nll

        one_adj_A = origin_A
        sparse_loss = CONFIG.tau_A * torch_.sum(torch_.abs(one_adj_A))

        if CONFIG.use_A_connect_loss:
            connect_gap = A_connect_loss(one_adj_A, CONFIG.graph_threshold, z_gap)
            loss += lambda_A_val * connect_gap + 0.5 * c_A_val * connect_gap * connect_gap

        if CONFIG.use_A_positiver_loss:
            positive_gap = A_positive_loss(one_adj_A, z_positive)
            loss += 0.1 * (lambda_A_val * positive_gap + 0.5 * c_A_val * positive_gap * positive_gap)

        h_A = _h_A(origin_A, data_variable_size)
        loss = (
            loss
            + lambda_A_val * h_A
            + 0.5 * c_A_val * h_A * h_A
            + 100.0 * torch_.trace(origin_A * origin_A)
            + sparse_loss
        )

        loss.backward()
        optimizer.step()

        # shrink
        myA.data = _stau(myA.data, float(CONFIG.tau_A) * lr)

        graph = origin_A.data.clone().cpu().numpy()
        graph[np.abs(graph) < float(CONFIG.graph_threshold)] = 0
        return float(loss.item()), graph, origin_A

    step_k = 0
    origin_A = None
    try:
        for step_k in range(int(k_max_iter)):
            if max_seconds is not None and (time.time() - start_time) > float(max_seconds):
                break

            while c_A < 1e20:
                for _epoch in range(int(CONFIG.epochs)):
                    if max_seconds is not None and (time.time() - start_time) > float(max_seconds):
                        break
                    loss, graph, origin_A = train_one_epoch(c_A, lambda_A)

                if origin_A is None:
                    break

                A_new = origin_A.data.clone()
                h_A_new = _h_A(A_new, data_variable_size)
                if h_A_new.item() > float(gamma) * h_A_old:
                    c_A *= int(eta)
                else:
                    break

            if origin_A is None:
                break

            h_A_old = float(_h_A(origin_A.data.clone(), data_variable_size).item())
            lambda_A += c_A * h_A_old

            if h_A_old <= h_tol:
                break

    except KeyboardInterrupt:
        pass

    if origin_A is None:
        adj = np.zeros((data_variable_size, data_variable_size), dtype=float)
    else:
        adj = origin_A.data.clone().cpu().numpy()

    # Upstream code thresholds multiple times; keep a single threshold here.
    adj[np.abs(adj) < float(graph_threshold)] = 0.0

    mat = np.abs(adj.T)
    if PageRank is None:
        scores = _pagerank_numpy(mat)
    else:
        pagerank = PageRank()
        # scikit-network API differs across versions:
        # - some versions expose fit_transform
        # - others expose fit + scores_
        try:
            scores = pagerank.fit_transform(mat)
        except AttributeError:
            pagerank.fit(mat)
            scores = pagerank.scores_
    scores = np.asarray(scores, dtype=np.float64)
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    var_scores: Dict[str, float] = {col_names[i]: float(scores[i]) for i in range(len(col_names))}

    service_scores: Dict[str, float] = {}
    for var, sc in var_scores.items():
        svc = _extract_service_from_var(var)
        if not svc:
            continue
        # aggregate by max score (works well when multiple metrics per svc)
        prev = service_scores.get(svc)
        if prev is None or sc > prev:
            service_scores[svc] = sc

    return CausalRCAResult(adjacency=adj, var_scores=var_scores, service_scores=service_scores)
