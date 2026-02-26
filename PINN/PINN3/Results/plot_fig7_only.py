import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge

"""
plot_fig7_only.py

Fig.7: Show improvement from Adam -> LBFGS (same architecture, different checkpoints).

What this script produces (all enabled by default)
--------------------------------------------------
(1) RMSE comparison (vs data baseline field):
    - RMSE_p (MPa) and RMSE_S at 5 times (0/25/50/75/100%)
(2) Residual statistics comparison:
    - mean and p95 of |r_w|+|r_c| on randomly sampled interior points (holes excluded)
(3) Side-by-side error maps at a representative time (default 50%):
    - |p error| and |S error| for Adam vs LBFGS (2x2)

Important notes
---------------
- Your .pt files are state_dict weights only, so we recreate TwoNetPINN and load state_dict.
- beta is NOT saved in state_dict -> we set beta=30 (default) for both checkpoints.
- For residual evaluation we need rho_CO2(p) surrogate; coefficients are not stored in weights.
  So we refit a Chebyshev rho(p) from dataset pack arrays['p'] and arrays['rho_c'] using random paired samples.

Big-data friendly
-----------------
- RMSE uses "data baseline" built by binning scattered points into a grid (cell-wise mean),
  using rejection sampling on indices to collect enough points for each time slice (no full scan).
"""

# =============================================================================
# (A) Constants (MUST match training script)
# =============================================================================
torch.set_default_dtype(torch.float64)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

L_ref = 5.0
T_ref = 1.0e5
K = 1.0e-14
phi = 0.2

rho_w_const = 1027.61
mu_w = 2.5e-4
mu_c = 2.25e-5

U_in = 5.4e-5
p0 = 10.0e6
p_out = 10.0e6

r_well = 0.5
inj_center = (0.0, 0.0)
out_center = (5.0, 5.0)

mu_ref = mu_w
U_ref = U_in
k_ref = K

P_ref = mu_ref * U_ref * L_ref / k_ref
p_out_tilde = (p_out - p0) / P_ref
A_time = L_ref / (U_ref * T_ref)

Sw_irr = 0.2
Snr = 0.2
krw0 = 1.0
krc0 = 1.0
nw = 2.0
nc = 2.0

BETA_PLOT_DEFAULT = 30.0


# =============================================================================
# (B) Model definition (MUST match EXACTLY)
# =============================================================================
class FourierFeatures(nn.Module):
    def __init__(self, in_dim=3, m=32, scale=6.0):
        super().__init__()
        B = torch.randn(in_dim, m) * scale
        self.register_buffer("B", B)

    def forward(self, x):
        proj = x @ self.B
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, width=128, depth=6):
        super().__init__()
        layers = [nn.Linear(in_dim, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class TwoNetPINN(nn.Module):
    def __init__(self, width=128, use_fourier=True):
        super().__init__()
        self.use_fourier = use_fourier
        if use_fourier:
            self.ff = FourierFeatures(3, 32, 6.0)
            in_dim = 64
        else:
            in_dim = 3

        self.pnet = MLP(in_dim, 1, width=width, depth=6)
        self.snet = MLP(in_dim, 1, width=width, depth=8)
        self.beta = 1.0

    def set_beta(self, beta: float):
        self.beta = float(beta)

    def forward(self, x_t, y_t, t_t):
        x_in = 2.0 * x_t - 1.0
        y_in = 2.0 * y_t - 1.0
        t_in = 2.0 * t_t - 1.0
        X = torch.cat([x_in, y_in, t_in], dim=1)

        Z = self.ff(X) if self.use_fourier else X
        p_tilde = self.pnet(Z)
        s_logit = self.snet(Z)
        S_hat = torch.sigmoid(self.beta * s_logit)

        Sco2 = Snr + (1.0 - Sw_irr - Snr) * S_hat
        return p_tilde, Sco2


def load_model(weights_path: str, beta_plot: float):
    model = TwoNetPINN(width=128, use_fourier=True).to(DEVICE)
    sd = torch.load(weights_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[WARN] ({weights_path}) Missing keys:", missing)
    if unexpected:
        print(f"[WARN] ({weights_path}) Unexpected keys:", unexpected)
    model.set_beta(beta_plot)
    model.eval()
    return model


# =============================================================================
# (C) time_unique and dataset pack
# =============================================================================
def load_time_unique(time_unique_npy: str | None, data_pt: str):
    if time_unique_npy and os.path.exists(time_unique_npy):
        tu = np.load(time_unique_npy)
        tu = np.asarray(tu, dtype=np.float64)
        tu = np.unique(tu); tu.sort()
        return tu
    pack = torch.load(data_pt, map_location="cpu")
    tu = pack.get("time_unique", None)
    if tu is None:
        t = pack["arrays"]["t"]
        if isinstance(t, torch.Tensor):
            tu = torch.unique(t.detach().cpu()).numpy()
        else:
            tu = np.unique(np.asarray(t))
    if isinstance(tu, torch.Tensor):
        tu = tu.detach().cpu().numpy()
    tu = np.asarray(tu, dtype=np.float64)
    tu = np.unique(tu); tu.sort()
    np.save("time_unique.npy", tu)
    return tu


def pick_times_from_unique(time_unique, quantiles=(0.00, 0.25, 0.50, 0.75, 1.00)):
    tu = np.asarray(time_unique, dtype=np.float64)
    tu = np.unique(tu); tu.sort()
    idx_sel = []
    for q in quantiles:
        j = int(round(float(q) * (tu.size - 1)))
        if j not in idx_sel:
            idx_sel.append(j)
    t_sel = [float(tu[j]) for j in idx_sel]
    pct_sel = [float(j / (tu.size - 1)) * 100.0 if tu.size > 1 else 0.0 for j in idx_sel]
    labels = [f"{p:.0f}%\n({t:.3g}s)" for t, p in zip(t_sel, pct_sel)]
    return t_sel, pct_sel, labels


def load_dataset_arrays(data_pt: str):
    pack = torch.load(data_pt, map_location="cpu")
    if not isinstance(pack, dict) or "arrays" not in pack:
        raise ValueError("Invalid dataset pack .pt (missing 'arrays').")
    arrays = pack["arrays"]
    need = ["x", "y", "t", "p", "Sco2", "rho_c"]
    for k in need:
        if k not in arrays:
            raise ValueError(f"Dataset pack missing arrays['{k}']")
    return arrays


def _to_numpy_1d(a):
    if isinstance(a, torch.Tensor):
        return a.detach().cpu().view(-1).numpy()
    return np.asarray(a).reshape(-1)


# =============================================================================
# (D) Data baseline grid (for RMSE): binning + time-slice sampling
# =============================================================================
def sample_indices_at_time(t_all: np.ndarray, t_target: float, n_target: int,
                           chunk_size: int, max_rounds: int, seed: int):
    rng = np.random.default_rng(seed)
    N = int(t_all.shape[0])
    tol = max(1e-6 * abs(t_target), 1e-6)

    collected = []
    got = 0
    for _ in range(max_rounds):
        m = min(int(chunk_size), N)
        idx = rng.integers(0, N, size=m, endpoint=False)
        mask = np.abs(t_all[idx] - t_target) <= tol
        if np.any(mask):
            take = idx[mask]
            collected.append(take)
            got += int(take.size)
            if got >= n_target:
                break
    if got == 0:
        raise RuntimeError(f"Cannot sample any points at t={t_target}. Increase chunk_size/max_rounds.")
    idx_all = np.concatenate(collected, axis=0)
    return idx_all[:n_target] if idx_all.size > n_target else idx_all


def bin_to_grid_mean(x, y, val, nx, ny, L):
    counts, _, _ = np.histogram2d(x, y, bins=[nx, ny], range=[[0.0, L], [0.0, L]])
    sums, _, _ = np.histogram2d(x, y, bins=[nx, ny], range=[[0.0, L], [0.0, L]], weights=val)
    counts = counts.T
    sums = sums.T
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = sums / counts
    mean[counts == 0] = np.nan
    return mean, counts


def make_grid_centers(nx, ny, L):
    x_edges = np.linspace(0.0, L, nx + 1)
    y_edges = np.linspace(0.0, L, ny + 1)
    x_cent = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_cent = 0.5 * (y_edges[:-1] + y_edges[1:])
    Xc, Yc = np.meshgrid(x_cent, y_cent, indexing="xy")
    return Xc, Yc


def well_hole_mask_outside(X, Y):
    cx1, cy1 = float(inj_center[0]), float(inj_center[1])
    cx2, cy2 = float(out_center[0]), float(out_center[1])
    R = float(r_well)
    inside = ((X - cx1) ** 2 + (Y - cy1) ** 2 < R**2) | ((X - cx2) ** 2 + (Y - cy2) ** 2 < R**2)
    return ~inside


@torch.no_grad()
def predict_on_centers(model, Xc, Yc, t_phys):
    hole_ok = well_hole_mask_outside(Xc, Yc)
    x_t = torch.tensor(Xc.reshape(-1, 1), dtype=torch.float64, device=DEVICE) / float(L_ref)
    y_t = torch.tensor(Yc.reshape(-1, 1), dtype=torch.float64, device=DEVICE) / float(L_ref)
    t_t = torch.full_like(x_t, float(t_phys) / float(T_ref))
    p_tilde, S = model(x_t, y_t, t_t)
    p_phys = float(p0) + float(P_ref) * p_tilde
    p_mpa = (p_phys.detach().cpu().numpy().reshape(Xc.shape)) / 1e6
    S = S.detach().cpu().numpy().reshape(Xc.shape)
    p_mpa = np.where(hole_ok, p_mpa, np.nan)
    S = np.where(hole_ok, S, np.nan)
    return p_mpa, S


def rmse(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if not np.any(m):
        return np.nan
    d = a[m] - b[m]
    return float(np.sqrt(np.mean(d * d)))


# =============================================================================
# (E) Residual stats on random interior points (holes excluded)
# =============================================================================
def _sample_pair_numpy(p, rho, n_sample: int, seed: int = 0):
    if isinstance(p, torch.Tensor):
        n = int(p.numel())
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)
        idx = torch.randint(0, n, (min(n, n_sample),), generator=g, device="cpu", dtype=torch.int64)
        p_s = p.detach().cpu().view(-1).index_select(0, idx).to(torch.float64).numpy()
        rho_s = rho.detach().cpu().view(-1).index_select(0, idx).to(torch.float64).numpy()
        return p_s, rho_s
    rng = np.random.default_rng(seed)
    p_np = np.asarray(p).reshape(-1).astype(np.float64, copy=False)
    rho_np = np.asarray(rho).reshape(-1).astype(np.float64, copy=False)
    idx = rng.integers(0, p_np.size, size=min(p_np.size, n_sample))
    return p_np[idx], rho_np[idx]


def compress_by_pressure(p_pa: np.ndarray, rho: np.ndarray):
    dtmp = pd.DataFrame({"p": p_pa, "rho": rho})
    g = dtmp.groupby("p", as_index=False)["rho"].mean().sort_values("p")
    return g["p"].to_numpy(np.float64), g["rho"].to_numpy(np.float64)


def fit_rho_cheb(p_pa: np.ndarray, rho: np.ndarray,
                 deg_list=(10, 12, 14, 16, 18, 20), n_val=20000, seed=0):
    rng = np.random.default_rng(seed)
    p_u, rho_u = compress_by_pressure(p_pa, rho)
    pmin, pmax = float(np.min(p_u)), float(np.max(p_u))
    pmid = 0.5 * (pmin + pmax)
    prng = 0.5 * (pmax - pmin)

    rho_ref = float(np.interp(pmid, p_u, rho_u))
    z = (p_u - pmid) / prng
    rho_tilde = rho_u / rho_ref

    pv = rng.uniform(pmin, pmax, size=n_val)
    rho_true = np.interp(pv, p_u, rho_u)
    zv = (pv - pmid) / prng

    best = None
    for deg in deg_list:
        coeffs = np.polynomial.chebyshev.chebfit(z, rho_tilde, deg=deg)
        rho_pred = rho_ref * np.polynomial.chebyshev.chebval(zv, coeffs)
        rel = np.abs((rho_pred - rho_true) / rho_true)
        score = float(np.max(rel))
        if best is None or score < best[0]:
            best = (score, deg, coeffs, rho_ref, pmin, pmax)
    _, deg, coeffs, rho_ref, pmin, pmax = best
    print(f"[EOS] Cheb deg={deg}, range=[{pmin/1e6:.3f},{pmax/1e6:.3f}]MPa")
    return coeffs, rho_ref, pmin, pmax


class ChebRhoCO2(nn.Module):
    def __init__(self, coeffs_np, rho_ref, pmin, pmax, eps=1e-12):
        super().__init__()
        self.register_buffer("c", torch.tensor(coeffs_np, dtype=torch.float64))
        self.rho_ref = float(rho_ref)
        self.pmin = float(pmin)
        self.pmax = float(pmax)
        self.eps = float(eps)
        self.pmid = 0.5 * (self.pmin + self.pmax)
        self.prng = 0.5 * (self.pmax - self.pmin)

    def _chebval(self, z):
        c = self.c
        if c.numel() == 1:
            return c[0] + 0.0 * z
        b1 = torch.zeros_like(z)
        b2 = torch.zeros_like(z)
        for a in torch.flip(c[1:], dims=[0]):
            b0 = 2.0 * z * b1 - b2 + a
            b2 = b1
            b1 = b0
        return z * b1 - b2 + c[0]

    def forward(self, p_pa):
        z = (p_pa.to(torch.float64) - self.pmid) / self.prng
        z = torch.clamp(z, -1.0 + self.eps, 1.0 - self.eps)
        return self.rho_ref * self._chebval(z)


def relperm_from_Sco2(Sco2: torch.Tensor):
    Sw = 1.0 - Sco2
    denom = 1.0 - Sw_irr - Snr
    denom_t = torch.tensor(denom, device=Sco2.device, dtype=Sco2.dtype)
    Se_w = torch.clamp((Sw - Sw_irr) / denom_t, 0.0, 1.0)
    Se_c = torch.clamp((Sco2 - Snr) / denom_t, 0.0, 1.0)
    krw = krw0 * Se_w ** nw
    krc = krc0 * Se_c ** nc
    return krw, krc


def grad(u, x):
    return torch.autograd.grad(
        u, x, grad_outputs=torch.ones_like(u),
        create_graph=True, retain_graph=True
    )[0]


def divergence(fx, fy, x, y):
    return grad(fx, x) + grad(fy, y)


def laplacian(u, x, y):
    ux = grad(u, x); uy = grad(u, y)
    uxx = grad(ux, x); uyy = grad(uy, y)
    return uxx + uyy


def pde_residual(model, rho_model, rho_c_ref, x_t, y_t, t_t, eps=1e-6):
    p_tilde, Sco2 = model(x_t, y_t, t_t)
    Sw = 1.0 - Sco2
    p_phys = p0 + P_ref * p_tilde
    rho_c = rho_model(p_phys)
    rho_w = torch.full_like(rho_c, rho_w_const)

    rho_c_tilde = rho_c / rho_c_ref
    rho_w_tilde = rho_w / rho_w_const

    krw, krc = relperm_from_Sco2(Sco2)
    dpdx = grad(p_tilde, x_t)
    dpdy = grad(p_tilde, y_t)

    K_tilde = K / k_ref
    vwx = -K_tilde * (mu_ref / mu_w) * krw * dpdx
    vwy = -K_tilde * (mu_ref / mu_w) * krw * dpdy
    vcx = -K_tilde * (mu_ref / mu_c) * krc * dpdx
    vcy = -K_tilde * (mu_ref / mu_c) * krc * dpdy

    d_rhoS_w_dt = grad(rho_w_tilde * Sw, t_t)
    d_rhoS_c_dt = grad(rho_c_tilde * Sco2, t_t)

    div_w = divergence(rho_w_tilde * vwx, rho_w_tilde * vwy, x_t, y_t)
    div_c = divergence(rho_c_tilde * vcx, rho_c_tilde * vcy, x_t, y_t)

    r_w = phi * A_time * d_rhoS_w_dt + div_w
    r_c = phi * A_time * d_rhoS_c_dt + div_c

    if eps is not None and float(eps) > 0.0:
        diff = float(eps) * laplacian(Sco2, x_t, y_t)
        r_c = r_c - diff
        r_w = r_w + diff

    return r_w, r_c


def sample_interior_points(N, t_phys, oversample=1.3, seed=0):
    """
    Uniformly sample points in [0,L]^2 excluding well holes.
    Returns x_t,y_t,t_t as torch tensors with requires_grad=True.
    """
    rng = np.random.default_rng(seed)
    need = N
    xs, ys = [], []
    while need > 0:
        M = int(np.ceil(need * oversample)) + 128
        x = rng.random(M) * L_ref
        y = rng.random(M) * L_ref
        # reject holes
        d1 = (x - inj_center[0]) ** 2 + (y - inj_center[1]) ** 2
        d2 = (x - out_center[0]) ** 2 + (y - out_center[1]) ** 2
        ok = (d1 >= r_well**2) & (d2 >= r_well**2)
        x_ok = x[ok]; y_ok = y[ok]
        take = min(need, x_ok.size)
        xs.append(x_ok[:take]); ys.append(y_ok[:take])
        need -= take

    x = np.concatenate(xs).reshape(-1, 1)
    y = np.concatenate(ys).reshape(-1, 1)

    x_t = (torch.tensor(x, device=DEVICE, dtype=torch.float64) / L_ref).requires_grad_(True)
    y_t = (torch.tensor(y, device=DEVICE, dtype=torch.float64) / L_ref).requires_grad_(True)
    t_t = torch.full_like(x_t, float(t_phys) / T_ref).requires_grad_(True)
    return x_t, y_t, t_t


def residual_stats_for_model(model, rho_model, rho_c_ref, t_phys, N_res=2000, eps=1e-6, seed=0):
    x_t, y_t, t_t = sample_interior_points(N_res, t_phys, seed=seed)
    r_w, r_c = pde_residual(model, rho_model, rho_c_ref, x_t, y_t, t_t, eps=eps)
    rsum = (r_w.abs() + r_c.abs()).detach().cpu().numpy().reshape(-1)
    rsum = rsum[np.isfinite(rsum)]
    mean = float(np.mean(rsum)) if rsum.size else np.nan
    p95 = float(np.percentile(rsum, 95)) if rsum.size else np.nan
    return mean, p95


# =============================================================================
# (F) Fig7 plots
# =============================================================================
def plot_metric_curves(out_png, times, labels,
                       rmse_p_adam, rmse_p_lb,
                       rmse_s_adam, rmse_s_lb,
                       rmean_adam, rmean_lb,
                       rp95_adam, rp95_lb):
    fig, axes = plt.subplots(2, 1, figsize=(10.8, 8.2), constrained_layout=True, sharex=True)

    ax = axes[0]
    ax.plot(times, rmse_p_adam, label="RMSE_p Adam (MPa)")
    ax.plot(times, rmse_p_lb, label="RMSE_p LBFGS (MPa)")
    ax.plot(times, rmse_s_adam, label="RMSE_S Adam")
    ax.plot(times, rmse_s_lb, label="RMSE_S LBFGS")
    ax.set_ylabel("RMSE")
    ax.set_title("Fig.7  Error metrics: Adam vs LBFGS")
    ax.grid(True, alpha=0.25)
    ax.legend()

    ax = axes[1]
    ax.plot(times, rmean_adam, label="mean(|rw|+|rc|) Adam")
    ax.plot(times, rmean_lb, label="mean(|rw|+|rc|) LBFGS")
    ax.plot(times, rp95_adam, label="p95(|rw|+|rc|) Adam")
    ax.plot(times, rp95_lb, label="p95(|rw|+|rc|) LBFGS")
    ax.set_xlabel("Time t [s]")
    ax.set_ylabel("Residual magnitude")
    ax.set_title("Fig.7  PDE residual stats: Adam vs LBFGS")
    ax.grid(True, alpha=0.25)
    ax.legend()

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png}")


def plot_error_maps(out_png, L, p_err_adam, s_err_adam, p_err_lb, s_err_lb,
                    p_err_vmax=None, s_err_vmax=None):
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.6), constrained_layout=True)

    if p_err_vmax is None:
        p_err_vmax = float(np.nanmax([p_err_adam, p_err_lb]))
    if s_err_vmax is None:
        s_err_vmax = float(np.nanmax([s_err_adam, s_err_lb]))

    im00 = axes[0, 0].imshow(p_err_adam, origin="lower", extent=[0, L, 0, L], vmin=0.0, vmax=p_err_vmax, aspect="equal")
    im01 = axes[0, 1].imshow(s_err_adam, origin="lower", extent=[0, L, 0, L], vmin=0.0, vmax=s_err_vmax, aspect="equal")
    im10 = axes[1, 0].imshow(p_err_lb, origin="lower", extent=[0, L, 0, L], vmin=0.0, vmax=p_err_vmax, aspect="equal")
    im11 = axes[1, 1].imshow(s_err_lb, origin="lower", extent=[0, L, 0, L], vmin=0.0, vmax=s_err_vmax, aspect="equal")

    axes[0, 0].set_title("Adam: |p error| [MPa]")
    axes[0, 1].set_title("Adam: |S error| [-]")
    axes[1, 0].set_title("LBFGS: |p error| [MPa]")
    axes[1, 1].set_title("LBFGS: |S error| [-]")

    for ax in axes.ravel():
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.add_patch(Wedge(inj_center, float(r_well), 0, 90, fill=False, edgecolor="k", linewidth=1.0))
        ax.add_patch(Wedge(out_center, float(r_well), 180, 270, fill=False, edgecolor="k", linewidth=1.0))

    c0 = fig.colorbar(im00, ax=[axes[0, 0], axes[1, 0]], shrink=0.95, pad=0.02)
    c0.set_label("|p error| [MPa]")
    c1 = fig.colorbar(im01, ax=[axes[0, 1], axes[1, 1]], shrink=0.95, pad=0.02)
    c1.set_label("|S error| [-]")

    fig.suptitle("Fig.7  Representative error maps (same time): Adam vs LBFGS", y=1.02)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png}")


# =============================================================================
# (G) Main
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights_adam", required=True, help="Adam-end checkpoint state_dict .pt")
    ap.add_argument("--weights_lbfgs", required=True, help="LBFGS-end checkpoint state_dict .pt")
    ap.add_argument("--data_pt", required=True, help="Dataset pack .pt")
    ap.add_argument("--time_unique_npy", default="time_unique.npy", help="time_unique.npy path")
    ap.add_argument("--beta", type=float, default=BETA_PLOT_DEFAULT, help="Plot beta for BOTH models (default=30)")

    ap.add_argument("--quantiles", default="0.00,0.25,0.50,0.75,1.00", help="5 time quantiles for curves")
    ap.add_argument("--tmap_quantile", type=float, default=0.50, help="Quantile time for error-map panel (default 0.50)")

    # RMSE (data baseline binning)
    ap.add_argument("--nx", type=int, default=180, help="Grid bins in x for data baseline")
    ap.add_argument("--ny", type=int, default=180, help="Grid bins in y for data baseline")
    ap.add_argument("--n_points", type=int, default=250000, help="Data points sampled per time slice")
    ap.add_argument("--chunk_size", type=int, default=2000000, help="Random indices per round")
    ap.add_argument("--max_rounds", type=int, default=200, help="Max rounds for time-slice sampling")
    ap.add_argument("--min_count", type=int, default=3, help="Mask bins with count < min_count")

    # Residual stats
    ap.add_argument("--n_res", type=int, default=2000, help="Interior points per time for residual stats")
    ap.add_argument("--eps_res", type=float, default=1e-6, help="eps used in residual eval (match late training)")
    ap.add_argument("--eos_sample", type=int, default=300000, help="(p,rho) samples for EOS refit")

    ap.add_argument("--out_prefix", default="Fig7", help="Output prefix")
    args = ap.parse_args()

    # Load models
    m_adam = load_model(args.weights_adam, beta_plot=args.beta)
    m_lb = load_model(args.weights_lbfgs, beta_plot=args.beta)
    print("Loaded checkpoints:")
    print("  Adam :", args.weights_adam)
    print("  LBFGS:", args.weights_lbfgs)

    # Times
    tu = load_time_unique(args.time_unique_npy, args.data_pt)
    q_list = [float(s.strip()) for s in args.quantiles.split(",") if s.strip()]
    t_sel, pct_sel, labels = pick_times_from_unique(tu, quantiles=q_list)
    # Representative time for error maps
    t_map, pct_map, _ = pick_times_from_unique(tu, quantiles=(float(args.tmap_quantile),))
    t_map = t_map[0]
    pct_map = pct_map[0]

    # Load dataset arrays for RMSE baseline and EOS refit
    arrays = load_dataset_arrays(args.data_pt)
    x_all = _to_numpy_1d(arrays["x"])
    y_all = _to_numpy_1d(arrays["y"])
    t_all = _to_numpy_1d(arrays["t"])
    p_all = _to_numpy_1d(arrays["p"])
    Sco2_raw_all = _to_numpy_1d(arrays["Sco2"])

    # Grid centers for prediction and masking
    Xc, Yc = make_grid_centers(args.nx, args.ny, float(L_ref))
    hole_ok = well_hole_mask_outside(Xc, Yc)

    # Refit EOS surrogate once (used by residual stats)
    p_s, rho_s = _sample_pair_numpy(arrays["p"], arrays["rho_c"], args.eos_sample, seed=0)
    coeffs, rho_c_ref, pmin, pmax = fit_rho_cheb(p_s, rho_s)
    rho_model = ChebRhoCO2(coeffs, rho_c_ref, pmin, pmax).to(DEVICE)
    rho_model.eval()

    # Storage
    rmse_p_adam, rmse_s_adam = [], []
    rmse_p_lb, rmse_s_lb = [], []
    rmean_adam, rp95_adam = [], []
    rmean_lb, rp95_lb = [], []

    rows = []

    for k, (t_phys, pct) in enumerate(zip(t_sel, pct_sel)):
        # --- Build data baseline field at this time via sampling + binning
        idx = sample_indices_at_time(
            t_all=t_all,
            t_target=float(t_phys),
            n_target=int(args.n_points),
            chunk_size=int(args.chunk_size),
            max_rounds=int(args.max_rounds),
            seed=1234 + 17 * k,
        )
        x = x_all[idx].astype(np.float64, copy=False)
        y = y_all[idx].astype(np.float64, copy=False)
        p_mpa_pts = (p_all[idx].astype(np.float64, copy=False) / 1e6)

        s_raw = np.clip(Sco2_raw_all[idx].astype(np.float64, copy=False), 0.0, 1.0)
        s_pts = Snr + (1.0 - Sw_irr - Snr) * s_raw

        p_data, cnt = bin_to_grid_mean(x, y, p_mpa_pts, args.nx, args.ny, float(L_ref))
        s_data, _ = bin_to_grid_mean(x, y, s_pts, args.nx, args.ny, float(L_ref))

        if args.min_count > 1:
            good = (cnt >= int(args.min_count))
            p_data = np.where(good, p_data, np.nan)
            s_data = np.where(good, s_data, np.nan)

        p_data = np.where(hole_ok, p_data, np.nan)
        s_data = np.where(hole_ok, s_data, np.nan)

        # --- Predictions
        pA, sA = predict_on_centers(m_adam, Xc, Yc, float(t_phys))
        pL, sL = predict_on_centers(m_lb, Xc, Yc, float(t_phys))

        # --- RMSE
        rpA = rmse(pA, p_data); rsA = rmse(sA, s_data)
        rpL = rmse(pL, p_data); rsL = rmse(sL, s_data)

        rmse_p_adam.append(rpA); rmse_s_adam.append(rsA)
        rmse_p_lb.append(rpL); rmse_s_lb.append(rsL)

        # --- Residual stats at same time
        meanA, p95A = residual_stats_for_model(m_adam, rho_model, rho_c_ref, float(t_phys),
                                               N_res=int(args.n_res), eps=float(args.eps_res), seed=777 + 29 * k)
        meanL, p95L = residual_stats_for_model(m_lb, rho_model, rho_c_ref, float(t_phys),
                                               N_res=int(args.n_res), eps=float(args.eps_res), seed=888 + 31 * k)

        rmean_adam.append(meanA); rp95_adam.append(p95A)
        rmean_lb.append(meanL); rp95_lb.append(p95L)

        rows.append([t_phys, pct, rpA, rsA, rpL, rsL, meanA, p95A, meanL, p95L])
        print(f"[t={t_phys:.3g}s {pct:.0f}%] RMSE_p(MPa) Adam={rpA:.3e} LBFGS={rpL:.3e} | "
              f"RMSE_S Adam={rsA:.3e} LBFGS={rsL:.3e} | "
              f"p95(|r|) Adam={p95A:.3e} LBFGS={p95L:.3e}")

        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    # --- Curves plot
    out_curves = f"{args.out_prefix}_metrics_curves.png"
    plot_metric_curves(
        out_curves,
        np.asarray(t_sel, dtype=np.float64),
        labels,
        np.asarray(rmse_p_adam), np.asarray(rmse_p_lb),
        np.asarray(rmse_s_adam), np.asarray(rmse_s_lb),
        np.asarray(rmean_adam), np.asarray(rmean_lb),
        np.asarray(rp95_adam), np.asarray(rp95_lb),
    )

    # --- Representative error maps (uses same data baseline as Fig3 at t_map)
    # Build baseline at t_map
    idx = sample_indices_at_time(
        t_all=t_all, t_target=float(t_map),
        n_target=int(args.n_points),
        chunk_size=int(args.chunk_size),
        max_rounds=int(args.max_rounds),
        seed=4242,
    )
    x = x_all[idx].astype(np.float64, copy=False)
    y = y_all[idx].astype(np.float64, copy=False)
    p_mpa_pts = (p_all[idx].astype(np.float64, copy=False) / 1e6)
    s_raw = np.clip(Sco2_raw_all[idx].astype(np.float64, copy=False), 0.0, 1.0)
    s_pts = Snr + (1.0 - Sw_irr - Snr) * s_raw

    p_data, cnt = bin_to_grid_mean(x, y, p_mpa_pts, args.nx, args.ny, float(L_ref))
    s_data, _ = bin_to_grid_mean(x, y, s_pts, args.nx, args.ny, float(L_ref))
    if args.min_count > 1:
        good = (cnt >= int(args.min_count))
        p_data = np.where(good, p_data, np.nan)
        s_data = np.where(good, s_data, np.nan)
    p_data = np.where(hole_ok, p_data, np.nan)
    s_data = np.where(hole_ok, s_data, np.nan)

    pA, sA = predict_on_centers(m_adam, Xc, Yc, float(t_map))
    pL, sL = predict_on_centers(m_lb, Xc, Yc, float(t_map))

    p_err_adam = np.abs(pA - p_data)
    s_err_adam = np.abs(sA - s_data)
    p_err_lb = np.abs(pL - p_data)
    s_err_lb = np.abs(sL - s_data)

    out_maps = f"{args.out_prefix}_error_maps_tmid.png"
    plot_error_maps(out_maps, float(L_ref), p_err_adam, s_err_adam, p_err_lb, s_err_lb)

    # --- Save CSV metrics
    csv_path = f"{args.out_prefix}_metrics.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("t_seconds,percent,"
                "rmse_p_adam_mpa,rmse_s_adam,rmse_p_lbfgs_mpa,rmse_s_lbfgs,"
                "mean_abs_rsum_adam,p95_abs_rsum_adam,mean_abs_rsum_lbfgs,p95_abs_rsum_lbfgs\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
    
    
#python plot_fig7_only.py \
#  --weights_adam pinn_adam_end.pt \
#  --weights_lbfgs pinn_co2_brine_TwoNet_beta_RAR_eps_frontw_bigdata.pt \
#  --data_pt /home/henglai_pc/pythonTesst/PINN3/tables_cache_tensor.pt \
#  --time_unique_npy time_unique.npy