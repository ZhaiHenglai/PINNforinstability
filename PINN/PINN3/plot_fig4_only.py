import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge

"""
plot_fig4_only.py

Fig.4: Physics consistency via PDE residual fields for brine–CO2 two-phase flow PINN.

Outputs (by default)
--------------------
- 5 times (quantiles): 0%, 25%, 50%, 75%, 100%
- Per-time figures (3 per time):
    Fig4_rw_t{i}_...png   : log10(|r_w|)
    Fig4_rc_t{i}_...png   : log10(|r_c|)
    Fig4_rsum_t{i}_...png : log10(|r_w| + |r_c|)
- Mosaic summary figures (paper-friendly):
    Fig4_rw_mosaic.png
    Fig4_rc_mosaic.png
    Fig4_rsum_mosaic.png
- Metrics CSV:
    Fig4_metrics.csv  (mean and 95th percentile of abs residuals)

Important implementation notes
------------------------------
1) Residual needs first/second derivatives -> we use torch.autograd.grad(create_graph=True).
2) rho_CO2(p) is a differentiable Chebyshev surrogate.
   The surrogate coefficients are not stored in the model weights, so we refit them from dataset pack
   using random paired samples (p, rho_c) (default eos_sample=300k).
3) Inputs to the PINN are DIMENSIONLESS: x/L_ref, y/L_ref, t/T_ref and must have requires_grad=True
   for residual evaluation.
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
        # This buffer is saved in state_dict; trained B restored by load_state_dict.
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
        self.beta = 1.0  # NOT in state_dict

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
        print("[WARN] Missing keys:", missing)
    if unexpected:
        print("[WARN] Unexpected keys:", unexpected)
    model.set_beta(beta_plot)
    model.eval()
    return model


# =============================================================================
# (C) Dataset pack utilities (time_unique + arrays)
# =============================================================================
def load_time_unique(time_unique_npy: str | None, data_pt: str):
    if time_unique_npy and os.path.exists(time_unique_npy):
        tu = np.load(time_unique_npy)
        tu = np.asarray(tu, dtype=np.float64)
        tu = np.unique(tu)
        tu.sort()
        print(f"Loaded time_unique from {time_unique_npy} (n={tu.size})")
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
    tu = np.unique(tu)
    tu.sort()

    out = "time_unique.npy"
    np.save(out, tu)
    print(f"Extracted time_unique from dataset pack and saved: {out} (n={tu.size})")
    return tu


def pick_times_from_unique(time_unique, quantiles):
    tu = np.asarray(time_unique, dtype=np.float64)
    tu = np.unique(tu)
    tu.sort()
    if tu.size == 0:
        raise ValueError("time_unique is empty.")

    idx_sel = []
    for q in quantiles:
        q = float(q)
        q = min(max(q, 0.0), 1.0)
        j = int(round(q * (tu.size - 1)))
        if j not in idx_sel:
            idx_sel.append(j)

    t_sel = [float(tu[j]) for j in idx_sel]
    pct_sel = [float(j / (tu.size - 1)) * 100.0 if tu.size > 1 else 0.0 for j in idx_sel]
    return t_sel, pct_sel


def load_dataset_arrays(data_pt: str):
    pack = torch.load(data_pt, map_location="cpu")
    if not isinstance(pack, dict) or "arrays" not in pack:
        raise ValueError(f"{data_pt} is not a valid dataset pack (missing 'arrays').")
    arrays = pack["arrays"]
    for k in ["p", "rho_c"]:
        if k not in arrays:
            raise ValueError(f"Dataset pack missing arrays['{k}'] (needed for EOS fit)")
    return arrays


def _sample_pair_numpy(p, rho, n_sample: int, seed: int = 0):
    """
    Sample matched (p, rho) pairs without loading all rows into float64.
    Accepts torch Tensor or numpy.
    """
    if isinstance(p, torch.Tensor):
        n = int(p.numel())
    else:
        n = int(np.asarray(p).size)

    n_eff = min(n, int(n_sample))

    if isinstance(p, torch.Tensor):
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)
        idx = torch.randint(0, n, (n_eff,), generator=g, device="cpu", dtype=torch.int64)
        p_s = p.detach().cpu().view(-1).index_select(0, idx).to(torch.float64).numpy()
        rho_s = rho.detach().cpu().view(-1).index_select(0, idx).to(torch.float64).numpy()
        return p_s, rho_s

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=n_eff)
    p_np = np.asarray(p).reshape(-1).astype(np.float64, copy=False)
    rho_np = np.asarray(rho).reshape(-1).astype(np.float64, copy=False)
    return p_np[idx], rho_np[idx]


def compress_by_pressure(p_pa: np.ndarray, rho: np.ndarray):
    """Average duplicate pressures -> a clean p->rho curve."""
    dtmp = pd.DataFrame({"p": p_pa, "rho": rho})
    g = dtmp.groupby("p", as_index=False)["rho"].mean().sort_values("p")
    p_u = g["p"].to_numpy(dtype=np.float64)
    rho_u = g["rho"].to_numpy(dtype=np.float64)
    return p_u, rho_u


def fit_rho_cheb(
    p_pa: np.ndarray, rho: np.ndarray,
    deg_list=(10, 12, 14, 16, 18, 20),
    n_val=20000, seed=0
):
    """
    Fit rho(p) using Chebyshev series. Returns coeffs, rho_ref, pmin, pmax.
    """
    rng = np.random.default_rng(seed)

    p_u, rho_u = compress_by_pressure(p_pa, rho)
    pmin = float(np.min(p_u))
    pmax = float(np.max(p_u))
    pmid = 0.5 * (pmin + pmax)
    prng = 0.5 * (pmax - pmin)

    rho_ref = float(np.interp(pmid, p_u, rho_u))
    rho_tilde = rho_u / rho_ref
    z = (p_u - pmid) / prng

    pv = rng.uniform(pmin, pmax, size=n_val)
    rho_true = np.interp(pv, p_u, rho_u)
    zv = (pv - pmid) / prng

    best = None
    for deg in deg_list:
        coeffs = np.polynomial.chebyshev.chebfit(z, rho_tilde, deg=deg)
        rho_pred = rho_ref * np.polynomial.chebyshev.chebval(zv, coeffs)
        rel = np.abs((rho_pred - rho_true) / rho_true)
        max_rel = float(np.max(rel))
        rms_rel = float(np.sqrt(np.mean(rel ** 2)))
        if best is None or max_rel < best[0]:
            best = (max_rel, rms_rel, deg, coeffs, rho_ref, pmin, pmax)

    max_rel, rms_rel, deg, coeffs, rho_ref, pmin, pmax = best
    print(f"[EOS] Cheb deg={deg}, max_rel={max_rel:.3e}, rms_rel={rms_rel:.3e}, "
          f"range=[{pmin/1e6:.3f},{pmax/1e6:.3f}] MPa (fit sample={len(p_pa)})")
    return coeffs, rho_ref, pmin, pmax


class ChebRhoCO2(nn.Module):
    """Differentiable CO2 density surrogate rho(p) via Chebyshev polynomials (Clenshaw)."""
    def __init__(self, coeffs_np, rho_ref, pmin, pmax, eps=1e-12):
        super().__init__()
        self.register_buffer("c", torch.tensor(coeffs_np, dtype=torch.float64))
        self.rho_ref = float(rho_ref)
        self.pmin = float(pmin)
        self.pmax = float(pmax)
        self.eps = float(eps)
        self.pmid = 0.5 * (self.pmin + self.pmax)
        self.prng = 0.5 * (self.pmax - self.pmin)

    def _chebval_clenshaw(self, z):
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
        p_pa = p_pa.to(dtype=torch.float64)
        z = (p_pa - self.pmid) / self.prng
        z = torch.clamp(z, -1.0 + self.eps, 1.0 - self.eps)
        rho_tilde = self._chebval_clenshaw(z)
        return self.rho_ref * rho_tilde


# =============================================================================
# (D) Residual components (same as training)
# =============================================================================
def relperm_from_Sco2(Sco2: torch.Tensor):
    Sw = 1.0 - Sco2
    denom = 1.0 - Sw_irr - Snr
    denom_t = torch.tensor(denom, device=Sco2.device, dtype=Sco2.dtype)

    Se_w = (Sw - Sw_irr) / denom_t
    Se_c = (Sco2 - Snr) / denom_t
    Se_w = torch.clamp(Se_w, 0.0, 1.0)
    Se_c = torch.clamp(Se_c, 0.0, 1.0)

    krw = krw0 * Se_w ** nw
    krc = krc0 * Se_c ** nc
    return krw, krc


def grad(u, x):
    return torch.autograd.grad(
        u, x,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True,
    )[0]


def divergence(fx, fy, x, y):
    return grad(fx, x) + grad(fy, y)


def laplacian(u, x, y):
    ux = grad(u, x)
    uy = grad(u, y)
    uxx = grad(ux, x)
    uyy = grad(uy, y)
    return uxx + uyy


def pde_residual(model, rho_co2_model, rho_c_ref,
                 x_t, y_t, t_t, eps=0.0):
    """
    Compute residuals r_w and r_c at given points (dimensionless inputs).
    This matches your training script logic.
    """
    p_tilde, Sco2 = model(x_t, y_t, t_t)
    Sw = 1.0 - Sco2

    p_phys = p0 + P_ref * p_tilde
    rho_c = rho_co2_model(p_phys)
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


# =============================================================================
# (E) Grid + mask
# =============================================================================
def make_grid_phys(nx, ny, L):
    x = np.linspace(0.0, L, nx)
    y = np.linspace(0.0, L, ny)
    X, Y = np.meshgrid(x, y, indexing="xy")
    return X, Y


def mask_outside_wells(X, Y):
    cx1, cy1 = float(inj_center[0]), float(inj_center[1])
    cx2, cy2 = float(out_center[0]), float(out_center[1])
    R = float(r_well)
    inside = ((X - cx1) ** 2 + (Y - cy1) ** 2 < R**2) | ((X - cx2) ** 2 + (Y - cy2) ** 2 < R**2)
    return ~inside


def add_hole_outlines(ax):
    ax.add_patch(Wedge(inj_center, float(r_well), 0, 90, fill=False, edgecolor="k", linewidth=1.0))
    ax.add_patch(Wedge(out_center, float(r_well), 180, 270, fill=False, edgecolor="k", linewidth=1.0))


@torch.no_grad()
def _dummy():
    return


def compute_residual_on_grid(model, rho_co2_model, rho_c_ref,
                             X_phys, Y_phys, t_phys,
                             eps, log_eps):
    """
    Compute rw, rc, rsum on a grid (returns numpy arrays, masked with NaN in holes).
    log_eps is a small number added inside log10 to avoid log(0).
    """
    mask_ok = mask_outside_wells(X_phys, Y_phys)

    # Dimensionless inputs, require_grad=True for autograd derivatives.
    x_t = (torch.tensor(X_phys.reshape(-1, 1), device=DEVICE, dtype=torch.float64) / L_ref).requires_grad_(True)
    y_t = (torch.tensor(Y_phys.reshape(-1, 1), device=DEVICE, dtype=torch.float64) / L_ref).requires_grad_(True)
    t_t = torch.full_like(x_t, float(t_phys) / T_ref).requires_grad_(True)

    # Residual (autograd graph created internally)
    r_w, r_c = pde_residual(model, rho_co2_model, rho_c_ref, x_t, y_t, t_t, eps=eps)

    rw = r_w.detach().cpu().numpy().reshape(X_phys.shape)
    rc = r_c.detach().cpu().numpy().reshape(X_phys.shape)
    rsum = np.abs(rw) + np.abs(rc)

    rw = np.where(mask_ok, rw, np.nan)
    rc = np.where(mask_ok, rc, np.nan)
    rsum = np.where(mask_ok, rsum, np.nan)

    # log fields
    lrw = np.log10(np.abs(rw) + log_eps)
    lrc = np.log10(np.abs(rc) + log_eps)
    lrs = np.log10(np.abs(rsum) + log_eps)

    # free memory early (important on GPU)
    del x_t, y_t, t_t, r_w, r_c
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    return lrw, lrc, lrs, rw, rc, rsum


# =============================================================================
# (F) Plot functions (per-time + mosaic)
# =============================================================================
def plot_one_field(field, title, out_png, L, vmin, vmax):
    fig, ax = plt.subplots(figsize=(6.2, 5.5), constrained_layout=True)
    im = ax.imshow(field, origin="lower", extent=[0, L, 0, L], vmin=vmin, vmax=vmax, aspect="equal")
    add_hole_outlines(ax)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title)
    cb = fig.colorbar(im, ax=ax, shrink=0.95, pad=0.02)
    cb.set_label("log10(residual magnitude)")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png}")


def plot_mosaic(fields, titles_top, out_png, L, vmin, vmax):
    """
    Mosaic: 1 row x N cols (for one type of residual)
    fields: list of 2D arrays length N
    titles_top: list of per-column titles length N
    """
    ncols = len(fields)
    fig, axes = plt.subplots(1, ncols, figsize=(3.2 * ncols, 4.3), constrained_layout=True)
    if ncols == 1:
        axes = [axes]

    im = None
    for j in range(ncols):
        ax = axes[j]
        im = ax.imshow(fields[j], origin="lower", extent=[0, L, 0, L], vmin=vmin, vmax=vmax, aspect="equal")
        add_hole_outlines(ax)
        ax.set_title(titles_top[j])
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")

    cb = fig.colorbar(im, ax=axes, shrink=0.95, pad=0.02)
    cb.set_label("log10(residual magnitude)")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved mosaic: {out_png}")


# =============================================================================
# (G) Main
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="Model weights state_dict .pt")
    ap.add_argument("--data_pt", required=True, help="Dataset pack .pt (for time_unique + EOS fitting)")
    ap.add_argument("--time_unique_npy", default="time_unique.npy", help="time_unique.npy (recommended)")
    ap.add_argument("--beta", type=float, default=BETA_PLOT_DEFAULT, help="Plot beta (default=30)")
    ap.add_argument("--eps", type=float, default=1e-6,
                    help="Artificial diffusion eps used in residual evaluation (default=1e-6 to match late training)")
    ap.add_argument("--log_eps", type=float, default=1e-30,
                    help="Small number added inside log10 to avoid log(0).")
    ap.add_argument("--nx", type=int, default=101, help="Residual grid resolution in x (keep modest; residual needs 2nd derivatives)")
    ap.add_argument("--ny", type=int, default=101, help="Residual grid resolution in y")
    ap.add_argument("--quantiles", default="0.00,0.25,0.50,0.75,1.00",
                    help="Time quantiles (default 5 times)")
    ap.add_argument("--eos_sample", type=int, default=300000,
                    help="Number of (p, rho_c) samples for EOS refit (default 300k).")
    ap.add_argument("--out_prefix", default="Fig4", help="Output prefix")
    args = ap.parse_args()

    # --- load model weights
    model = load_model(args.weights, beta_plot=args.beta)
    print(f"Loaded weights: {args.weights} | beta={args.beta}")

    # --- load time_unique and choose times
    time_unique = load_time_unique(args.time_unique_npy, args.data_pt)
    q_list = [float(s.strip()) for s in args.quantiles.split(",") if s.strip() != ""]
    t_sel, pct_sel = pick_times_from_unique(time_unique, q_list)
    print("Selected times (s):", t_sel)

    # --- load arrays and refit EOS surrogate for plotting residuals
    arrays = load_dataset_arrays(args.data_pt)
    p_s, rho_s = _sample_pair_numpy(arrays["p"], arrays["rho_c"], args.eos_sample, seed=0)
    coeffs, rho_c_ref, pmin, pmax = fit_rho_cheb(p_s, rho_s)
    rho_co2_model = ChebRhoCO2(coeffs, rho_c_ref, pmin, pmax).to(DEVICE)
    rho_co2_model.eval()

    # --- grid
    X, Y = make_grid_phys(args.nx, args.ny, float(L_ref))

    # --- compute residual fields for all selected times
    lrw_list, lrc_list, lrs_list = [], [], []
    metrics_rows = []

    for i, (t_phys, pct) in enumerate(zip(t_sel, pct_sel)):
        lrw, lrc, lrs, rw, rc, rsum = compute_residual_on_grid(
            model, rho_co2_model, rho_c_ref, X, Y, t_phys,
            eps=args.eps, log_eps=args.log_eps
        )

        lrw_list.append(lrw)
        lrc_list.append(lrc)
        lrs_list.append(lrs)

        # stats on abs residuals (finite cells only)
        def stats(a):
            m = np.isfinite(a)
            if not np.any(m):
                return (np.nan, np.nan)
            v = np.abs(a[m])
            return (float(np.mean(v)), float(np.percentile(v, 95)))

        mean_rw, p95_rw = stats(rw)
        mean_rc, p95_rc = stats(rc)
        mean_rs, p95_rs = stats(rsum)

        metrics_rows.append([t_phys, pct, mean_rw, p95_rw, mean_rc, p95_rc, mean_rs, p95_rs])

        # per-time figures (three)
        # For consistent colorbars, we will set vmin/vmax later using global ranges,
        # so here we only store arrays; plotting happens after global ranges computed.
        print(f"[Done] t={t_phys:.6g}s ({pct:.0f}%)  mean|rw|={mean_rw:.3e} p95|rw|={p95_rw:.3e} "
              f"mean|rc|={mean_rc:.3e} p95|rc|={p95_rc:.3e}")

    # --- global color ranges for log-fields (consistent across times)
    vmin_rw = float(np.nanmin(np.stack(lrw_list, axis=0)))
    vmax_rw = float(np.nanmax(np.stack(lrw_list, axis=0)))
    vmin_rc = float(np.nanmin(np.stack(lrc_list, axis=0)))
    vmax_rc = float(np.nanmax(np.stack(lrc_list, axis=0)))
    vmin_rs = float(np.nanmin(np.stack(lrs_list, axis=0)))
    vmax_rs = float(np.nanmax(np.stack(lrs_list, axis=0)))

    # --- per-time plots now with consistent ranges
    for i, (t_phys, pct) in enumerate(zip(t_sel, pct_sel)):
        plot_one_field(
            lrw_list[i],
            title=f"{args.out_prefix}: log10(|r_w|)  t={t_phys:.3g}s ({pct:.0f}%)",
            out_png=f"{args.out_prefix}_rw_t{i}_t{t_phys:.6g}.png",
            L=float(L_ref), vmin=vmin_rw, vmax=vmax_rw
        )
        plot_one_field(
            lrc_list[i],
            title=f"{args.out_prefix}: log10(|r_c|)  t={t_phys:.3g}s ({pct:.0f}%)",
            out_png=f"{args.out_prefix}_rc_t{i}_t{t_phys:.6g}.png",
            L=float(L_ref), vmin=vmin_rc, vmax=vmax_rc
        )
        plot_one_field(
            lrs_list[i],
            title=f"{args.out_prefix}: log10(|r_w|+|r_c|)  t={t_phys:.3g}s ({pct:.0f}%)",
            out_png=f"{args.out_prefix}_rsum_t{i}_t{t_phys:.6g}.png",
            L=float(L_ref), vmin=vmin_rs, vmax=vmax_rs
        )

    # --- mosaic plots (paper-friendly)
    col_titles = [f"{pct:.0f}%\n(t={t:.3g}s)" for t, pct in zip(t_sel, pct_sel)]
    plot_mosaic(lrw_list, col_titles, f"{args.out_prefix}_rw_mosaic.png", float(L_ref), vmin_rw, vmax_rw)
    plot_mosaic(lrc_list, col_titles, f"{args.out_prefix}_rc_mosaic.png", float(L_ref), vmin_rc, vmax_rc)
    plot_mosaic(lrs_list, col_titles, f"{args.out_prefix}_rsum_mosaic.png", float(L_ref), vmin_rs, vmax_rs)

    # --- save metrics CSV
    csv_path = f"{args.out_prefix}_metrics.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("t_seconds,percent,mean_abs_rw,p95_abs_rw,mean_abs_rc,p95_abs_rc,mean_abs_rsum,p95_abs_rsum\n")
        for r in metrics_rows:
            f.write(",".join(str(x) for x in r) + "\n")
    print(f"Saved metrics CSV: {csv_path}")


if __name__ == "__main__":
    main()