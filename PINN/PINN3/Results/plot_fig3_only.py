import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge

"""
plot_fig3_only.py  (UPDATED)

Fig.3: Compare PINN prediction vs dataset ("baseline/data") and visualize error fields.

Update in this version
----------------------
1) Default: 5 time snapshots (0%, 25%, 50%, 75%, 100%).
2) Pressure and Saturation are plotted in SEPARATE figures to avoid overcrowding:
   - For each time t:
       Fig3P_t{i}_t{...}.png : [p_data | p_pred | |p_err|]  (1 x 3)
       Fig3S_t{i}_t{...}.png : [S_data | S_pred | |S_err|]  (1 x 3)

Big-data safe sampling strategy
-------------------------------
- Do NOT scan all rows to filter one time slice (too expensive for N ~ 1e8).
- Instead, repeatedly RANDOM-SAMPLE indices and keep those with time == target time.
- Once enough points are collected, stop.

Then we bin scattered points into a uniform grid (cell-wise mean) to create a smooth "data field".
"""

# =============================================================================
# (A) Constants: MUST match your training script
# =============================================================================
L_ref = 5.0          # [m]
T_ref = 1.0e5        # [s]
p0 = 10.0e6          # [Pa]
U_in = 5.4e-5        # [m/s]
K = 1.0e-14          # [m^2]
mu_w = 2.5e-4
mu_ref = mu_w
U_ref = U_in
k_ref = K

P_ref = mu_ref * U_ref * L_ref / k_ref
Sw_irr = 0.2
Snr = 0.2

r_well = 0.5
inj_center = (0.0, 0.0)
out_center = (5.0, 5.0)

# beta is NOT stored in state_dict; use your final beta
BETA_PLOT_DEFAULT = 30.0

torch.set_default_dtype(torch.float64)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# (B) Model definition (must match EXACTLY)
# =============================================================================
class FourierFeatures(nn.Module):
    def __init__(self, in_dim=3, m=32, scale=6.0):
        super().__init__()
        B = torch.randn(in_dim, m) * scale
        # Buffer is saved in state_dict, so trained B will be restored correctly.
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
        # inputs are dimensionless (x/L, y/L, t/T), mapped to [-1,1]
        x_in = 2.0 * x_t - 1.0
        y_in = 2.0 * y_t - 1.0
        t_in = 2.0 * t_t - 1.0
        X = torch.cat([x_in, y_in, t_in], dim=1)

        Z = self.ff(X) if self.use_fourier else X
        p_tilde = self.pnet(Z)
        s_logit = self.snet(Z)
        S_hat = torch.sigmoid(self.beta * s_logit)

        # method B saturation range
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
# (C) time_unique
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
    """
    Pick nearest times on the DISCRETE time grid using quantiles.
    """
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


# =============================================================================
# (D) dataset arrays + time-slice sampling
# =============================================================================
def _to_numpy_1d(arr):
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().view(-1).numpy()
    return np.asarray(arr).reshape(-1)


def load_dataset_arrays(data_pt: str):
    pack = torch.load(data_pt, map_location="cpu")
    if not isinstance(pack, dict) or "arrays" not in pack:
        raise ValueError(f"{data_pt} is not a valid dataset pack (missing 'arrays').")
    arrays = pack["arrays"]
    for k in ["x", "y", "t", "p", "Sco2"]:
        if k not in arrays:
            raise ValueError(f"Dataset pack missing arrays['{k}']")
    return arrays


def sample_indices_at_time(
    t_all: np.ndarray,
    t_target: float,
    n_target: int,
    chunk_size: int,
    max_rounds: int,
    seed: int,
):
    """
    Rejection sampling on indices:
    - Draw random indices, keep those with time ~= t_target.
    - Repeat until we have n_target points.
    """
    rng = np.random.default_rng(seed)
    N = int(t_all.shape[0])

    tol = max(1e-6 * abs(t_target), 1e-6)

    collected = []
    got = 0

    for _ in range(max_rounds):
        m = min(int(chunk_size), N)
        idx = rng.integers(0, N, size=m, endpoint=False)
        tt = t_all[idx]
        mask = np.abs(tt - t_target) <= tol
        if np.any(mask):
            take = idx[mask]
            collected.append(take)
            got += int(take.size)
            if got >= n_target:
                break

    if got == 0:
        raise RuntimeError(
            f"Failed to sample any points at time t={t_target}. "
            f"Try increasing --chunk_size or --max_rounds."
        )

    idx_all = np.concatenate(collected, axis=0)
    if idx_all.size > n_target:
        idx_all = idx_all[:n_target]
    return idx_all


# =============================================================================
# (E) Grid binning utilities
# =============================================================================
def bin_to_grid_mean(x, y, val, nx, ny, L):
    """
    2D binning: mean in each cell.
    Returns grid_mean (ny,nx) and grid_cnt (ny,nx).
    """
    counts, _, _ = np.histogram2d(
        x, y,
        bins=[nx, ny],
        range=[[0.0, L], [0.0, L]]
    )
    sums, _, _ = np.histogram2d(
        x, y,
        bins=[nx, ny],
        range=[[0.0, L], [0.0, L]],
        weights=val
    )

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


# =============================================================================
# (F) Model prediction on grid centers
# =============================================================================
@torch.no_grad()
def predict_on_centers(model, Xc, Yc, t_phys):
    mask_outside = well_hole_mask_outside(Xc, Yc)

    x_t = torch.tensor(Xc.reshape(-1, 1), dtype=torch.float64, device=DEVICE) / float(L_ref)
    y_t = torch.tensor(Yc.reshape(-1, 1), dtype=torch.float64, device=DEVICE) / float(L_ref)
    t_t = torch.full_like(x_t, float(t_phys) / float(T_ref))

    p_tilde, S = model(x_t, y_t, t_t)
    p_phys = float(p0) + float(P_ref) * p_tilde

    p_mpa = (p_phys.detach().cpu().numpy().reshape(Xc.shape)) / 1e6
    S = S.detach().cpu().numpy().reshape(Xc.shape)

    p_mpa = np.where(mask_outside, p_mpa, np.nan)
    S = np.where(mask_outside, S, np.nan)
    return p_mpa, S


# =============================================================================
# (G) Plot helpers: Pressure-only and Saturation-only (each 1x3)
# =============================================================================
def _add_hole_outlines(ax):
    ax.add_patch(Wedge(inj_center, float(r_well), 0, 90, fill=False, edgecolor="k", linewidth=1.0))
    ax.add_patch(Wedge(out_center, float(r_well), 180, 270, fill=False, edgecolor="k", linewidth=1.0))


def plot_pressure_triptych(t_phys, pct, p_data, p_pred, p_err, L, out_png,
                           p_range=None, p_err_range=None, note=None):
    """
    One-row figure:
      [p_data | p_pred | |p_err|]
    """
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.6), constrained_layout=True)

    vmin_p, vmax_p = p_range if p_range else (np.nanmin([p_data, p_pred]), np.nanmax([p_data, p_pred]))
    vmin_e, vmax_e = p_err_range if p_err_range else (0.0, float(np.nanmax(p_err)))

    im0 = axes[0].imshow(p_data, origin="lower", extent=[0, L, 0, L], vmin=vmin_p, vmax=vmax_p, aspect="equal")
    im1 = axes[1].imshow(p_pred, origin="lower", extent=[0, L, 0, L], vmin=vmin_p, vmax=vmax_p, aspect="equal")
    im2 = axes[2].imshow(p_err,  origin="lower", extent=[0, L, 0, L], vmin=vmin_e, vmax=vmax_e, aspect="equal")

    axes[0].set_title("Pressure (data) [MPa]")
    axes[1].set_title("Pressure (PINN) [MPa]")
    axes[2].set_title("|Pressure error| [MPa]")

    for ax in axes:
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        _add_hole_outlines(ax)

    c0 = fig.colorbar(im1, ax=axes[:2], shrink=0.95, pad=0.02)
    c0.set_label("Pressure [MPa]")
    c1 = fig.colorbar(im2, ax=axes[2], shrink=0.95, pad=0.02)
    c1.set_label("|Error| [MPa]")

    title = f"Fig.3 (Pressure)  t={t_phys:.6g} s ({pct:.0f}%)"
    if note:
        title += f"\n{note}"
    fig.suptitle(title, y=1.02)

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Fig.3P] Saved: {out_png}")


def plot_saturation_triptych(t_phys, pct, s_data, s_pred, s_err, L, out_png,
                             s_err_range=None, note=None):
    """
    One-row figure:
      [S_data | S_pred | |S_err|]
    """
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.6), constrained_layout=True)

    vmin_s, vmax_s = float(Snr), float(1.0 - Sw_irr)
    vmin_e, vmax_e = s_err_range if s_err_range else (0.0, float(np.nanmax(s_err)))

    im0 = axes[0].imshow(s_data, origin="lower", extent=[0, L, 0, L], vmin=vmin_s, vmax=vmax_s, aspect="equal")
    im1 = axes[1].imshow(s_pred, origin="lower", extent=[0, L, 0, L], vmin=vmin_s, vmax=vmax_s, aspect="equal")
    im2 = axes[2].imshow(s_err,  origin="lower", extent=[0, L, 0, L], vmin=vmin_e, vmax=vmax_e, aspect="equal")

    axes[0].set_title("Saturation (data) [-]")
    axes[1].set_title("Saturation (PINN) [-]")
    axes[2].set_title("|Saturation error| [-]")

    for ax in axes:
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        _add_hole_outlines(ax)

    c0 = fig.colorbar(im1, ax=axes[:2], shrink=0.95, pad=0.02)
    c0.set_label("Saturation [-]")
    c1 = fig.colorbar(im2, ax=axes[2], shrink=0.95, pad=0.02)
    c1.set_label("|Error| [-]")

    title = f"Fig.3 (Saturation)  t={t_phys:.6g} s ({pct:.0f}%)"
    if note:
        title += f"\n{note}"
    fig.suptitle(title, y=1.02)

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Fig.3S] Saved: {out_png}")


# =============================================================================
# (H) Main
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="Model weights state_dict .pt")
    ap.add_argument("--data_pt", required=True, help="Dataset pack .pt (contains arrays x,y,t,p,Sco2)")
    ap.add_argument("--beta", type=float, default=BETA_PLOT_DEFAULT, help="Plot beta (default=30)")
    ap.add_argument("--time_unique_npy", default="time_unique.npy", help="time_unique.npy (recommended)")

    # Default 5 times: 0, 25, 50, 75, 100 (%)
    ap.add_argument("--quantiles", default="0.00,0.25,0.50,0.75,1.00",
                    help="Time quantiles, default 5 times: '0,0.25,0.5,0.75,1'")

    ap.add_argument("--nx", type=int, default=180, help="Binning grid cells in x")
    ap.add_argument("--ny", type=int, default=180, help="Binning grid cells in y")

    ap.add_argument("--n_points", type=int, default=250000, help="Target sampled points per time slice")
    ap.add_argument("--chunk_size", type=int, default=2000000, help="Random indices per round for time-slice sampling")
    ap.add_argument("--max_rounds", type=int, default=200, help="Max rounds for time-slice sampling")
    ap.add_argument("--min_count", type=int, default=3, help="Mask bins with count < min_count")
    ap.add_argument("--seed", type=int, default=0, help="Random seed")
    ap.add_argument("--out_prefix", default="Fig3", help="Output prefix")
    args = ap.parse_args()

    q_list = [float(s.strip()) for s in args.quantiles.split(",") if s.strip() != ""]
    if len(q_list) == 0:
        raise ValueError("No quantiles provided.")

    model = load_model(args.weights, beta_plot=args.beta)
    print(f"Loaded weights: {args.weights} | beta={args.beta}")

    time_unique = load_time_unique(args.time_unique_npy, args.data_pt)
    t_sel, pct_sel = pick_times_from_unique(time_unique, q_list)
    print("Selected times (s):", t_sel)

    arrays = load_dataset_arrays(args.data_pt)
    x_all = _to_numpy_1d(arrays["x"])        # [m]
    y_all = _to_numpy_1d(arrays["y"])        # [m]
    t_all = _to_numpy_1d(arrays["t"])        # [s]
    p_all = _to_numpy_1d(arrays["p"])        # [Pa]
    Sco2_raw_all = _to_numpy_1d(arrays["Sco2"])  # raw [0,1]

    Xc, Yc = make_grid_centers(args.nx, args.ny, float(L_ref))
    hole_ok = well_hole_mask_outside(Xc, Yc)

    rows = []
    grids = []

    for k, (t_phys, pct) in enumerate(zip(t_sel, pct_sel)):
        idx = sample_indices_at_time(
            t_all=t_all,
            t_target=float(t_phys),
            n_target=int(args.n_points),
            chunk_size=int(args.chunk_size),
            max_rounds=int(args.max_rounds),
            seed=int(args.seed + 101 * k),
        )

        x = x_all[idx].astype(np.float64, copy=False)
        y = y_all[idx].astype(np.float64, copy=False)
        p_mpa = (p_all[idx].astype(np.float64, copy=False) / 1e6)

        s_raw = Sco2_raw_all[idx].astype(np.float64, copy=False)
        s_raw = np.clip(s_raw, 0.0, 1.0)
        s_data_pts = Snr + (1.0 - Sw_irr - Snr) * s_raw

        p_data_grid, cnt = bin_to_grid_mean(x, y, p_mpa, args.nx, args.ny, float(L_ref))
        s_data_grid, _ = bin_to_grid_mean(x, y, s_data_pts, args.nx, args.ny, float(L_ref))

        # mask low-count bins
        if int(args.min_count) > 1:
            good = (cnt >= int(args.min_count))
            p_data_grid = np.where(good, p_data_grid, np.nan)
            s_data_grid = np.where(good, s_data_grid, np.nan)

        # mask well holes
        p_data_grid = np.where(hole_ok, p_data_grid, np.nan)
        s_data_grid = np.where(hole_ok, s_data_grid, np.nan)

        p_pred_grid, s_pred_grid = predict_on_centers(model, Xc, Yc, float(t_phys))

        p_err = np.abs(p_pred_grid - p_data_grid)
        s_err = np.abs(s_pred_grid - s_data_grid)

        def rmse(a, b):
            m = np.isfinite(a) & np.isfinite(b)
            if not np.any(m):
                return np.nan
            d = a[m] - b[m]
            return float(np.sqrt(np.mean(d * d)))

        rmse_p = rmse(p_pred_grid, p_data_grid)  # MPa
        rmse_s = rmse(s_pred_grid, s_data_grid)

        rows.append([t_phys, pct, rmse_p, rmse_s, int(idx.size)])
        grids.append((t_phys, pct, p_data_grid, p_pred_grid, p_err, s_data_grid, s_pred_grid, s_err))

        print(f"[Metrics] t={t_phys:.6g}s ({pct:.0f}%) RMSE_p={rmse_p:.3e} MPa RMSE_S={rmse_s:.3e} n={idx.size}")

    # consistent color ranges across all 5 times
    p_vals = []
    pe_vals = []
    se_vals = []
    for (_, _, pD, pP, pE, _, _, sE) in grids:
        p_vals.append(pD); p_vals.append(pP)
        pe_vals.append(pE)
        se_vals.append(sE)

    p_vmin = float(np.nanmin(np.stack(p_vals, axis=0)))
    p_vmax = float(np.nanmax(np.stack(p_vals, axis=0)))
    pe_max = float(np.nanmax(np.stack(pe_vals, axis=0)))
    se_max = float(np.nanmax(np.stack(se_vals, axis=0)))

    note = f"Data baseline: cell-mean binning (nx={args.nx}, ny={args.ny}); bins with count<{args.min_count} masked."

    # plot output (two figures per time)
    for i, (t_phys, pct, pD, pP, pE, sD, sP, sE) in enumerate(grids):
        outP = f"{args.out_prefix}P_t{i}_t{t_phys:.6g}.png"
        outS = f"{args.out_prefix}S_t{i}_t{t_phys:.6g}.png"

        plot_pressure_triptych(
            t_phys, pct, pD, pP, pE, float(L_ref), outP,
            p_range=(p_vmin, p_vmax),
            p_err_range=(0.0, pe_max),
            note=note
        )
        plot_saturation_triptych(
            t_phys, pct, sD, sP, sE, float(L_ref), outS,
            s_err_range=(0.0, se_max),
            note=note
        )

    # save CSV metrics
    csv_path = f"{args.out_prefix}_metrics.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("t_seconds,percent,rmse_p_mpa,rmse_s,n_points\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]}\n")
    print(f"Saved metrics CSV: {csv_path}")


if __name__ == "__main__":
    main()