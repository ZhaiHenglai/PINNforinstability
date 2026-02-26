import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

"""
plot_fig6_only.py

Fig.6: Quantitative front-propagation curves.

We compute (both are produced):
  (1) Area metric:
        A(t) = area(S > S_thr)
  (2) Contour metric (front position):
        Extract contour S = S_thr on a grid, then compute distances of contour points
        to the injection center. We report:
          r_contour_mean(t), r_contour_p95(t)
  Also report an "equivalent radius" from area:
        r_eq(t) = sqrt(A(t) / pi)

Notes
-----
- Inputs to model are DIMENSIONLESS: x/L_ref, y/L_ref, t/T_ref (shape [N,1]).
- We mask well holes in the grid (PDE not valid there; also avoids nonsense front).
- This script is plot-only: it loads state_dict weights and recreates the model.
- Beta is not stored in state_dict; we set beta=30 (default) for plotting.

Optional reference overlay
--------------------------
If you have a reference CSV (from simulation/data processing), use --ref_csv.
Expected columns (at minimum):
  t_seconds, A_ref, r_ref
You can also include r_p95_ref, etc; we will plot what we find.

Performance tips
----------------
- If time_unique is long (hundreds+), use --max_times to downsample.
- Residual plots are heavy; this Fig.6 is lighter (forward only), but still use moderate nx/ny.
"""

# =============================================================================
# (A) Constants (MUST match training script)
# =============================================================================
torch.set_default_dtype(torch.float64)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

L_ref = 5.0
T_ref = 1.0e5
Sw_irr = 0.2
Snr = 0.2

# Geometry
r_well = 0.5
inj_center = (0.0, 0.0)
out_center = (5.0, 5.0)

# Beta for plotting (not in state_dict)
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
        self.beta = 1.0  # NOT stored in state_dict

    def set_beta(self, beta: float):
        self.beta = float(beta)

    def forward(self, x_t, y_t, t_t):
        x_in = 2.0 * x_t - 1.0
        y_in = 2.0 * y_t - 1.0
        t_in = 2.0 * t_t - 1.0
        X = torch.cat([x_in, y_in, t_in], dim=1)

        Z = self.ff(X) if self.use_fourier else X
        p_tilde = self.pnet(Z)   # not used in Fig.6
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
# (C) time_unique loading
# =============================================================================
def load_time_unique(time_unique_npy: str | None, data_pt: str | None):
    if time_unique_npy and os.path.exists(time_unique_npy):
        tu = np.load(time_unique_npy)
        tu = np.asarray(tu, dtype=np.float64)
        tu = np.unique(tu); tu.sort()
        print(f"Loaded time_unique from {time_unique_npy} (n={tu.size})")
        return tu

    if not data_pt:
        raise FileNotFoundError("Need either --time_unique_npy or --data_pt to obtain time_unique.")

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
    print(f"Extracted time_unique and saved time_unique.npy (n={tu.size})")
    return tu


def downsample_times(times: np.ndarray, max_times: int):
    """
    If times is long, pick evenly spaced indices so curves remain representative.
    """
    times = np.asarray(times, dtype=np.float64)
    if max_times <= 0 or times.size <= max_times:
        return times
    idx = np.linspace(0, times.size - 1, max_times).round().astype(int)
    idx = np.unique(idx)
    return times[idx]


# =============================================================================
# (D) Grid + masks
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


# =============================================================================
# (E) Predict S on grid (batched)
# =============================================================================
@torch.no_grad()
def predict_S_on_grid(model, X_phys, Y_phys, t_phys, batch=50000):
    """
    Predict saturation S on a grid for one time.
    Returns a 2D numpy array with NaN inside well holes.
    """
    mask_ok = mask_outside_wells(X_phys, Y_phys)

    x = X_phys.reshape(-1, 1)
    y = Y_phys.reshape(-1, 1)

    x_t_all = torch.tensor(x, dtype=torch.float64, device=DEVICE) / float(L_ref)
    y_t_all = torch.tensor(y, dtype=torch.float64, device=DEVICE) / float(L_ref)
    t_t_all = torch.full_like(x_t_all, float(t_phys) / float(T_ref))

    N = x_t_all.shape[0]
    S_out = np.empty((N,), dtype=np.float64)

    # Forward in batches to avoid large GPU memory peaks
    for i0 in range(0, N, batch):
        i1 = min(N, i0 + batch)
        _, S = model(x_t_all[i0:i1], y_t_all[i0:i1], t_t_all[i0:i1])
        S_out[i0:i1] = S.detach().cpu().numpy().reshape(-1)

    S_grid = S_out.reshape(X_phys.shape)
    S_grid = np.where(mask_ok, S_grid, np.nan)
    return S_grid


# =============================================================================
# (F) Metrics: area and contour distances
# =============================================================================
def compute_area_metric(S_grid, s_thr, dx, dy):
    """
    A(t) = area(S > s_thr) approximated by counting grid cells above threshold.
    """
    m = np.isfinite(S_grid)
    if not np.any(m):
        return np.nan
    above = (S_grid > s_thr) & m
    return float(np.sum(above) * dx * dy)


def compute_contour_distance_metric(X, Y, S_grid, s_thr, center_xy):
    """
    Extract contour S=s_thr and compute distances of contour vertices to center.
    Returns (mean, p95, n_points). If contour does not exist -> (nan, nan, 0).
    """
    # Mask NaNs so contour doesn't try to cross holes
    S_ma = np.ma.array(S_grid, mask=~np.isfinite(S_grid))
    try:
        cs = plt.contour(X, Y, S_ma, levels=[s_thr])
    except Exception:
        return (np.nan, np.nan, 0)

    cx, cy = float(center_xy[0]), float(center_xy[1])
    dists = []

    # Collect vertices from all contour segments
    for coll in cs.collections:
        for path in coll.get_paths():
            v = path.vertices  # shape (M,2): [x,y]
            if v.shape[0] < 2:
                continue
            dx = v[:, 0] - cx
            dy = v[:, 1] - cy
            d = np.sqrt(dx * dx + dy * dy)
            dists.append(d)

    plt.close()  # close the implicit contour figure

    if len(dists) == 0:
        return (np.nan, np.nan, 0)

    d_all = np.concatenate(dists, axis=0)
    mean = float(np.mean(d_all))
    p95 = float(np.percentile(d_all, 95))
    return (mean, p95, int(d_all.size))


# =============================================================================
# (G) Reference overlay (optional CSV)
# =============================================================================
def load_ref_csv(path):
    """
    Expected minimal columns:
      t_seconds, A_ref, r_ref
    Additional optional columns are allowed.
    """
    import pandas as pd
    df = pd.read_csv(path)
    if "t_seconds" not in df.columns:
        raise ValueError("ref_csv must contain column: t_seconds")
    return df


# =============================================================================
# (H) Plotting
# =============================================================================
def plot_fig6_curves(out_png, times, s_thr, area, r_eq, r_mean, r_p95, ref_df=None):
    """
    Two-panel figure:
      Top: r_f(t) (contour mean and p95) and r_eq(t)
      Bottom: A(t)
    """
    fig, axes = plt.subplots(2, 1, figsize=(10.8, 8.2), constrained_layout=True, sharex=True)

    # --- Top panel: radius-type metrics
    ax = axes[0]
    ax.plot(times, r_mean, label=f"Contour mean radius (S={s_thr})")
    ax.plot(times, r_p95, label=f"Contour p95 radius (S={s_thr})")
    ax.plot(times, r_eq, label="Equivalent radius from area (sqrt(A/pi))")
    ax.set_ylabel("Front radius [m]")
    ax.set_title("Fig.6  Front position metrics")
    ax.grid(True, alpha=0.25)

    # overlay reference if provided
    if ref_df is not None:
        if "r_ref" in ref_df.columns:
            ax.plot(ref_df["t_seconds"].to_numpy(), ref_df["r_ref"].to_numpy(),
                    linestyle="--", label="Reference r_ref")
        if "r_p95_ref" in ref_df.columns:
            ax.plot(ref_df["t_seconds"].to_numpy(), ref_df["r_p95_ref"].to_numpy(),
                    linestyle="--", label="Reference r_p95_ref")

    ax.legend()

    # --- Bottom panel: area metric
    ax = axes[1]
    ax.plot(times, area, label=f"Area A(t) = area(S>{s_thr})")
    ax.set_xlabel("Time t [s]")
    ax.set_ylabel("Area [m^2]")
    ax.set_title("Fig.6  Front area metric")
    ax.grid(True, alpha=0.25)

    if ref_df is not None and "A_ref" in ref_df.columns:
        ax.plot(ref_df["t_seconds"].to_numpy(), ref_df["A_ref"].to_numpy(),
                linestyle="--", label="Reference A_ref")
        ax.legend()

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png}")


# =============================================================================
# (I) Main
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="Model weights state_dict .pt")
    ap.add_argument("--time_unique_npy", default="time_unique.npy", help="time_unique.npy (recommended)")
    ap.add_argument("--data_pt", default=None, help="Dataset pack .pt (fallback to extract time_unique)")
    ap.add_argument("--beta", type=float, default=BETA_PLOT_DEFAULT, help="Plot beta (default=30)")

    ap.add_argument("--s_thr", type=float, default=0.5,
                    help="S threshold used for contour/area (default 0.5). Change this to other values if needed.")
    ap.add_argument("--nx", type=int, default=201, help="Grid resolution in x")
    ap.add_argument("--ny", type=int, default=201, help="Grid resolution in y")
    ap.add_argument("--batch", type=int, default=50000, help="Forward batch size for grid prediction")
    ap.add_argument("--max_times", type=int, default=0,
                    help="If >0, downsample time_unique to at most this many times (evenly). 0 means use all times.")
    ap.add_argument("--ref_csv", default=None,
                    help="Optional reference CSV with columns: t_seconds, A_ref, r_ref (and optional r_p95_ref).")
    ap.add_argument("--out_prefix", default="Fig6", help="Output prefix")
    args = ap.parse_args()

    model = load_model(args.weights, beta_plot=args.beta)
    print(f"Loaded weights: {args.weights} | beta={args.beta}")

    time_unique = load_time_unique(args.time_unique_npy, args.data_pt)
    times = downsample_times(time_unique, int(args.max_times))
    print(f"Using n_times={times.size} for Fig.6")

    X, Y = make_grid_phys(int(args.nx), int(args.ny), float(L_ref))
    mask_ok = mask_outside_wells(X, Y)
    dx = float(L_ref) / (int(args.nx) - 1)
    dy = float(L_ref) / (int(args.ny) - 1)

    area = np.full((times.size,), np.nan, dtype=np.float64)
    r_eq = np.full((times.size,), np.nan, dtype=np.float64)
    r_mean = np.full((times.size,), np.nan, dtype=np.float64)
    r_p95 = np.full((times.size,), np.nan, dtype=np.float64)
    n_contour = np.zeros((times.size,), dtype=np.int64)

    for i, t in enumerate(times):
        S_grid = predict_S_on_grid(model, X, Y, float(t), batch=int(args.batch))
        # safety mask (holes)
        S_grid = np.where(mask_ok, S_grid, np.nan)

        A = compute_area_metric(S_grid, float(args.s_thr), dx, dy)
        area[i] = A
        if np.isfinite(A) and A > 0.0:
            r_eq[i] = np.sqrt(A / np.pi)
        else:
            r_eq[i] = 0.0 if np.isfinite(A) else np.nan

        rm, rp, nc = compute_contour_distance_metric(X, Y, S_grid, float(args.s_thr), inj_center)
        r_mean[i] = rm
        r_p95[i] = rp
        n_contour[i] = nc

        if (i + 1) % max(1, times.size // 10) == 0:
            print(f"[{i+1}/{times.size}] t={t:.6g}s  A={A:.3e}  r_mean={rm:.3g}  contour_pts={nc}")

        if DEVICE.type == "cuda" and (i % 10 == 0):
            torch.cuda.empty_cache()

    # Optional reference overlay
    ref_df = None
    if args.ref_csv is not None:
        ref_df = load_ref_csv(args.ref_csv)
        print(f"Loaded reference CSV: {args.ref_csv}")

    # Save CSV of computed metrics
    csv_path = f"{args.out_prefix}_front_metrics.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("t_seconds,time_ratio,S_thr,A_pred_m2,r_eq_from_area_m,r_contour_mean_m,r_contour_p95_m,n_contour_points\n")
        t_end = float(times[-1]) if times.size > 0 else 1.0
        for i, t in enumerate(times):
            time_ratio = float(t / t_end) if t_end > 0 else 0.0
            f.write(f"{t},{time_ratio},{args.s_thr},{area[i]},{r_eq[i]},{r_mean[i]},{r_p95[i]},{n_contour[i]}\n")
    print(f"Saved metrics CSV: {csv_path}")

    # Plot figure
    out_png = f"{args.out_prefix}_front_curves.png"
    plot_fig6_curves(out_png, times, float(args.s_thr), area, r_eq, r_mean, r_p95, ref_df=ref_df)


if __name__ == "__main__":
    main()