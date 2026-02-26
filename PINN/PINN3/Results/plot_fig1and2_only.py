"""
plot_fig12_only.py

Plot-only postprocessing script for your 2D brine–CO2 PINN.

This script generates:
  - Fig.1: geometry + ALL BC/IC annotations (domain, well holes, boundary types)
  - Fig.2: pressure & CO2 saturation snapshots at selected times (quantiles)

Why this exists
---------------
Your training output .pt is a *state_dict* (weights only). To do inference and plot figures,
we must:
  1) Recreate the exact model architecture (TwoNetPINN)
  2) Load the state_dict
  3) Set beta manually (beta is NOT saved in state_dict)
  4) Provide time_unique to select representative times

Recommended workflow
--------------------
- Save time_unique once as a small file (time_unique.npy). Then plotting is fast.
- If time_unique.npy is missing, this script can extract time_unique from your dataset pack
  (tables_cache_tensor.pt / tables_cache.pt), and then save time_unique.npy automatically.

Model I/O (important reminder)
------------------------------
Your model forward signature is:
    p_tilde, S_co2 = model(x_t, y_t, t_t)

Inputs are 3 separate tensors of shape [N,1], all dimensionless:
    x_t = x / L_ref
    y_t = y / L_ref
    t_t = t / T_ref

Outputs:
    p_tilde = (p - p0) / P_ref
    S_co2 in [Snr, 1 - Sw_irr]  (residual-saturation method B)
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Wedge


# -----------------------------
# (A) Put your physical/scaling constants here (must match training script)
# -----------------------------
L_ref = 5.0          # [m]
T_ref = 1.0e5        # [s]
p0 = 10.0e6          # [Pa]
U_in = 5.4e-5        # [m/s]
K = 1.0e-14          # [m^2]
mu_w = 2.5e-4
mu_ref = mu_w
U_ref = U_in
k_ref = K

P_ref = mu_ref * U_ref * L_ref / k_ref  # same definition as training
Sw_irr = 0.2
Snr = 0.2

# Geometry
r_well = 0.5
inj_center = (0.0, 0.0)
out_center = (5.0, 5.0)

# Plotting beta (NOT stored in state_dict). Default final beta in your training = 30.
BETA_PLOT_DEFAULT = 30.0


# -----------------------------
# (B) Match your model definition EXACTLY
# -----------------------------
torch.set_default_dtype(torch.float64)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FourierFeatures(nn.Module):
    def __init__(self, in_dim=3, m=32, scale=6.0):
        super().__init__()
        B = torch.randn(in_dim, m) * scale
        # Buffer is saved in state_dict, so load_state_dict will restore the trained B exactly.
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
        # Inputs are dimensionless: x/L_ref, y/L_ref, t/T_ref in [0,1] (roughly).
        # Internally we map them to [-1,1] (consistent with your training code).
        x_in = 2.0 * x_t - 1.0
        y_in = 2.0 * y_t - 1.0
        t_in = 2.0 * t_t - 1.0
        X = torch.cat([x_in, y_in, t_in], dim=1)

        Z = self.ff(X) if self.use_fourier else X
        p_tilde = self.pnet(Z)
        s_logit = self.snet(Z)
        S_hat = torch.sigmoid(self.beta * s_logit)

        # residual-saturation method B: S_co2 in [Snr, 1 - Sw_irr]
        Sco2 = Snr + (1.0 - Sw_irr - Snr) * S_hat
        return p_tilde, Sco2


def load_model(weights_path: str, beta_plot: float, device=DEVICE):
    """
    Create model, load state_dict, set beta, move to device.
    """
    model = TwoNetPINN(width=128, use_fourier=True).to(device)
    sd = torch.load(weights_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(sd, strict=False)

    # If strict=False, we print mismatches to help debugging.
    if missing:
        print("[WARN] Missing keys when loading state_dict:", missing)
    if unexpected:
        print("[WARN] Unexpected keys when loading state_dict:", unexpected)

    model.set_beta(beta_plot)
    model.eval()
    return model


# -----------------------------
# (C) time_unique loading
# -----------------------------
def load_time_unique(time_unique_npy: str | None, data_pt: str | None):
    """
    Prefer time_unique.npy.
    If missing, try extracting from dataset pack (.pt) and save time_unique.npy.
    """
    if time_unique_npy and os.path.exists(time_unique_npy):
        tu = np.load(time_unique_npy)
        tu = np.asarray(tu, dtype=np.float64)
        tu = np.unique(tu)
        tu.sort()
        print(f"Loaded time_unique from {time_unique_npy} (n={tu.size})")
        return tu

    # Fallback: extract from dataset pack
    if data_pt is None:
        raise FileNotFoundError(
            "time_unique.npy not found and --data_pt not provided.\n"
            "Provide either --time_unique_npy time_unique.npy OR --data_pt tables_cache_tensor.pt"
        )

    print(f"[INFO] Extracting time_unique from dataset pack: {data_pt}")
    pack = torch.load(data_pt, map_location="cpu")
    if not isinstance(pack, dict) or "arrays" not in pack:
        raise ValueError(f"{data_pt} is not a valid pack (missing 'arrays').")

    tu = pack.get("time_unique", None)
    if tu is None:
        # derive from arrays["t"]
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

    # save for next time
    out = "time_unique.npy"
    np.save(out, tu)
    print(f"Saved extracted time_unique to {out} (n={tu.size})")
    return tu


def pick_times_from_unique(time_unique, quantiles=(0.00, 0.10, 0.25, 0.50, 0.75, 1.00)):
    """
    Select representative times by quantiles on the discrete time grid.
    This avoids selecting a time that does not exist.
    """
    tu = np.asarray(time_unique, dtype=np.float64)
    tu = np.unique(tu)
    tu.sort()
    if tu.size == 0:
        raise ValueError("time_unique is empty.")

    idx = []
    for q in quantiles:
        q = float(q)
        q = min(max(q, 0.0), 1.0)
        j = int(round(q * (tu.size - 1)))
        idx.append(j)

    # remove duplicates (e.g., if tu has few entries)
    idx_sel = []
    for j in idx:
        if j not in idx_sel:
            idx_sel.append(j)

    t_sel = [float(tu[j]) for j in idx_sel]
    pct_sel = [float(j / (tu.size - 1)) * 100.0 if tu.size > 1 else 0.0 for j in idx_sel]
    return t_sel, pct_sel


# -----------------------------
# (D) Geometry & mask helpers
# -----------------------------
def make_xy_grid_phys(L_ref, nx=201, ny=201):
    x = np.linspace(0.0, float(L_ref), int(nx))
    y = np.linspace(0.0, float(L_ref), int(ny))
    X, Y = np.meshgrid(x, y, indexing="xy")
    return X, Y


def well_hole_mask_outside(X, Y, inj_center, out_center, r_well):
    """
    True outside well holes (PDE valid region).
    We use full circles for masking; inside the square they become quarter-circles naturally.
    """
    cx1, cy1 = float(inj_center[0]), float(inj_center[1])
    cx2, cy2 = float(out_center[0]), float(out_center[1])
    R = float(r_well)
    inside = ((X - cx1) ** 2 + (Y - cy1) ** 2 < R**2) | ((X - cx2) ** 2 + (Y - cy2) ** 2 < R**2)
    return ~inside


# -----------------------------
# (E) Fig.1
# -----------------------------
def plot_fig1_geometry(savepath="Fig1_geometry_BCs.png", dpi=300):
    """
    Fig.1: Geometry and ALL BC/IC annotations.
    """
    L = float(L_ref)
    R = float(r_well)

    fig, ax = plt.subplots(figsize=(7.8, 7.0))

    # Domain
    ax.add_patch(Rectangle((0, 0), L, L, fill=False, linewidth=2))

    # Well holes (quarter-circle wedges)
    ax.add_patch(Wedge(inj_center, R, 0, 90, facecolor="lightgray", edgecolor="k", linewidth=1.5))
    ax.add_patch(Wedge(out_center, R, 180, 270, facecolor="lightgray", edgecolor="k", linewidth=1.5))

    # Centers
    ax.plot([inj_center[0]], [inj_center[1]], marker="o")
    ax.plot([out_center[0]], [out_center[1]], marker="o")

    # Numbered labels to make "all options" explicit (good for paper captions)
    ax.text(0.05 * L, 0.16 * L, "(1) Injection arc BC", fontsize=11, weight="bold")
    ax.text(0.55 * L, 0.62 * L, "(2) Production arc BC", fontsize=11, weight="bold")
    ax.text(0.05 * L, 0.92 * L, "(3) Outer boundary BC", fontsize=11, weight="bold")
    ax.text(0.52 * L, 0.08 * L, "(4) Initial condition", fontsize=11, weight="bold")

    # Explanatory text boxes (English, so you can reuse in the paper/caption)
    ax.text(
        0.05 * L, 0.03 * L,
        "Injection well hole is removed (PDE not enforced inside).\n"
        "BC on injection arc:\n"
        "  • total normal velocity  v_t · n = 1  (dimensionless)\n"
        "  • CO2 saturation          S_CO2 = 1 - S_w,irr",
        ha="left", va="bottom", fontsize=10
    )

    ax.text(
        0.55 * L, 0.48 * L,
        "Production well hole is removed.\n"
        "BC on production arc:\n"
        "  • pressure  p~ = p~_out",
        ha="left", va="bottom", fontsize=10
    )

    ax.text(
        0.05 * L, 0.74 * L,
        "Outer boundary (square edges):\n"
        "  • no-flow  v_t · n = 0",
        ha="left", va="bottom", fontsize=10
    )

    ax.text(
        0.52 * L, 0.03 * L,
        "Initial condition at t=0:\n"
        "  • p~ = 0\n"
        "  • S_CO2 = S_nr",
        ha="left", va="bottom", fontsize=10
    )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-0.05 * L, 1.05 * L)
    ax.set_ylim(-0.05 * L, 1.05 * L)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Fig.1  Geometry and boundary/initial conditions (well holes removed)")
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(savepath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[Fig.1] Saved: {savepath}")


# -----------------------------
# (F) Fig.2
# -----------------------------
@torch.no_grad()
def predict_on_grid(model, X_phys, Y_phys, t_phys):
    """
    Predict p(x,y,t) and S_CO2(x,y,t) on a grid.

    Returns
    -------
    p_mpa : 2D array, MPa
    S     : 2D array, saturation
    """
    mask_outside = well_hole_mask_outside(X_phys, Y_phys, inj_center, out_center, r_well)

    # Convert PHYSICAL -> DIMENSIONLESS inputs expected by the model
    x_t = torch.tensor(X_phys.reshape(-1, 1), dtype=torch.float64, device=DEVICE) / float(L_ref)
    y_t = torch.tensor(Y_phys.reshape(-1, 1), dtype=torch.float64, device=DEVICE) / float(L_ref)
    t_t = torch.full_like(x_t, float(t_phys) / float(T_ref))

    p_tilde, S = model(x_t, y_t, t_t)

    # Convert dimensionless pressure -> physical pressure
    p_phys = float(p0) + float(P_ref) * p_tilde
    p_mpa = (p_phys.detach().cpu().numpy().reshape(X_phys.shape)) / 1e6
    S = S.detach().cpu().numpy().reshape(X_phys.shape)

    # Mask the well holes as NaN so they appear as empty holes in the plots
    p_mpa = np.where(mask_outside, p_mpa, np.nan)
    S = np.where(mask_outside, S, np.nan)
    return p_mpa, S


def plot_fig2_snapshots(model, time_unique, nx=201, ny=201, savepath="Fig2_p_S_snapshots.png", dpi=300):
    """
    Fig.2: Pressure & saturation snapshots at representative times.
    Default times are quantiles: 0%, 10%, 25%, 50%, 75%, 100%.
    """
    quantiles = (0.00, 0.10, 0.25, 0.50, 0.75, 1.00)
    t_sel, pct_sel = pick_times_from_unique(time_unique, quantiles=quantiles)

    X, Y = make_xy_grid_phys(L_ref, nx=nx, ny=ny)

    P_list, S_list = [], []
    for t in t_sel:
        p_mpa, s = predict_on_grid(model, X, Y, t)
        P_list.append(p_mpa)
        S_list.append(s)

    # Colorbar ranges
    s_vmin = float(Snr)
    s_vmax = float(1.0 - Sw_irr)

    p_all = np.stack(P_list, axis=0)
    p_vmin = float(np.nanmin(p_all))
    p_vmax = float(np.nanmax(p_all))

    ncols = len(t_sel)
    fig, axes = plt.subplots(2, ncols, figsize=(3.2 * ncols, 6.2), constrained_layout=True)

    for j, (t, pct) in enumerate(zip(t_sel, pct_sel)):
        axp = axes[0, j]
        axs = axes[1, j]

        im_p = axp.imshow(
            P_list[j],
            origin="lower",
            extent=[0, float(L_ref), 0, float(L_ref)],
            vmin=p_vmin, vmax=p_vmax,
            aspect="equal"
        )
        axp.set_title(f"t = {t:.3g} s  ({pct:.0f}%)")
        axp.set_xlabel("x [m]")
        if j == 0:
            axp.set_ylabel("y [m]")

        # Draw hole outlines for clarity
        axp.add_patch(Wedge(inj_center, float(r_well), 0, 90, fill=False, edgecolor="k", linewidth=1.0))
        axp.add_patch(Wedge(out_center, float(r_well), 180, 270, fill=False, edgecolor="k", linewidth=1.0))

        im_s = axs.imshow(
            S_list[j],
            origin="lower",
            extent=[0, float(L_ref), 0, float(L_ref)],
            vmin=s_vmin, vmax=s_vmax,
            aspect="equal"
        )
        axs.set_xlabel("x [m]")
        if j == 0:
            axs.set_ylabel("y [m]")

        # Optional: show front contour S=0.5 if it exists in the plotted range
        try:
            axs.contour(X, Y, S_list[j], levels=[0.5], linewidths=1.0)
        except Exception:
            pass

        axs.add_patch(Wedge(inj_center, float(r_well), 0, 90, fill=False, edgecolor="k", linewidth=1.0))
        axs.add_patch(Wedge(out_center, float(r_well), 180, 270, fill=False, edgecolor="k", linewidth=1.0))

    cbar_p = fig.colorbar(im_p, ax=axes[0, :], shrink=0.95, pad=0.02)
    cbar_p.set_label("Pressure p [MPa]")

    cbar_s = fig.colorbar(im_s, ax=axes[1, :], shrink=0.95, pad=0.02)
    cbar_s.set_label("CO2 saturation $S_{CO2}$ [-]")

    fig.suptitle("Fig.2  PINN snapshots over time (well holes masked out)", y=1.02)
    fig.savefig(savepath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"[Fig.2] Saved: {savepath}")
    print("[Fig.2] Selected times (s):", t_sel)


# -----------------------------
# (G) Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="Path to model weights state_dict .pt")
    ap.add_argument("--beta", type=float, default=BETA_PLOT_DEFAULT, help="beta used in plotting (default final beta=30)")
    ap.add_argument("--time_unique_npy", default="time_unique.npy", help="Path to time_unique.npy (recommended)")
    ap.add_argument("--data_pt", default=None, help="Optional dataset pack .pt to extract time_unique if npy not found")
    ap.add_argument("--nx", type=int, default=201, help="Grid resolution in x")
    ap.add_argument("--ny", type=int, default=201, help="Grid resolution in y")
    args = ap.parse_args()

    # 1) load model weights
    model = load_model(args.weights, beta_plot=args.beta, device=DEVICE)
    print(f"Loaded model weights from: {args.weights}")
    print(f"Using beta for plotting: {args.beta}")

    # 2) load time_unique
    time_unique = load_time_unique(args.time_unique_npy, args.data_pt)

    # 3) plot figures
    plot_fig1_geometry(savepath="Fig1_geometry_BCs.png", dpi=300)
    plot_fig2_snapshots(model, time_unique, nx=args.nx, ny=args.ny, savepath="Fig2_p_S_snapshots.png", dpi=300)


if __name__ == "__main__":
    main()