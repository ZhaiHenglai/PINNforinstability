import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

"""
plot_fig5_only.py

Fig.5: Boundary condition satisfaction (distribution plots).

We evaluate FOUR boundary/IC constraints used in training, at 5 representative times:
  (A) Injection arc flux:      (v_t · n) - 1      -> target 0
  (B) Outer boundary no-flow:  (v_t · n) - 0      -> target 0
  (C) Production arc pressure: (p_tilde - p_out_tilde) -> target 0
  (D) Injection arc saturation: S - (1 - Sw_irr)  -> target 0

Outputs
-------
1) Fig5_boxplots.png   : 2x2 subplots, boxplots across the 5 selected times
2) Fig5_histograms.png : 2x2 subplots, histograms aggregated over all 5 times
3) Fig5_metrics.csv    : per-time summary stats (mean/std/p95/maxabs)

Important reminder
------------------
- v_t is the dimensionless total Darcy velocity used in your PINN (vwx+vcx, vwy+vcy).
- We compute dp/dx, dp/dy via autograd (requires x_t, y_t to have requires_grad=True).
- Inputs to the model are dimensionless: x/L_ref, y/L_ref, t/T_ref.
"""

# =============================================================================
# (A) Constants (MUST match training script)
# =============================================================================
torch.set_default_dtype(torch.float64)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

L_ref = 5.0
T_ref = 1.0e5
K = 1.0e-14

mu_w = 2.5e-4
mu_c = 2.25e-5
mu_ref = mu_w
k_ref = K

U_in = 5.4e-5
p0 = 10.0e6
p_out = 10.0e6

P_ref = mu_ref * U_in * L_ref / k_ref
p_out_tilde = (p_out - p0) / P_ref  # usually 0 in your settings

Sw_irr = 0.2
Snr = 0.2
krw0 = 1.0
krc0 = 1.0
nw = 2.0
nc = 2.0

# Geometry
r_well = 0.5
inj_center = (0.0, 0.0)
out_center = (5.0, 5.0)
_GEOM_EPS = 1e-9

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
# (C) time_unique helper (reuse your existing approach)
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


def pick_times_from_unique(time_unique, quantiles=(0.00, 0.25, 0.50, 0.75, 1.00)):
    tu = np.asarray(time_unique, dtype=np.float64)
    tu = np.unique(tu); tu.sort()
    if tu.size == 0:
        raise ValueError("time_unique is empty.")

    idx_sel = []
    for q in quantiles:
        j = int(round(float(q) * (tu.size - 1)))
        if j not in idx_sel:
            idx_sel.append(j)

    t_sel = [float(tu[j]) for j in idx_sel]
    pct_sel = [float(j / (tu.size - 1)) * 100.0 if tu.size > 1 else 0.0 for j in idx_sel]
    labels = [f"{p:.0f}%\n({t:.3g}s)" for t, p in zip(t_sel, pct_sel)]
    return t_sel, pct_sel, labels


# =============================================================================
# (D) Physics helpers to compute total velocity v_t
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


def grad_first(u, x):
    """First derivative only; no need create_graph=True for boundary checks."""
    return torch.autograd.grad(
        u, x,
        grad_outputs=torch.ones_like(u),
        create_graph=False,
        retain_graph=True,
    )[0]


def total_velocity(model, x_t, y_t, t_t):
    """
    Compute v_t = v_w + v_c (dimensionless total Darcy velocity),
    consistent with training code.
    """
    p_tilde, Sco2 = model(x_t, y_t, t_t)
    krw, krc = relperm_from_Sco2(Sco2)

    dpdx = grad_first(p_tilde, x_t)
    dpdy = grad_first(p_tilde, y_t)

    K_tilde = K / k_ref

    vwx = -K_tilde * (mu_ref / mu_w) * krw * dpdx
    vwy = -K_tilde * (mu_ref / mu_w) * krw * dpdy
    vcx = -K_tilde * (mu_ref / mu_c) * krc * dpdx
    vcy = -K_tilde * (mu_ref / mu_c) * krc * dpdy

    vtx = vwx + vcx
    vty = vwy + vcy
    return p_tilde, Sco2, vtx, vty


# =============================================================================
# (E) Boundary sampling (match your training geometry)
# =============================================================================
def sample_injection_arc(N, t_phys):
    """
    Injection arc at (0,0), theta in (0, pi/2).
    Returns dimensionless x_t,y_t,t_t with requires_grad=True, and normal n=(cos, sin).
    """
    theta = _GEOM_EPS + torch.rand(N, 1, device=DEVICE, dtype=torch.float64) * (0.5 * np.pi - 2.0 * _GEOM_EPS)
    x = inj_center[0] + r_well * torch.cos(theta)
    y = inj_center[1] + r_well * torch.sin(theta)

    x_t = (x / L_ref).requires_grad_(True)
    y_t = (y / L_ref).requires_grad_(True)
    t_t = torch.full_like(x_t, float(t_phys) / T_ref)  # no grad needed for velocity BC

    nx = torch.cos(theta)
    ny = torch.sin(theta)
    return x_t, y_t, t_t, nx, ny


def sample_outlet_arc(N, t_phys):
    """
    Production arc at (L,L), theta in (pi, 3pi/2).
    For Fig.5 we only need p_tilde on this arc.
    """
    theta = np.pi + _GEOM_EPS + torch.rand(N, 1, device=DEVICE, dtype=torch.float64) * (0.5 * np.pi - 2.0 * _GEOM_EPS)
    x = out_center[0] + r_well * torch.cos(theta)
    y = out_center[1] + r_well * torch.sin(theta)

    x_t = (x / L_ref).requires_grad_(True)  # grad not required for p, but harmless
    y_t = (y / L_ref).requires_grad_(True)
    t_t = torch.full_like(x_t, float(t_phys) / T_ref)
    return x_t, y_t, t_t


def sample_outer_boundary(N_each, t_phys):
    """
    Outer boundary no-flow points on square edges, skipping segments near well arcs (same as training).

    Edges:
      x=0, y in [R, L-R]
      y=0, x in [R, L-R]
      x=L, y in [0, L-R]
      y=L, x in [0, L-R]
    """
    L = float(L_ref)
    R = float(r_well)

    # left edge (x=0), normal (-1,0)
    y1 = (R + _GEOM_EPS) + torch.rand(N_each, 1, device=DEVICE, dtype=torch.float64) * (L - R - 2.0 * _GEOM_EPS)
    x1 = torch.zeros_like(y1)
    n1 = torch.tensor([-1.0, 0.0], device=DEVICE, dtype=torch.float64).view(1, 2).repeat(N_each, 1)

    # bottom edge (y=0), normal (0,-1)
    x2 = (R + _GEOM_EPS) + torch.rand(N_each, 1, device=DEVICE, dtype=torch.float64) * (L - R - 2.0 * _GEOM_EPS)
    y2 = torch.zeros_like(x2)
    n2 = torch.tensor([0.0, -1.0], device=DEVICE, dtype=torch.float64).view(1, 2).repeat(N_each, 1)

    # right edge (x=L), normal (1,0)
    y3 = torch.rand(N_each, 1, device=DEVICE, dtype=torch.float64) * (L - R - _GEOM_EPS)
    x3 = torch.full_like(y3, L)
    n3 = torch.tensor([1.0, 0.0], device=DEVICE, dtype=torch.float64).view(1, 2).repeat(N_each, 1)

    # top edge (y=L), normal (0,1)
    x4 = torch.rand(N_each, 1, device=DEVICE, dtype=torch.float64) * (L - R - _GEOM_EPS)
    y4 = torch.full_like(x4, L)
    n4 = torch.tensor([0.0, 1.0], device=DEVICE, dtype=torch.float64).view(1, 2).repeat(N_each, 1)

    x = torch.cat([x1, x2, x3, x4], dim=0)
    y = torch.cat([y1, y2, y3, y4], dim=0)
    n = torch.cat([n1, n2, n3, n4], dim=0)

    x_t = (x / L_ref).requires_grad_(True)
    y_t = (y / L_ref).requires_grad_(True)
    t_t = torch.full_like(x_t, float(t_phys) / T_ref)

    nx = n[:, 0:1]
    ny = n[:, 1:2]
    return x_t, y_t, t_t, nx, ny


# =============================================================================
# (F) Plotting utilities
# =============================================================================
def boxplot_panel(ax, data_by_time, labels, title):
    ax.boxplot(data_by_time, labels=labels, showfliers=False)
    ax.axhline(0.0, linewidth=1.0, linestyle="--")
    ax.set_title(title)
    ax.set_ylabel("Error (value - target)")
    ax.tick_params(axis="x", rotation=0)


def hist_panel(ax, data_all, title, bins=80):
    ax.hist(data_all, bins=bins)
    ax.axvline(0.0, linewidth=1.0, linestyle="--")
    ax.set_title(title)
    ax.set_xlabel("Error (value - target)")
    ax.set_ylabel("Count")


def stats_row(arr):
    arr = np.asarray(arr)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return (np.nan, np.nan, np.nan, np.nan)
    return (
        float(np.mean(arr)),
        float(np.std(arr)),
        float(np.percentile(np.abs(arr), 95)),
        float(np.max(np.abs(arr))),
    )


# =============================================================================
# (G) Main
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="Model weights state_dict .pt")
    ap.add_argument("--data_pt", default=None, help="Dataset pack .pt (only needed if time_unique.npy missing)")
    ap.add_argument("--time_unique_npy", default="time_unique.npy", help="time_unique.npy (recommended)")
    ap.add_argument("--beta", type=float, default=BETA_PLOT_DEFAULT, help="Plot beta (default=30)")

    ap.add_argument("--N_inj", type=int, default=4000, help="Samples on injection arc per time")
    ap.add_argument("--N_out", type=int, default=4000, help="Samples on outlet arc per time")
    ap.add_argument("--N_each", type=int, default=1200, help="Samples per outer-edge segment per time")

    ap.add_argument("--quantiles", default="0.00,0.25,0.50,0.75,1.00", help="5 time quantiles by default")
    ap.add_argument("--out_prefix", default="Fig5", help="Output prefix")
    args = ap.parse_args()

    # Load model
    model = load_model(args.weights, beta_plot=args.beta)
    print(f"Loaded weights: {args.weights} | beta={args.beta}")

    # Times
    time_unique = load_time_unique(args.time_unique_npy, args.data_pt)
    q_list = [float(s.strip()) for s in args.quantiles.split(",") if s.strip() != ""]
    t_sel, pct_sel, labels = pick_times_from_unique(time_unique, quantiles=q_list)
    print("Selected times:", t_sel)

    # Collect per-time error samples
    err_inj_flux_list = []   # (v·n - 1)
    err_outer_noflow_list = []  # (v·n - 0)
    err_out_p_list = []      # (p_tilde - p_out_tilde)
    err_inj_sat_list = []    # (S - (1 - Sw_irr))

    for i, t_phys in enumerate(t_sel):
        # Injection arc
        xi, yi, ti, nxi, nyi = sample_injection_arc(args.N_inj, t_phys)
        p_t, S_i, vtx_i, vty_i = total_velocity(model, xi, yi, ti)
        vn_in = (vtx_i * nxi + vty_i * nyi)
        err_inj_flux = (vn_in - 1.0).detach().cpu().numpy().reshape(-1)
        err_inj_sat = (S_i - (1.0 - Sw_irr)).detach().cpu().numpy().reshape(-1)

        # Outer boundary
        xb, yb, tb, nxb, nyb = sample_outer_boundary(args.N_each, t_phys)
        _, _, vtx_b, vty_b = total_velocity(model, xb, yb, tb)
        vn_b = (vtx_b * nxb + vty_b * nyb)
        err_outer = (vn_b - 0.0).detach().cpu().numpy().reshape(-1)

        # Outlet arc (pressure)
        xo, yo, to = sample_outlet_arc(args.N_out, t_phys)
        p_o, _ = model(xo, yo, to)
        err_out_p = (p_o - p_out_tilde).detach().cpu().numpy().reshape(-1)

        err_inj_flux_list.append(err_inj_flux)
        err_inj_sat_list.append(err_inj_sat)
        err_outer_noflow_list.append(err_outer)
        err_out_p_list.append(err_out_p)

        print(f"[t={t_phys:.3g}s] collected: inj={err_inj_flux.size}, outer={err_outer.size}, out={err_out_p.size}")

        # free GPU cache a bit
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    # Aggregate (all times)
    err_inj_flux_all = np.concatenate(err_inj_flux_list, axis=0)
    err_outer_all = np.concatenate(err_outer_noflow_list, axis=0)
    err_out_p_all = np.concatenate(err_out_p_list, axis=0)
    err_inj_sat_all = np.concatenate(err_inj_sat_list, axis=0)

    # ------------------------
    # Fig5: boxplots (by time)
    # ------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.0), constrained_layout=True)
    boxplot_panel(axes[0, 0], err_inj_flux_list, labels, "Injection arc: (v_t·n) - 1  (target 0)")
    boxplot_panel(axes[0, 1], err_outer_noflow_list, labels, "Outer boundary: (v_t·n) - 0  (target 0)")
    boxplot_panel(axes[1, 0], err_out_p_list, labels, "Production arc: (p~ - p~out)  (target 0)")
    boxplot_panel(axes[1, 1], err_inj_sat_list, labels, "Injection arc: S - (1 - Sw_irr)  (target 0)")
    fig.suptitle("Fig.5  Boundary condition satisfaction (boxplots by time)", y=1.02)
    out_box = f"{args.out_prefix}_boxplots.png"
    fig.savefig(out_box, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_box}")

    # ------------------------
    # Fig5: histograms (all-time)
    # ------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.0), constrained_layout=True)
    hist_panel(axes[0, 0], err_inj_flux_all, "Injection arc: (v_t·n) - 1  (all times)")
    hist_panel(axes[0, 1], err_outer_all, "Outer boundary: (v_t·n) - 0  (all times)")
    hist_panel(axes[1, 0], err_out_p_all, "Production arc: (p~ - p~out)  (all times)")
    hist_panel(axes[1, 1], err_inj_sat_all, "Injection arc: S - (1 - Sw_irr)  (all times)")
    fig.suptitle("Fig.5  Boundary condition satisfaction (histograms aggregated)", y=1.02)
    out_hist = f"{args.out_prefix}_histograms.png"
    fig.savefig(out_hist, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_hist}")

    # ------------------------
    # Metrics CSV (per time)
    # ------------------------
    rows = []
    for (t_phys, pct, a, b, c, d) in zip(
        t_sel, pct_sel,
        err_inj_flux_list, err_outer_noflow_list, err_out_p_list, err_inj_sat_list
    ):
        m1, s1, p951, mx1 = stats_row(a)
        m2, s2, p952, mx2 = stats_row(b)
        m3, s3, p953, mx3 = stats_row(c)
        m4, s4, p954, mx4 = stats_row(d)
        rows.append([t_phys, pct, m1, s1, p951, mx1, m2, s2, p952, mx2, m3, s3, p953, mx3, m4, s4, p954, mx4])

    csv_path = f"{args.out_prefix}_metrics.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(
            "t_seconds,percent,"
            "mean_err_inj_flux,std_err_inj_flux,p95_abs_err_inj_flux,max_abs_err_inj_flux,"
            "mean_err_outer,std_err_outer,p95_abs_err_outer,max_abs_err_outer,"
            "mean_err_out_p,std_err_out_p,p95_abs_err_out_p,max_abs_err_out_p,"
            "mean_err_inj_sat,std_err_inj_sat,p95_abs_err_inj_sat,max_abs_err_inj_sat\n"
        )
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")
    print(f"Saved metrics CSV: {csv_path}")


if __name__ == "__main__":
    main()