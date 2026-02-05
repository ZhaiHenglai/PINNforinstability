#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.patches import Arc

# ============================================================
# 0) 用户配置区
# ============================================================

WEIGHTS_PATH = "/home/henglai_pc/pythonTesst/pinn_co2_brine_multitime_TwoNet_beta.pt"

# 如果想从数据里自动提取时间点，就填你的 glob；不想就置为 None
DATA_GLOB = "/home/henglai_pc/pythonTesst/data/Test1/Visc_fingering_*1_points.csv"
# DATA_GLOB = None

# 画哪些时刻：如果 DATA_GLOB 不为 None，会从数据取 unique times；
# 如果 DATA_GLOB 为 None，则用下面 TIMES_SEC
TIMES_SEC = None  # e.g. [0.0, 0.2e5, 0.4e5, 0.6e5, 0.8e5, 1.0e5]

# 网格分辨率（越大越清晰但越慢）
NX, NY = 251, 251

# 前沿定义阈值（S_star用于等值线/侵入区域；S_bt用于出口breakthrough检测）
S_STAR = 0.5
S_BT = 0.5

# q分位前沿（更稳健，推荐0.9~0.99）
Q_FRONT = 0.95

# 出口弧采样点数（越大越稳健但越慢）
OUTLET_NTHETA = 400

# 是否保存图片/CSV
SAVE_FIG = True
SAVE_CSV = True

# ============================================================
# 1) 必须与训练时一致的物理/尺度参数
# ============================================================

torch.set_default_dtype(torch.float64)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

L_ref = 5.0          # m
T_ref = 1.0e5        # s
K = 1.0e-14          # m^2
mu_w = 2.5e-4
mu_c = 2.25e-5
U_in = 5.4e-5
p0 = 10.0e6          # Pa

mu_ref = mu_w
U_ref = U_in
k_ref = K
P_ref = mu_ref * U_ref * L_ref / k_ref  # Pa

# 几何：两段四分之一圆弧
r_well = 0.5
inj_center = (0.0, 0.0)
out_center = (5.0, 5.0)

# ============================================================
# 2) 网络结构（必须与训练一致）
# ============================================================

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
    """
    输出：
      p_tilde（无量纲压力）
      Sco2 in [0,1]
    """
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

        self.beta = 30.0  # 推理建议使用训练末期的 beta_max

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
        Sco2 = torch.sigmoid(self.beta * s_logit)
        return p_tilde, Sco2

# ============================================================
# 3) 工具函数：读数据时间点（可选）
# ============================================================

def read_times_from_data(glob_pattern: str):
    files = sorted(glob.glob(glob_pattern))
    if not files:
        raise FileNotFoundError(f"No files matched: {glob_pattern}")

    times = []
    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception:
            df = pd.read_csv(f, delim_whitespace=True)

        # 列名清理
        df.columns = [c.strip() for c in df.columns]
        if "phase1::Time" not in df.columns:
            raise ValueError(f"{os.path.basename(f)} missing column phase1::Time")

        t = pd.to_numeric(df["phase1::Time"], errors="coerce").dropna().to_numpy()
        times.append(t)

    t_all = np.concatenate(times, axis=0)
    t_unique = np.unique(t_all)
    t_unique.sort()
    return t_unique

def pick_evenly_spaced(times, n=6):
    times = np.array(times, dtype=float)
    if len(times) <= n:
        return times
    idx = np.linspace(0, len(times)-1, n).round().astype(int)
    return times[idx]

# ============================================================
# 4) 网格预测 + 画弧
# ============================================================

@torch.no_grad()
def predict_grid(model, Nx=251, Ny=251, t_sec=0.0, batch=65536):
    xs = np.linspace(0.0, L_ref, Nx)
    ys = np.linspace(0.0, L_ref, Ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")

    x = torch.tensor(XX.reshape(-1,1), device=DEVICE) / L_ref
    y = torch.tensor(YY.reshape(-1,1), device=DEVICE) / L_ref
    t = torch.full_like(x, float(t_sec)/T_ref)

    p_list, s_list = [], []
    N = x.shape[0]
    for i in range(0, N, batch):
        p_tilde, S = model(x[i:i+batch], y[i:i+batch], t[i:i+batch])
        p_list.append(p_tilde.detach().cpu())
        s_list.append(S.detach().cpu())

    p_tilde = torch.cat(p_list, dim=0).numpy().reshape(Ny, Nx)
    Sco2 = torch.cat(s_list, dim=0).numpy().reshape(Ny, Nx)

    p_phys = p0 + P_ref * p_tilde  # Pa
    return XX, YY, p_phys, Sco2

def draw_arcs(ax):
    ax.add_patch(Arc(inj_center, 2*r_well, 2*r_well, angle=0, theta1=0, theta2=90, lw=2))
    ax.add_patch(Arc(out_center, 2*r_well, 2*r_well, angle=0, theta1=180, theta2=270, lw=2))

# ============================================================
# 5) 前沿指标（投影到(1,1)方向）+ invaded area
# ============================================================

def front_metrics_from_field(XX, YY, Sco2, S_star=0.5, q=0.95):
    d = np.array([1.0, 1.0], dtype=float)
    d = d / np.linalg.norm(d)

    mask = Sco2 >= S_star
    if not np.any(mask):
        return 0.0, 0.0, 0.0

    x = XX[mask]
    y = YY[mask]
    s = x * d[0] + y * d[1]  # m

    front_max = float(np.max(s))
    front_q = float(np.quantile(s, q))

    dx = float(XX[0,1] - XX[0,0])
    dy = float(YY[1,0] - YY[0,0])
    area_inv = float(np.sum(mask) * dx * dy)

    return front_max, front_q, area_inv

# ============================================================
# 6) Breakthrough：出口弧上检测 max(Sco2)
# ============================================================

@torch.no_grad()
def outlet_arc_s_stats(model, t_sec, N_theta=400, batch=65536):
    theta = np.linspace(np.pi, 1.5*np.pi, N_theta).reshape(-1, 1)
    x = out_center[0] + r_well * np.cos(theta)
    y = out_center[1] + r_well * np.sin(theta)

    x_t = torch.tensor(x, device=DEVICE) / L_ref
    y_t = torch.tensor(y, device=DEVICE) / L_ref
    t_t = torch.full_like(x_t, float(t_sec)/T_ref)

    S_list = []
    N = x_t.shape[0]
    for i in range(0, N, batch):
        _, S = model(x_t[i:i+batch], y_t[i:i+batch], t_t[i:i+batch])
        S_list.append(S.detach().cpu())
    S_all = torch.cat(S_list, dim=0).numpy().reshape(-1)

    return float(S_all.max()), float(S_all.mean())

@torch.no_grad()
def find_breakthrough_time(model, times_sec, S_bt=0.5, N_theta=400, refine=True, n_bisect=12):
    times = np.array(sorted(times_sec), dtype=float)

    outlet_maxS = []
    for t in times:
        smax, _ = outlet_arc_s_stats(model, t, N_theta=N_theta)
        outlet_maxS.append(smax)
    outlet_maxS = np.array(outlet_maxS, dtype=float)

    hit = np.where(outlet_maxS >= S_bt)[0]
    if len(hit) == 0:
        return None, None, outlet_maxS

    i = int(hit[0])
    if (not refine) or i == 0:
        return float(times[i]), i, outlet_maxS

    t_lo = float(times[i-1])
    t_hi = float(times[i])

    smax_lo, _ = outlet_arc_s_stats(model, t_lo, N_theta=N_theta)
    if smax_lo >= S_bt:
        return t_lo, i-1, outlet_maxS

    for _ in range(n_bisect):
        t_mid = 0.5 * (t_lo + t_hi)
        smax_mid, _ = outlet_arc_s_stats(model, t_mid, N_theta=N_theta)
        if smax_mid >= S_bt:
            t_hi = t_mid
        else:
            t_lo = t_mid

    return float(t_hi), i, outlet_maxS

# ============================================================
# 7) 汇总：多时刻计算前沿 + outlet Smax/Smean
# ============================================================

def compute_front_vs_time(model, times_sec, Nx=251, Ny=251, S_star=0.5, q=0.95, batch=65536, outlet_N=400):
    rows = []
    for t_sec in times_sec:
        XX, YY, p_phys, Sco2 = predict_grid(model, Nx=Nx, Ny=Ny, t_sec=float(t_sec), batch=batch)
        fmax, fq, area = front_metrics_from_field(XX, YY, Sco2, S_star=S_star, q=q)

        smax_out, smean_out = outlet_arc_s_stats(model, float(t_sec), N_theta=outlet_N)
        rows.append([float(t_sec), fmax, fq, area, smax_out, smean_out])

    dfm = pd.DataFrame(rows, columns=[
        "time_s",
        "front_max_m",
        f"front_q{int(q*100)}_m",
        "invaded_area_m2",
        "outlet_Smax",
        "outlet_Smean"
    ])
    return dfm

# ============================================================
# 8) 作图：前沿曲线 / outlet曲线 / 等值线前沿图
# ============================================================

def plot_front_curve(dfm, S_star, q, save=False):
    plt.figure(figsize=(7,5))
    plt.plot(dfm["time_s"]/T_ref, dfm[f"front_q{int(q*100)}_m"], marker="o",
             label=f"front q{int(q*100)} (S>={S_star})")
    plt.plot(dfm["time_s"]/T_ref, dfm["front_max_m"], marker="x",
             label=f"front max (S>={S_star})")
    plt.xlabel("t / T_ref")
    plt.ylabel("Front position along (1,1) direction (m)")
    plt.grid(True)
    plt.legend()
    if save:
        plt.savefig(f"front_curve_S{S_star}_q{int(q*100)}.png", dpi=200)
    plt.show()

def plot_outlet_curve(dfm, S_bt, t_bt=None, save=False):
    plt.figure(figsize=(7,5))
    plt.plot(dfm["time_s"]/T_ref, dfm["outlet_Smax"], marker="o", label="Outlet max Sco2")
    plt.axhline(S_bt, linestyle="--", label=f"S_bt={S_bt}")
    if t_bt is not None:
        plt.axvline(t_bt/T_ref, linestyle="--", label=f"t_bt/T_ref={t_bt/T_ref:.4f}")
    plt.xlabel("t / T_ref")
    plt.ylabel("Outlet arc max Sco2")
    plt.grid(True)
    plt.legend()
    plt.title("Breakthrough detection at outlet arc")
    if save:
        plt.savefig(f"outlet_Smax_vs_time_Sbt{S_bt}.png", dpi=200)
    plt.show()

def plot_front_contours(model, times_sec, Nx=251, Ny=251, S_star=0.5, save=False):
    for t_sec in times_sec:
        XX, YY, p_phys, Sco2 = predict_grid(model, Nx=Nx, Ny=Ny, t_sec=float(t_sec))
        plt.figure(figsize=(6,5))
        plt.imshow(Sco2, origin="lower", extent=[0,L_ref,0,L_ref],
                   aspect="equal", vmin=0, vmax=1)
        plt.colorbar()

        cs = plt.contour(XX, YY, Sco2, levels=[S_star])
        plt.clabel(cs, inline=True, fontsize=8)

        plt.title(f"Sco2 contour S={S_star}, t={t_sec:.2e}s")
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.xlim(0, L_ref)
        plt.ylim(0, L_ref)
        draw_arcs(plt.gca())

        if save:
            plt.savefig(f"front_contour_t_{t_sec:.3e}_S{S_star}.png", dpi=200)
        plt.show()

# ============================================================
# 9) main
# ============================================================

def main():
    if not os.path.isfile(WEIGHTS_PATH):
        raise FileNotFoundError(f"Cannot find weights: {WEIGHTS_PATH}")

    # 加载模型
    model = TwoNetPINN(width=128, use_fourier=True).to(DEVICE)
    state = torch.load(WEIGHTS_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.set_beta(30.0)
    model.eval()

    # 选择时间点
    if DATA_GLOB is not None:
        times_all = read_times_from_data(DATA_GLOB)
        # 为了避免太多时刻画图很慢：默认挑6个等间距快照
        times_plot = pick_evenly_spaced(times_all, n=6)
        print(f"Loaded times from data: {len(times_all)} unique, plotting {len(times_plot)} snapshots.")
        times_sec = times_all
        times_snap = times_plot
    else:
        if TIMES_SEC is None:
            # 默认给6个
            times_snap = np.array([0.0, 0.2*T_ref, 0.4*T_ref, 0.6*T_ref, 0.8*T_ref, 1.0*T_ref])
            times_sec = times_snap
        else:
            times_sec = np.array(TIMES_SEC, dtype=float)
            times_sec.sort()
            times_snap = times_sec

    # 计算前沿与出口指标（对 times_sec 全部计算；如果你只想算快照就用 times_snap）
    dfm = compute_front_vs_time(
        model, times_sec,
        Nx=NX, Ny=NY,
        S_star=S_STAR, q=Q_FRONT,
        outlet_N=OUTLET_NTHETA
    )

    # breakthrough time（可二分细化）
    t_bt, idx_bt, outlet_maxS_list = find_breakthrough_time(
        model, times_sec,
        S_bt=S_BT, N_theta=OUTLET_NTHETA,
        refine=True
    )

    if t_bt is None:
        print(f"[BT] No breakthrough up to t={float(np.max(times_sec)):.3e}s for S_bt={S_BT}")
    else:
        print(f"[BT] Breakthrough at t ≈ {t_bt:.6e} s  (t/T_ref={t_bt/T_ref:.6f}), threshold S_bt={S_BT}")

    # 标记是否breakthrough
    dfm["breakthrough"] = dfm["outlet_Smax"] >= S_BT

    print(dfm.head())
    print("Saved outputs will be in current directory.")

    if SAVE_CSV:
        out_csv = f"front_metrics_with_BT_S{S_STAR}_q{int(Q_FRONT*100)}_Sbt{S_BT}.csv"
        dfm.to_csv(out_csv, index=False)
        print(f"[CSV] {out_csv}")

    # 作图：前沿推进曲线 / outlet曲线
    plot_front_curve(dfm, S_star=S_STAR, q=Q_FRONT, save=SAVE_FIG)
    plot_outlet_curve(dfm, S_bt=S_BT, t_bt=t_bt, save=SAVE_FIG)

    # 作图：等值线前沿（默认只画快照，避免太多）
    plot_front_contours(model, times_snap, Nx=NX, Ny=NY, S_star=S_STAR, save=SAVE_FIG)

if __name__ == "__main__":
    print(f"DEVICE={DEVICE}, P_ref={P_ref:.3e} Pa, T_ref={T_ref:.3e} s")
    main()
