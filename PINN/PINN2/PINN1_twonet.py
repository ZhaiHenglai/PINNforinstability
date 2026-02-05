import os
import glob
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ============================================================
# dtype / device
# ============================================================
torch.set_default_dtype(torch.float64)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 0) 数据文件：多文件 glob
# ============================================================
DATA_GLOB = "/home/henglai_pc/pythonTesst/data/Test2/Visc_fingering_*_points.csv"


# ============================================================
# 1) 物理参数
# ============================================================
L_ref = 5.0          # m
T_ref = 1.0e5        # s
K = 1.0e-14          # m^2
phi = 0.2

rho_w_const = 1027.61
mu_w = 2.5e-4
mu_c = 2.25e-5

U_in = 5.4e-5        # m/s
p0 = 10.0e6          # Pa
p_out = 10.0e6       # Pa

# 几何：两段四分之一圆弧
r_well = 0.5
inj_center = (0.0, 0.0)   # 左下角
out_center = (5.0, 5.0)   # 右上角

# 无量纲压力尺度：Darcy 压降尺度
mu_ref = mu_w
U_ref = U_in
k_ref = K
P_ref = mu_ref * U_ref * L_ref / k_ref

# 出口无量纲压力
p_out_tilde = (p_out - p0) / P_ref  # 这里=0

# 时间项系数
A_time = L_ref / (U_ref * T_ref)

# 残余饱和度
Sw_irr = 0.2
Snr = 0.2
krw0 = 1.0
krc0 = 1.0
nw = 2.0
nc = 2.0


# ============================================================
# 2) 读多个表格文件并合并
# ============================================================
REQUIRED_COLS = [
    "X", "Y", "phase1::Pressure", "phase1::Time",
    "phase2::PhaseVolumeFraction", "phase2::Density",
    "phase1::Density", "phase1::Viscosity_0", "phase2::Viscosity_0"
]

def read_one_table(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except Exception:
        df = pd.read_csv(path, delim_whitespace=True)

    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{os.path.basename(path)} missing columns: {missing}")

    for c in REQUIRED_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=REQUIRED_COLS).reset_index(drop=True)
    df["__file__"] = os.path.basename(path)
    return df

def load_all_tables(glob_pattern: str):
    files = sorted(glob.glob(glob_pattern))
    if not files:
        raise FileNotFoundError(f"No files matched: {glob_pattern}")
    dfs = [read_one_table(f) for f in files]
    df_all = pd.concat(dfs, axis=0, ignore_index=True)
    return df_all, files

df, files = load_all_tables(DATA_GLOB)
print(f"Loaded {len(files)} files, total rows: {len(df)}")
print("Unique times:", df["phase1::Time"].nunique())
print("Median brine rho, mu:", float(df["phase1::Density"].median()), float(df["phase1::Viscosity_0"].median()))
print("Median CO2 rho, mu :", float(df["phase2::Density"].median()), float(df["phase2::Viscosity_0"].median()))


# ============================================================
# 3) CO2 密度 surrogate：rho_co2(p)（Chebyshev, 可导）
# ============================================================
def fit_rho_cheb(p_pa: np.ndarray, rho: np.ndarray,
                 deg_list=(10, 12, 14, 16, 18, 20),
                 n_val=5000, seed=0):
    rng = np.random.default_rng(seed)

    pmin = float(np.min(p_pa))
    pmax = float(np.max(p_pa))
    pmid = 0.5 * (pmin + pmax)
    prng = 0.5 * (pmax - pmin)

    order = np.argsort(p_pa)
    p_sorted = p_pa[order]
    rho_sorted = rho[order]

    rho_ref = float(np.interp(pmid, p_sorted, rho_sorted))
    rho_tilde = rho / rho_ref
    z = (p_pa - pmid) / prng

    pv = rng.uniform(pmin, pmax, size=n_val)
    rho_true = np.interp(pv, p_sorted, rho_sorted)
    zv = (pv - pmid) / prng

    best = None
    for deg in deg_list:
        coeffs = np.polynomial.chebyshev.chebfit(z, rho_tilde, deg=deg)
        rho_pred = rho_ref * np.polynomial.chebyshev.chebval(zv, coeffs)
        rel = np.abs((rho_pred - rho_true) / rho_true)
        max_rel = float(np.max(rel))
        rms_rel = float(np.sqrt(np.mean(rel**2)))
        if best is None or max_rel < best[0]:
            best = (max_rel, rms_rel, deg, coeffs, rho_ref, pmin, pmax)

    max_rel, rms_rel, deg, coeffs, rho_ref, pmin, pmax = best
    print(f"[EOS] Cheb deg={deg}, max_rel={max_rel:.3e}, rms_rel={rms_rel:.3e}, "
          f"range=[{pmin/1e6:.3f},{pmax/1e6:.3f}] MPa")
    return coeffs, rho_ref, pmin, pmax

p_data = df["phase1::Pressure"].to_numpy(dtype=np.float64)
rho_c_data = df["phase2::Density"].to_numpy(dtype=np.float64)

cheb_coeffs_np, rho_c_ref, eos_pmin, eos_pmax = fit_rho_cheb(p_data, rho_c_data)

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

rho_co2_model = ChebRhoCO2(cheb_coeffs_np, rho_c_ref, eos_pmin, eos_pmax).to(DEVICE)


# ============================================================
# 4) 相对渗透率：有效饱和度
# ============================================================
def relperm_from_Sco2(Sco2: torch.Tensor):
    Sw = 1.0 - Sco2
    denom = 1.0 - Sw_irr - Snr  # 0.6
    denom_t = Sco2.new_tensor(denom)

    Se_w = (Sw - Sw_irr) / denom_t
    Se_c = (Sco2 - Snr) / denom_t
    Se_w = torch.clamp(Se_w, 0.0, 1.0)
    Se_c = torch.clamp(Se_c, 0.0, 1.0)

    krw = krw0 * Se_w**nw
    krc = krc0 * Se_c**nc
    return krw, krc


# ============================================================
# 5) TwoNetPINN：压力网 + 饱和度网 + beta 退火接口
# ============================================================
class FourierFeatures(nn.Module):
    def __init__(self, in_dim=3, m=32, scale=6.0):
        super().__init__()
        B = torch.randn(in_dim, m, dtype=torch.float64) * scale
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
        Sco2 = torch.sigmoid(self.beta * s_logit)
        return p_tilde, Sco2


# ============================================================
# 6) autograd helper（二阶导需要 create_graph=True）
# ============================================================
def grad(u, x):
    return torch.autograd.grad(
        u, x,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True
    )[0]

def divergence(fx, fy, x, y):
    return grad(fx, x) + grad(fy, y)


# ============================================================
# 7) PDE residual（无量纲）
# ============================================================
def pde_residual(model, x_t, y_t, t_t):
    p_tilde, Sco2 = model(x_t, y_t, t_t)
    Sw = 1.0 - Sco2

    p_phys = p0 + P_ref * p_tilde

    rho_c = rho_co2_model(p_phys)                 # kg/m3
    rho_w = torch.full_like(rho_c, rho_w_const)   # constant

    rho_c_tilde = rho_c / rho_c_ref
    rho_w_tilde = rho_w / rho_w_const

    krw, krc = relperm_from_Sco2(Sco2)

    dpdx = grad(p_tilde, x_t)
    dpdy = grad(p_tilde, y_t)

    K_tilde = K / k_ref  # =1

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

    vtx = vwx + vcx
    vty = vwy + vcy
    return r_w, r_c, p_tilde, Sco2, vtx, vty, p_phys


# ============================================================
# 8) 监督数据：放 CPU（只把 batch 搬 GPU）  <-- (4)
# ============================================================
# 注意：不要 device=DEVICE，留在 CPU
x_cpu = torch.tensor(df["X"].to_numpy(), dtype=torch.float64).view(-1, 1) / L_ref
y_cpu = torch.tensor(df["Y"].to_numpy(), dtype=torch.float64).view(-1, 1) / L_ref
t_cpu = torch.tensor(df["phase1::Time"].to_numpy(), dtype=torch.float64).view(-1, 1) / T_ref

p_true_cpu = torch.tensor(df["phase1::Pressure"].to_numpy(), dtype=torch.float64).view(-1, 1)
p_tilde_true_cpu = (p_true_cpu - p0) / P_ref

Sco2_true_cpu = torch.tensor(df["phase2::PhaseVolumeFraction"].to_numpy(), dtype=torch.float64).view(-1, 1)

N_data = x_cpu.shape[0]

def sample_data_batch(batch_size=4096):
    idx = torch.randint(0, N_data, (batch_size,), device="cpu")
    xb = x_cpu[idx].to(DEVICE)
    yb = y_cpu[idx].to(DEVICE)
    tb = t_cpu[idx].to(DEVICE)
    pb = p_tilde_true_cpu[idx].to(DEVICE)
    sb = Sco2_true_cpu[idx].to(DEVICE)
    return xb, yb, tb, pb, sb


# ============================================================
# 9) 采样：PDE 内点 + BC/IC 点
# ============================================================
def sample_interior(N):
    x = torch.rand(N, 1, device=DEVICE) * L_ref
    y = torch.rand(N, 1, device=DEVICE) * L_ref
    t = torch.rand(N, 1, device=DEVICE) * T_ref
    xt = (x / L_ref).requires_grad_(True)
    yt = (y / L_ref).requires_grad_(True)
    tt = (t / T_ref).requires_grad_(True)
    return xt, yt, tt

def sample_injection_arc(N):
    """
    左下角注入井：中心 (0,0)，半径 r_well，theta in [0, pi/2]
    这里返回“域的外法向”：对内边界(孔)而言，外法向指向井内 = (-cos, -sin)
    注入进域内，则 v·n = -1  <-- (5)
    """
    theta = torch.rand(N, 1, device=DEVICE) * (0.5 * math.pi)
    x = inj_center[0] + r_well * torch.cos(theta)
    y = inj_center[1] + r_well * torch.sin(theta)
    t = torch.rand(N, 1, device=DEVICE) * T_ref

    xt = (x / L_ref).requires_grad_(True)
    yt = (y / L_ref).requires_grad_(True)
    tt = (t / T_ref).requires_grad_(True)

    # 域的外法向（指向井内）
    nx = -torch.cos(theta)
    ny = -torch.sin(theta)
    return xt, yt, tt, nx, ny

def sample_outlet_arc(N):
    """
    右上角产出井：中心 (5,5)，弧在域内的左下象限 theta in [pi, 3pi/2]
    """
    theta = math.pi + torch.rand(N, 1, device=DEVICE) * (0.5 * math.pi)
    x = out_center[0] + r_well * torch.cos(theta)
    y = out_center[1] + r_well * torch.sin(theta)
    t = torch.rand(N, 1, device=DEVICE) * T_ref
    xt = x / L_ref
    yt = y / L_ref
    tt = t / T_ref
    return xt, yt, tt

def sample_outer_boundary(N_each=600):
    # square boundary excluding corner arcs
    y1 = (0.5 + torch.rand(N_each, 1, device=DEVICE) * (5.0 - 0.5))
    x1 = torch.zeros_like(y1)
    n1 = torch.tensor([-1.0, 0.0], device=DEVICE, dtype=torch.float64).view(1, 2).repeat(N_each, 1)

    x2 = (0.5 + torch.rand(N_each, 1, device=DEVICE) * (5.0 - 0.5))
    y2 = torch.zeros_like(x2)
    n2 = torch.tensor([0.0, -1.0], device=DEVICE, dtype=torch.float64).view(1, 2).repeat(N_each, 1)

    y3 = torch.rand(N_each, 1, device=DEVICE) * 4.5
    x3 = torch.full_like(y3, 5.0)
    n3 = torch.tensor([1.0, 0.0], device=DEVICE, dtype=torch.float64).view(1, 2).repeat(N_each, 1)

    x4 = torch.rand(N_each, 1, device=DEVICE) * 4.5
    y4 = torch.full_like(x4, 5.0)
    n4 = torch.tensor([0.0, 1.0], device=DEVICE, dtype=torch.float64).view(1, 2).repeat(N_each, 1)

    x = torch.cat([x1, x2, x3, x4], dim=0)
    y = torch.cat([y1, y2, y3, y4], dim=0)
    n = torch.cat([n1, n2, n3, n4], dim=0)

    t = torch.rand(x.shape[0], 1, device=DEVICE) * T_ref

    xt = (x / L_ref).requires_grad_(True)
    yt = (y / L_ref).requires_grad_(True)
    tt = (t / T_ref).requires_grad_(True)
    nx = n[:, 0:1]
    ny = n[:, 1:2]
    return xt, yt, tt, nx, ny

def sample_initial(N):
    x = torch.rand(N, 1, device=DEVICE) * L_ref
    y = torch.rand(N, 1, device=DEVICE) * L_ref
    t = torch.zeros(N, 1, device=DEVICE)
    return x / L_ref, y / L_ref, t / T_ref


# ============================================================
# 10) 训练：分段 backward 累积梯度  <-- (3)
#     + peak 显存打印            <-- (7)
# ============================================================
def train():
    model = TwoNetPINN(width=128, use_fourier=True).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=2e-4)

    # 权重
    w_pde = 1.0
    w_data_p = 50.0
    w_data_s = 200.0
    w_inj_flux = 50.0
    w_inj_sat = 200.0
    w_out_p = 50.0
    w_noflow = 20.0
    w_ic = 200.0

    # beta 退火
    beta_max = 30.0
    warm_steps = 5000.0

    for it in range(1, 100001):

        # (7) 每 200 步测一次峰值显存
        do_mem = (torch.cuda.is_available() and (it % 200 == 0))
        if do_mem:
            torch.cuda.reset_peak_memory_stats()

        opt.zero_grad()

        beta = min(beta_max, 1.0 + (it / warm_steps) * (beta_max - 1.0))
        model.set_beta(beta)

        # -------------------------
        # 1) PDE interior（不改点数）
        # -------------------------
        xf, yf, tf = sample_interior(32000)
        r_w, r_c, *_ = pde_residual(model, xf, yf, tf)
        loss_pde = (r_w**2).mean() + (r_c**2).mean()
        (w_pde * loss_pde).backward()

        # -------------------------
        # 2) Data supervision（不改点数）
        # -------------------------
        xd, yd, td, ptd_true, Sd_true = sample_data_batch(8192)
        ptd_pred, Sd_pred = model(xd, yd, td)
        loss_data_p = ((ptd_pred - ptd_true)**2).mean()
        loss_data_s = ((Sd_pred - Sd_true)**2).mean()
        (w_data_p * loss_data_p + w_data_s * loss_data_s).backward()

        # -------------------------
        # 3) Injection arc BC（不改点数）
        #    由于返回的是“域外法向(指向井内)”，注入进域内 => v·n = -1
        # -------------------------
        xi, yi, ti, nxi, nyi = sample_injection_arc(3000)
        r_wi, r_ci, p_t_i, S_i, vtx_i, vty_i, *_ = pde_residual(model, xi, yi, ti)
        vn_in = vtx_i * nxi + vty_i * nyi
        loss_inj_flux = ((vn_in + 1.0)**2).mean()  # target = -1
        loss_inj_sat = ((S_i - 1.0)**2).mean()
        (w_inj_flux * loss_inj_flux + w_inj_sat * loss_inj_sat).backward()

        # -------------------------
        # 4) Outlet arc Dirichlet p（不改点数）
        # -------------------------
        xo, yo, to = sample_outlet_arc(3000)
        p_t_o, S_o = model(xo, yo, to)
        loss_out_p = ((p_t_o - p_out_tilde)**2).mean()
        (w_out_p * loss_out_p).backward()

        # -------------------------
        # 5) Outer boundary no-flow（不改点数）
        # -------------------------
        xb, yb, tb, nxb, nyb = sample_outer_boundary(1200)
        r_wb, r_cb, p_t_b, S_b, vtx_b, vty_b, *_ = pde_residual(model, xb, yb, tb)
        vn_b = vtx_b * nxb + vty_b * nyb
        loss_noflow = (vn_b**2).mean()
        (w_noflow * loss_noflow).backward()

        # -------------------------
        # 6) Initial condition（不改点数）
        # -------------------------
        x0, y0, t0 = sample_initial(8000)
        p_t_0, S_0 = model(x0, y0, t0)
        loss_ic = ((p_t_0 - 0.0)**2).mean() + ((S_0 - 0.0)**2).mean()
        (w_ic * loss_ic).backward()

        opt.step()

        if it % 200 == 0:
            # 只用于打印：detach 避免无意中保留图
            loss_total_val = (
                w_pde * loss_pde.detach()
                + w_data_p * loss_data_p.detach()
                + w_data_s * loss_data_s.detach()
                + w_inj_flux * loss_inj_flux.detach()
                + w_inj_sat * loss_inj_sat.detach()
                + w_out_p * loss_out_p.detach()
                + w_noflow * loss_noflow.detach()
                + w_ic * loss_ic.detach()
            ).item()

            msg = (
                f"it={it:6d} beta={beta:5.2f} loss={loss_total_val:.3e} | "
                f"pde={loss_pde.item():.2e} dataP={loss_data_p.item():.2e} dataS={loss_data_s.item():.2e} | "
                f"injF={loss_inj_flux.item():.2e} injS={loss_inj_sat.item():.2e} outP={loss_out_p.item():.2e} "
                f"noflow={loss_noflow.item():.2e} ic={loss_ic.item():.2e} | "
                f"A_time={A_time:.3f}"
            )
            if do_mem:
                peak_gb = torch.cuda.max_memory_allocated() / 1024**3
                msg += f" | peak_mem={peak_gb:.2f} GB"
            print(msg)

    torch.save(model.state_dict(), "pinn_co2_brine_multitime_TwoNet_beta.pt")
    print("Saved model to pinn_co2_brine_multitime_TwoNet_beta.pt")
    return model


if __name__ == "__main__":
    print(f"P_ref={P_ref:.3e} Pa, p_out_tilde={p_out_tilde:.3e}, A_time={A_time:.3f}")
    train()
