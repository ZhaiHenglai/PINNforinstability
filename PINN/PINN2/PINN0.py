import os
import glob
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

torch.set_default_dtype(torch.float64)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 0) 数据文件：一个时间点一个文件（改这里）
# ============================================================
DATA_GLOB = "./tables/*.csv"   # 也可以是 *.txt 或 *.tsv
# 如果你的文件是空格/Tab分隔但没有逗号，也能读（代码会自动尝试）


# ============================================================
# 1) 物理参数（按你的新设定）
# ============================================================
L_ref = 5.0          # m
T_ref = 1.0e5        # s  (你说 1e5 ~ L/U)
K = 1.0e-14          # m^2
phi = 0.2

rho_w_const = 1027.61
mu_w = 2.5e-4
mu_c = 2.25e-5

U_in = 5.4e-5        # m/s
p0 = 10.0e6          # Pa (initial)
p_out = 10.0e6       # Pa (outlet absolute, corrected)

# 几何：两段四分之一圆弧
r_well = 0.5
inj_center = (0.0, 0.0)
out_center = (5.0, 5.0)

# 无量纲压力尺度：Darcy压降尺度（参考 mu_w, U_in）
mu_ref = mu_w
U_ref = U_in
k_ref = K
P_ref = mu_ref * U_ref * L_ref / k_ref  # Pa

# 出口无量纲压力
p_out_tilde = (p_out - p0) / P_ref  # 现在等于0

# 时间项系数（严格正确；此时约等于1）
A_time = L_ref / (U_ref * T_ref)

# 残余饱和度（你给）
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

def load_all_tables(glob_pattern: str) -> pd.DataFrame:
    files = sorted(glob.glob(glob_pattern))
    if not files:
        raise FileNotFoundError(f"No files matched: {glob_pattern}")

    dfs = []
    for f in files:
        dfs.append(read_one_table(f))
    df_all = pd.concat(dfs, axis=0, ignore_index=True)
    return df_all, files

df, files = load_all_tables(DATA_GLOB)
print(f"Loaded {len(files)} files, total rows: {len(df)}")
print("Unique times:", df["phase1::Time"].nunique())

# 可选 sanity check：物性是否恒定
print("Median brine rho, mu:", float(df["phase1::Density"].median()), float(df["phase1::Viscosity_0"].median()))
print("Median CO2 rho, mu :", float(df["phase2::Density"].median()), float(df["phase2::Viscosity_0"].median()))


# ============================================================
# 3) 用所有数据拟合 CO2 密度 surrogate：rho_co2(p)（可导 Chebyshev）
# ============================================================
def fit_rho_cheb(p_pa: np.ndarray, rho: np.ndarray, deg_list=(10,12,14,16,18,20), n_val=5000, seed=0):
    """
    Fit rho(p) over data pressure range using Chebyshev in z∈[-1,1].
    Select degree by minimizing max relative error on a validation set
    sampled uniformly in pressure range and evaluated by linear interp
    of the data (保守；你也可以换成CoolProp真值).
    """
    rng = np.random.default_rng(seed)

    pmin = float(np.min(p_pa))
    pmax = float(np.max(p_pa))
    pmid = 0.5*(pmin + pmax)
    prng = 0.5*(pmax - pmin)

    # 用中点插值密度做 rho_ref
    order = np.argsort(p_pa)
    p_sorted = p_pa[order]
    rho_sorted = rho[order]
    rho_ref = float(np.interp(pmid, p_sorted, rho_sorted))
    rho_tilde = rho / rho_ref
    z = (p_pa - pmid) / prng

    # 验证点：在 [pmin,pmax] 均匀取
    pv = rng.uniform(pmin, pmax, size=n_val)
    # 用数据插值得到“真值”（如果你有CoolProp可在这里换成CoolProp）
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
    print(f"[EOS] Cheb deg={deg}, max_rel={max_rel:.3e}, rms_rel={rms_rel:.3e}, range=[{pmin/1e6:.3f},{pmax/1e6:.3f}] MPa")
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
        self.pmid = 0.5*(self.pmin + self.pmax)
        self.prng = 0.5*(self.pmax - self.pmin)

    def _chebval_clenshaw(self, z):
        c = self.c
        if c.numel() == 1:
            return c[0] + 0.0*z
        b1 = torch.zeros_like(z)
        b2 = torch.zeros_like(z)
        for a in torch.flip(c[1:], dims=[0]):
            b0 = 2.0*z*b1 - b2 + a
            b2 = b1
            b1 = b0
        return z*b1 - b2 + c[0]

    def forward(self, p_pa):
        p_pa = p_pa.to(dtype=torch.float64)
        # 将 p 映射到 z，做轻微裁剪避免训练早期外推炸掉
        z = (p_pa - self.pmid) / self.prng
        z = torch.clamp(z, -1.0 + self.eps, 1.0 - self.eps)
        rho_tilde = self._chebval_clenshaw(z)
        return self.rho_ref * rho_tilde

rho_co2_model = ChebRhoCO2(cheb_coeffs_np, rho_c_ref, eos_pmin, eos_pmax).to(DEVICE)


# ============================================================
# 4) 相对渗透率：用有效饱和度（Sw_irr=Snr=0.2）
# ============================================================
def relperm_from_Sco2(Sco2: torch.Tensor):
    Sw = 1.0 - Sco2
    denom = 1.0 - Sw_irr - Snr  # 0.6
    denom_t = torch.tensor(denom, device=Sco2.device, dtype=Sco2.dtype)

    Se_w = (Sw - Sw_irr) / denom_t
    Se_c = (Sco2 - Snr) / denom_t
    Se_w = torch.clamp(Se_w, 0.0, 1.0)
    Se_c = torch.clamp(Se_c, 0.0, 1.0)

    krw = krw0 * Se_w**nw
    krc = krc0 * Se_c**nc
    return krw, krc


# ============================================================
# 5) PINN 网络：输入无量纲(x~,y~,t~)，内部归一化到[-1,1]
# ============================================================
class FourierFeatures(nn.Module):
    def __init__(self, in_dim=3, m=32, scale=6.0):
        super().__init__()
        B = torch.randn(in_dim, m) * scale
        self.register_buffer("B", B)

    def forward(self, x):
        proj = x @ self.B
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)

class PINN(nn.Module):
    def __init__(self, width=128, depth=8, use_fourier=True):
        super().__init__()
        self.use_fourier = use_fourier
        if use_fourier:
            self.ff = FourierFeatures(3, 32, 6.0)
            in_dim = 64
        else:
            in_dim = 3

        layers = [nn.Linear(in_dim, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        self.backbone = nn.Sequential(*layers)

        self.head_p = nn.Linear(width, 1)
        self.head_s = nn.Linear(width, 1)

    def forward(self, x_t, y_t, t_t):
        # x_t,y_t,t_t: nondim in [0,1] ideally
        x_in = 2.0 * x_t - 1.0
        y_in = 2.0 * y_t - 1.0
        t_in = 2.0 * t_t - 1.0
        X = torch.cat([x_in, y_in, t_in], dim=1)

        Z = self.ff(X) if self.use_fourier else X
        h = self.backbone(Z)
        p_tilde = self.head_p(h)
        Sco2 = torch.sigmoid(self.head_s(h))  # keep [0,1] to match your data
        return p_tilde, Sco2


# ============================================================
# 6) autograd helper
# ============================================================
def grad(u, x):
    return torch.autograd.grad(
        u, x, grad_outputs=torch.ones_like(u),
        create_graph=True, retain_graph=True
    )[0]

def divergence(fx, fy, x, y):
    return grad(fx, x) + grad(fy, y)


# ============================================================
# 7) PDE residual（无量纲）
#    r = phi*A_time ∂t~(rho~S) + div~(rho~ v~) = 0
#    v~ = -K~*(mu_ref/mu)*kr * grad~(p~)
# ============================================================
def pde_residual(model, x_t, y_t, t_t):
    p_tilde, Sco2 = model(x_t, y_t, t_t)
    Sw = 1.0 - Sco2

    p_phys = p0 + P_ref * p_tilde

    rho_c = rho_co2_model(p_phys)                 # kg/m3 (differentiable)
    rho_w = torch.full_like(rho_c, rho_w_const)   # constant

    rho_c_tilde = rho_c / rho_c_ref
    rho_w_tilde = rho_w / rho_w_const  # =1

    krw, krc = relperm_from_Sco2(Sco2)

    dpdx = grad(p_tilde, x_t)
    dpdy = grad(p_tilde, y_t)

    K_tilde = K / k_ref  # 1

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
# 8) 监督数据张量（多时间点合并）
# ============================================================
x_t = torch.tensor(df["X"].to_numpy(), device=DEVICE).view(-1,1) / L_ref
y_t = torch.tensor(df["Y"].to_numpy(), device=DEVICE).view(-1,1) / L_ref
t_t = torch.tensor(df["phase1::Time"].to_numpy(), device=DEVICE).view(-1,1) / T_ref

p_true = torch.tensor(df["phase1::Pressure"].to_numpy(), device=DEVICE).view(-1,1)
p_tilde_true = (p_true - p0) / P_ref

Sco2_true = torch.tensor(df["phase2::PhaseVolumeFraction"].to_numpy(), device=DEVICE).view(-1,1)

N_data = x_t.shape[0]

def sample_data_batch(batch_size=4096):
    idx = torch.randint(0, N_data, (batch_size,), device=DEVICE)
    return x_t[idx], y_t[idx], t_t[idx], p_tilde_true[idx], Sco2_true[idx]


# ============================================================
# 9) 采样：PDE 内点 + BC/IC 点
# ============================================================
def sample_interior(N):
    x = torch.rand(N,1, device=DEVICE) * L_ref
    y = torch.rand(N,1, device=DEVICE) * L_ref
    t = torch.rand(N,1, device=DEVICE) * T_ref
    xt = (x / L_ref).requires_grad_(True)
    yt = (y / L_ref).requires_grad_(True)
    tt = (t / T_ref).requires_grad_(True)
    return xt, yt, tt

def sample_injection_arc(N):
    theta = torch.rand(N,1, device=DEVICE) * (0.5*math.pi)
    x = r_well * torch.cos(theta)
    y = r_well * torch.sin(theta)
    t = torch.rand(N,1, device=DEVICE) * T_ref

    xt = (x / L_ref).requires_grad_(True)
    yt = (y / L_ref).requires_grad_(True)
    tt = (t / T_ref).requires_grad_(True)

    nx = torch.cos(theta)
    ny = torch.sin(theta)
    return xt, yt, tt, nx, ny

def sample_outlet_arc(N):
    theta = math.pi + torch.rand(N,1, device=DEVICE) * (0.5*math.pi)
    x = out_center[0] + r_well * torch.cos(theta)
    y = out_center[1] + r_well * torch.sin(theta)
    t = torch.rand(N,1, device=DEVICE) * T_ref
    xt = x / L_ref
    yt = y / L_ref
    tt = t / T_ref
    return xt, yt, tt

def sample_outer_boundary(N_each=600):
    # square boundary excluding corner arcs
    y1 = (0.5 + torch.rand(N_each,1, device=DEVICE)*(5.0-0.5))
    x1 = torch.zeros_like(y1)
    n1 = torch.tensor([-1.0, 0.0], device=DEVICE).view(1,2).repeat(N_each,1)

    x2 = (0.5 + torch.rand(N_each,1, device=DEVICE)*(5.0-0.5))
    y2 = torch.zeros_like(x2)
    n2 = torch.tensor([0.0, -1.0], device=DEVICE).view(1,2).repeat(N_each,1)

    y3 = torch.rand(N_each,1, device=DEVICE) * 4.5
    x3 = torch.full_like(y3, 5.0)
    n3 = torch.tensor([1.0, 0.0], device=DEVICE).view(1,2).repeat(N_each,1)

    x4 = torch.rand(N_each,1, device=DEVICE) * 4.5
    y4 = torch.full_like(x4, 5.0)
    n4 = torch.tensor([0.0, 1.0], device=DEVICE).view(1,2).repeat(N_each,1)

    x = torch.cat([x1,x2,x3,x4], dim=0)
    y = torch.cat([y1,y2,y3,y4], dim=0)
    n = torch.cat([n1,n2,n3,n4], dim=0)

    t = torch.rand(x.shape[0],1, device=DEVICE) * T_ref

    xt = (x / L_ref).requires_grad_(True)
    yt = (y / L_ref).requires_grad_(True)
    tt = (t / T_ref).requires_grad_(True)
    nx = n[:,0:1]
    ny = n[:,1:2]
    return xt, yt, tt, nx, ny

def sample_initial(N):
    x = torch.rand(N,1, device=DEVICE) * L_ref
    y = torch.rand(N,1, device=DEVICE) * L_ref
    t = torch.zeros(N,1, device=DEVICE)
    return x/L_ref, y/L_ref, t/T_ref


# ============================================================
# 10) 训练：PDE + 数据监督 + BC/IC
# ============================================================
def train():
    model = PINN(width=128, depth=8, use_fourier=True).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=2e-4)

    # 权重（可调）
    w_pde = 1.0
    w_data_p = 50.0
    w_data_s = 200.0
    w_inj_flux = 50.0
    w_inj_sat = 200.0
    w_out_p = 50.0
    w_noflow = 20.0
    w_ic = 200.0

    for it in range(1, 20001):
        opt.zero_grad()

        # PDE
        xf, yf, tf = sample_interior(8000)
        r_w, r_c, p_t_f, S_f, vtx_f, vty_f, p_phys_f = pde_residual(model, xf, yf, tf)
        loss_pde = (r_w**2).mean() + (r_c**2).mean()

        # 数据监督
        xd, yd, td, ptd_true, Sd_true = sample_data_batch(4096)
        ptd_pred, Sd_pred = model(xd, yd, td)
        loss_data_p = ((ptd_pred - ptd_true)**2).mean()
        loss_data_s = ((Sd_pred - Sd_true)**2).mean()

        # 注入弧：总法向通量=U_in -> 无量纲=1；且 Sco2=1
        xi, yi, ti, nxi, nyi = sample_injection_arc(1500)
        r_wi, r_ci, p_t_i, S_i, vtx_i, vty_i, p_phys_i = pde_residual(model, xi, yi, ti)
        vn_in = vtx_i*nxi + vty_i*nyi
        loss_inj_flux = ((vn_in - 1.0)**2).mean()
        loss_inj_sat = ((S_i - 1.0)**2).mean()

        # 产出弧：p=10MPa -> p_tilde=0
        xo, yo, to = sample_outlet_arc(1500)
        p_t_o, S_o = model(xo, yo, to)
        loss_out_p = ((p_t_o - p_out_tilde)**2).mean()

        # 外边界无流：v_t · n = 0
        xb, yb, tb, nxb, nyb = sample_outer_boundary(600)
        r_wb, r_cb, p_t_b, S_b, vtx_b, vty_b, p_phys_b = pde_residual(model, xb, yb, tb)
        vn_b = vtx_b*nxb + vty_b*nyb
        loss_noflow = (vn_b**2).mean()

        # 初值：t=0, p=p0 -> p_tilde=0, Sco2=0
        x0, y0, t0 = sample_initial(4000)
        p_t_0, S_0 = model(x0, y0, t0)
        loss_ic = ((p_t_0 - 0.0)**2).mean() + ((S_0 - 0.0)**2).mean()

        loss = (
            w_pde * loss_pde
            + w_data_p * loss_data_p
            + w_data_s * loss_data_s
            + w_inj_flux * loss_inj_flux
            + w_inj_sat  * loss_inj_sat
            + w_out_p    * loss_out_p
            + w_noflow   * loss_noflow
            + w_ic       * loss_ic
        )

        loss.backward()
        opt.step()

        if it % 200 == 0:
            print(
                f"it={it:6d} loss={loss.item():.3e} | "
                f"pde={loss_pde.item():.2e} dataP={loss_data_p.item():.2e} dataS={loss_data_s.item():.2e} | "
                f"injF={loss_inj_flux.item():.2e} injS={loss_inj_sat.item():.2e} outP={loss_out_p.item():.2e} "
                f"noflow={loss_noflow.item():.2e} ic={loss_ic.item():.2e} | "
                f"A_time={A_time:.3f}"
            )

    torch.save(model.state_dict(), "pinn_co2_brine_multitime.pt")
    print("Saved model to pinn_co2_brine_multitime.pt")
    return model


if __name__ == "__main__":
    print(f"P_ref={P_ref:.3e} Pa, p_out_tilde={p_out_tilde:.3e}, A_time={A_time:.3f}")
    train()
