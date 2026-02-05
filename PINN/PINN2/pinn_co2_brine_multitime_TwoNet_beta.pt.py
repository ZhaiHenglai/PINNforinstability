import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Arc

# ====== 必须与训练时保持一致 ======
torch.set_default_dtype(torch.float64)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

L_ref = 5.0
T_ref = 10000

K = 1.0e-14
mu_w = 2.5e-4
U_in = 5.4e-5
p0 = 10.0e6

mu_ref = mu_w
U_ref = U_in
k_ref = K
P_ref = mu_ref * U_ref * L_ref / k_ref  # Pa

# 注入/出口几何（用于画弧线提示）
r_well = 0.5
inj_center = (0.0, 0.0)
out_center = (5.0, 5.0)

# ====== 你的网络结构（要和训练时一致）======
import torch.nn as nn
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
        self.beta = 30.0  # 推理时建议用训练末期的最大beta（让饱和度前沿更“硬”）
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

# ====== 加载训练好的权重 ======
model = TwoNetPINN(width=128, use_fourier=True).to(DEVICE)
state = torch.load("/home/henglai_pc/pythonTesst/pinn_co2_brine_multitime_TwoNet_beta.pt", map_location=DEVICE)
model.load_state_dict(state)
model.set_beta(30.0)
model.eval()





# ====== 在规则网格上预测（支持batch避免显存爆）======
@torch.no_grad()
def predict_grid(model, Nx=201, Ny=201, t_sec=0.0, batch=65536):
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
    # 注入：0~90度四分之一圆弧
    ax.add_patch(Arc(inj_center, 2*r_well, 2*r_well, angle=0, theta1=0, theta2=90, lw=2))
    # 出口：180~270度四分之一圆弧
    ax.add_patch(Arc(out_center, 2*r_well, 2*r_well, angle=0, theta1=180, theta2=270, lw=2))

def plot_snapshot(t_sec, Nx=201, Ny=201, save=False):
    XX, YY, p_phys, Sco2 = predict_grid(model, Nx=Nx, Ny=Ny, t_sec=t_sec)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    im0 = axes[0].imshow(p_phys/1e6, origin="lower", extent=[0,L_ref,0,L_ref], aspect="equal")
    axes[0].set_title(f"Pressure (MPa), t={t_sec:.2e} s")
    axes[0].set_xlabel("x (m)"); axes[0].set_ylabel("y (m)")
    draw_arcs(axes[0])
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(Sco2, origin="lower", extent=[0,L_ref,0,L_ref], aspect="equal", vmin=0, vmax=1)
    axes[1].set_title(f"Sco2, t={t_sec:.2e} s")
    axes[1].set_xlabel("x (m)"); axes[1].set_ylabel("y (m)")
    draw_arcs(axes[1])
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    if save:
        plt.savefig(f"snapshot_t_{t_sec:.3e}.png", dpi=200)
    plt.show()

# ====== 选几个时间点画图 ======
# 你可以改成你的数据真实时间：比如从df里取 unique times
times = [0.0, 0.2*T_ref, 0.4*T_ref, 0.6*T_ref, 0.8*T_ref, 1.0*T_ref]
for tsec in times:
    plot_snapshot(tsec, Nx=251, Ny=251, save=True)

# 假设你还在同一个脚本里有 df（和训练时一样）
# df 需要含 X,Y,phase1::Time,phase2::PhaseVolumeFraction,phase1::Pressure

