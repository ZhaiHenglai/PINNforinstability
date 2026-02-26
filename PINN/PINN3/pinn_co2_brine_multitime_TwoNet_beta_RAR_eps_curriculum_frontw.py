"""
PINN training script for 2D brine–CO2 two-phase flow (square domain with two well holes).

What this file is
-----------------
This is the exact training script I use in my own work, rewritten with clearer comments so that
I can hand it to someone else and explain the logic step-by-step.

Key design choices (important for large datasets)
-------------------------------------------------
1) Dataset input is a single binary file produced by `torch.save(pack, ...)` (see `DATA_PT_CANDIDATES`).
   The cached arrays may be stored as NumPy arrays OR as torch.Tensors; both are supported.
2) When N is huge (e.g. 1e8 rows), never convert the whole dataset to float64.
   - The full dataset stays as float32 on CPU to keep RAM reasonable.
   - Each mini-batch is copied to GPU and cast to float64 to match the PINN (float64 helps 2nd derivatives).
3) CO2 density rho(p) is fitted with a Chebyshev surrogate on a *random subset* of (p, rho) pairs.
   Fitting on all rows is unnecessary and often impossible in memory.

Physics / modelling notes
-------------------------
- The PINN solves two mass conservation residuals (brine and CO2) using Darcy velocities.
- Saturation output uses residual-saturation “method B”: Sco2 ∈ [Snr, 1 - Sw_irr].
- An optional diffusion (artificial viscosity) term is applied *consistently* between phases:
    r_co2 -= eps * ΔSco2,   r_brine += eps * ΔSco2   (because Sw = 1 - Sco2).
- Interior / IC / RAR sampling excludes the well holes (the PDE is not valid inside the holes).

How to run
----------
1) Put your binary dataset next to this script as `tables_cache_tensor.pt` or `tables_cache.pt`.
   Or set an absolute path via:  PINN_DATA_PT=/abs/path/to/file.pt
2) Run:  python this_script.py

Practical tuning knobs
----------------------
- `EOS_FIT_SAMPLE`: how many (p, rho_c) pairs to fit the EOS surrogate (1e6–5e6 is typical).
- Loss weights and schedules are in `train()`.
"""


# -----------------------------------------------------------------------------
# Imports and numeric setup
# -----------------------------------------------------------------------------
import os
import glob
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# We keep the PINN in float64 because we need stable second derivatives (Laplacian, gradients, etc.).
# The dataset itself stays float32 on CPU; we only cast batches to float64 on the GPU.
torch.set_default_dtype(torch.float64)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------------------------------------------------------
# Dataset configuration
# -----------------------------------------------------------------------------
# The script will use the first existing file in this list.
# You can also override the path with the environment variable PINN_DATA_PT.
DATA_PT_CANDIDATES = [
    "/home/henglai_pc/pythonTesst/PINN3/tables_cache_tensor.pt",
    "tables_cache.pt",
]
# CSV fallback (only used if the binary cache is not found).
DATA_GLOB = "./tables/*.csv"


# Number of paired (p, rho_c) samples used to fit the rho_co2(p) surrogate.
# Increase for higher accuracy; decrease if you run out of RAM during EOS fitting.
EOS_FIT_SAMPLE = 2_000_000  # number of paired samples used for EOS fitting


# -----------------------------------------------------------------------------
# Physical parameters and nondimensionalization
# -----------------------------------------------------------------------------
# The network uses dimensionless inputs (x/L_ref, y/L_ref, t/T_ref) and predicts dimensionless pressure p~.
# `P_ref` is the Darcy pressure-drop scale: P_ref = mu_ref * U_ref * L_ref / k_ref.
L_ref = 5.0  # [m]
T_ref = 1.0e5  # [s]
K = 1.0e-14  # [m^2]
# Domain size: [0, L_ref] x [0, L_ref] (meters)  |  Time scale: T_ref (seconds)
# Porosity is constant here.
phi = 0.2


# Fluid properties used inside the PDE.
# Note: the dataset may contain varying rho/mu, but this PINN currently uses constants for mu_w, mu_c, and rho_w.
# CO2 density is handled separately via a fitted EOS surrogate rho_co2(p).
rho_w_const = 1027.61
mu_w = 2.5e-4
mu_c = 2.25e-5

U_in = 5.4e-5  # [m/s] injection velocity scale used in nondimensionalization
p0 = 10.0e6  # [Pa] reference/initial pressure
p_out = 10.0e6  # [Pa] production boundary pressure


# Geometry: the square domain has two quarter-circle well holes removed.
# Injection well hole: center inj_center, radius r_well, arc angles (0, pi/2).
# Production well hole: center out_center, radius r_well, arc angles (pi, 3pi/2).
r_well = 0.5
inj_center = (0.0, 0.0)
out_center = (5.0, 5.0)

mu_ref = mu_w
U_ref = U_in
k_ref = K

# Pressure/time scaling for the dimensionless PDE:
#   p_tilde = (p - p0) / P_ref
#   A_time = L_ref / (U_ref * T_ref)
P_ref = mu_ref * U_ref * L_ref / k_ref

p_out_tilde = (p_out - p0) / P_ref
A_time = L_ref / (U_ref * T_ref)

Sw_irr = 0.2
Snr = 0.2
krw0 = 1.0
krc0 = 1.0
nw = 2.0
nc = 2.0


# -----------------------------------------------------------------------------
# Dataset loading utilities
# -----------------------------------------------------------------------------
# Binary mode: expects a dict with `pack["arrays"][key]` for the keys listed below.
# CSV mode: only for debugging / small datasets; huge datasets should always use the binary cache.
REQUIRED_COLS = [
    "X", "Y", "phase1::Pressure", "phase1::Time",
    "phase2::PhaseVolumeFraction", "phase2::Density",
    "phase1::Density", "phase1::Viscosity_0", "phase2::Viscosity_0",
]


def read_one_table(path: str) -> pd.DataFrame:
    """Read one snapshot table (CSV or whitespace-delimited) and validate columns."""
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


def _pick_dataset_path() -> str | None:
    for p in DATA_PT_CANDIDATES:
        if os.path.exists(p):
            return p
    envp = os.environ.get("PINN_DATA_PT", "").strip()
    if envp and os.path.exists(envp):
        return envp
    return None


def load_binary_cache(path: str):
    pack = torch.load(path, map_location="cpu")
    if not isinstance(pack, dict) or "arrays" not in pack:
        raise ValueError(f"{path} is not a valid dataset pack (missing 'arrays').")

    arrays = pack["arrays"]
    needed = ["x", "y", "t", "p", "Sco2", "rho_c", "rho_w", "mu_w", "mu_c"]
    missing = [k for k in needed if k not in arrays]
    if missing:
        raise ValueError(f"{path} is missing array keys: {missing}")

    file_id_to_name = pack.get("file_id_to_name", None)
    N = int(pack.get("N", len(arrays["x"])))

    time_unique = pack.get("time_unique", None)
    if time_unique is None:
        t = arrays["t"]
        if isinstance(t, torch.Tensor):
            time_unique = torch.unique(t.detach().cpu()).numpy()
        else:
            time_unique = np.unique(np.asarray(t))

    if isinstance(time_unique, torch.Tensor):
        time_unique = time_unique.detach().cpu().numpy()

    return arrays, file_id_to_name, N, time_unique


# `load_dataset()` chooses binary cache first; if not found it falls back to reading all CSV files.
def load_dataset():
    path = _pick_dataset_path()
    if path is not None:
        arrays, file_id_to_name, N, time_unique = load_binary_cache(path)
        print(f"Loaded binary dataset: {os.path.abspath(path)} (N={N})")
        print("Unique times:", int(np.asarray(time_unique).shape[0]))
        return arrays, file_id_to_name, N, time_unique

    df, files = load_all_tables(DATA_GLOB)
    print(f"Loaded {len(files)} CSV files, total rows: {len(df)}")
    time_unique = np.unique(df["phase1::Time"].to_numpy(np.float32))
    print("Unique times:", int(time_unique.shape[0]))

    arr = {
        "x": df["X"].to_numpy(np.float32),
        "y": df["Y"].to_numpy(np.float32),
        "t": df["phase1::Time"].to_numpy(np.float32),
        "p": df["phase1::Pressure"].to_numpy(np.float32),
        "Sco2": df["phase2::PhaseVolumeFraction"].to_numpy(np.float32),
        "rho_c": df["phase2::Density"].to_numpy(np.float32),
        "rho_w": df["phase1::Density"].to_numpy(np.float32),
        "mu_w": df["phase1::Viscosity_0"].to_numpy(np.float32),
        "mu_c": df["phase2::Viscosity_0"].to_numpy(np.float32),
    }

    files_cat = df["__file__"].astype("category")
    arr["file_id"] = files_cat.cat.codes.to_numpy(np.int32)
    file_id_to_name = list(files_cat.cat.categories)
    return arr, file_id_to_name, int(len(df)), time_unique


# Load the dataset once at startup. We keep it on CPU; training uses random mini-batches.
arrays, file_id_to_name, N_rows, time_unique = load_dataset()

def _median(x):
    if isinstance(x, torch.Tensor):
        return float(x.detach().cpu().to(torch.float64).median().item())
    return float(np.median(np.asarray(x)))

print("Median brine rho, mu:", _median(arrays["rho_w"]), _median(arrays["mu_w"]))
print("Median CO2  rho, mu:", _median(arrays["rho_c"]), _median(arrays["mu_c"]))


# -----------------------------------------------------------------------------
# EOS (rho_co2(p)) fitting
# -----------------------------------------------------------------------------
# We only need a 1D relationship rho(p). Fitting from *all* rows is wasteful and may be impossible for N~1e8.
# The important thing is that p and rho_c are sampled with the SAME random indices (paired sampling).
def _sample_pair_numpy(p, rho, n_sample: int, seed: int = 0):
    """Sample matched pairs (p, rho) and return numpy float64 arrays."""
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
    """Average duplicate pressures to get a clean p->rho curve."""
    dtmp = pd.DataFrame({"p": p_pa, "rho": rho})
    g = dtmp.groupby("p", as_index=False)["rho"].mean().sort_values("p")
    p_u = g["p"].to_numpy(dtype=np.float64)
    rho_u = g["rho"].to_numpy(dtype=np.float64)
    return p_u, rho_u


def fit_rho_cheb(p_pa: np.ndarray, rho: np.ndarray,
                 deg_list=(10, 12, 14, 16, 18, 20),
                 n_val=20000, seed=0):
    """Fit Chebyshev series for rho(p) with validation on random points."""
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
    print(
        f"[EOS] Cheb deg={deg}, max_rel={max_rel:.3e}, rms_rel={rms_rel:.3e}, "
        f"range=[{pmin/1e6:.3f},{pmax/1e6:.3f}] MPa (fit sample={len(p_pa)})"
    )
    return coeffs, rho_ref, pmin, pmax


p_s, rho_c_s = _sample_pair_numpy(arrays["p"], arrays["rho_c"], EOS_FIT_SAMPLE, seed=0)
cheb_coeffs_np, rho_c_ref, eos_pmin, eos_pmax = fit_rho_cheb(p_s, rho_c_s)


# After fitting Chebyshev coefficients in NumPy, we wrap them in a small torch module so that rho(p)
# is differentiable with respect to p (needed in the PINN residual).
class ChebRhoCO2(nn.Module):
    """Differentiable CO2 density surrogate rho(p) via Chebyshev polynomials (Clenshaw evaluation)."""

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


# -----------------------------------------------------------------------------
# Relative permeability model (Corey-type) using effective saturations
# -----------------------------------------------------------------------------
# We clamp effective saturations to [0,1] to keep the relperm model well-behaved during early training.
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


# -----------------------------------------------------------------------------
# Neural networks: pressure net + saturation net
# -----------------------------------------------------------------------------
# I split pressure and saturation into two separate MLPs because they have very different behaviors:
# - pressure is smoother, saturation can develop sharp fronts
# Fourier features often help represent multiscale behavior with fewer layers.
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


# -----------------------------------------------------------------------------
# Autograd helpers
# -----------------------------------------------------------------------------
# PINNs need first and second derivatives with respect to inputs. We use torch.autograd.grad with create_graph=True.
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


# -----------------------------------------------------------------------------
# PDE residual (dimensionless form)
# -----------------------------------------------------------------------------
# The model outputs p_tilde and Sco2. We compute Darcy velocities for each phase and then the two mass residuals.
# eps>0 adds an artificial diffusion term on saturation (complementary-consistent between phases).
def pde_residual(model, x_t, y_t, t_t, eps=0.0):
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

    vtx = vwx + vcx
    vty = vwy + vcy

    return r_w, r_c, p_tilde, Sco2, vtx, vty, p_phys


# -----------------------------------------------------------------------------
# Supervised data: big-data safe batching
# -----------------------------------------------------------------------------
# We store x,y,t,p,S on CPU float32. Each iteration samples indices on CPU, then moves only the batch to GPU.
# This avoids allocating (and duplicating) a full float64 dataset in RAM.
def _as_cpu_float32_col(v):
    """Return a CPU float32 tensor of shape [N,1] without duplicating data if possible."""
    if isinstance(v, torch.Tensor):
        t = v.detach()
        if t.device.type != "cpu":
            t = t.cpu()
        if t.dtype != torch.float32:
            t = t.to(torch.float32)
        return t.view(-1, 1)
    a = np.asarray(v)
    if a.dtype != np.float32:
        a = a.astype(np.float32, copy=False)
    return torch.from_numpy(a).view(-1, 1)


x_all = _as_cpu_float32_col(arrays["x"]) / float(L_ref)
y_all = _as_cpu_float32_col(arrays["y"]) / float(L_ref)
t_all = _as_cpu_float32_col(arrays["t"]) / float(T_ref)

p_all = _as_cpu_float32_col(arrays["p"])  # [Pa]
Sco2_raw_all = _as_cpu_float32_col(arrays["Sco2"])  # raw CO2 volume fraction (assumed in [0,1])

N_data = int(x_all.shape[0])


def sample_data_batch(batch_size=8192):
    """Sample a mini-batch; move to DEVICE and cast to float64 to match the model."""
    idx = torch.randint(0, N_data, (batch_size,), device="cpu", dtype=torch.int64)

    xd = x_all.index_select(0, idx).to(DEVICE, dtype=torch.float64)
    yd = y_all.index_select(0, idx).to(DEVICE, dtype=torch.float64)
    td = t_all.index_select(0, idx).to(DEVICE, dtype=torch.float64)

    p_true = p_all.index_select(0, idx).to(DEVICE, dtype=torch.float64)
    ptd = (p_true - p0) / P_ref

    Sco2_raw = Sco2_raw_all.index_select(0, idx).to(DEVICE, dtype=torch.float64)
    Sco2_raw = torch.clamp(Sco2_raw, 0.0, 1.0)
    Sd = Snr + (1.0 - Sw_irr - Snr) * Sco2_raw

    return xd, yd, td, ptd, Sd


_L = float(L_ref)
_R = float(r_well)
_T = float(T_ref)
_inj_cx, _inj_cy = float(inj_center[0]), float(inj_center[1])
_out_cx, _out_cy = float(out_center[0]), float(out_center[1])
_GEOM_EPS = 1e-9


def _outside_well_holes_xy_phys(x_phys: torch.Tensor, y_phys: torch.Tensor) -> torch.Tensor:
    dx1 = x_phys - _inj_cx
    dy1 = y_phys - _inj_cy
    dx2 = x_phys - _out_cx
    dy2 = y_phys - _out_cy
    in_inj = (dx1 * dx1 + dy1 * dy1) < (_R * _R)
    in_out = (dx2 * dx2 + dy2 * dy2) < (_R * _R)
    return ~(in_inj | in_out).squeeze(1)


def _rejection_sample_xy(N: int, oversample: float = 1.20):
    if N <= 0:
        raise ValueError("N must be positive")

    xs = []
    ys = []
    got = 0

    while got < N:
        M = int((N - got) * oversample) + 128
        x = torch.rand(M, 1, device=DEVICE, dtype=torch.get_default_dtype()) * _L
        y = torch.rand(M, 1, device=DEVICE, dtype=torch.get_default_dtype()) * _L
        mask = _outside_well_holes_xy_phys(x, y)
        if mask.any():
            x_ok = x[mask]
            y_ok = y[mask]
            xs.append(x_ok)
            ys.append(y_ok)
            got += x_ok.shape[0]

    x_all_phys = torch.cat(xs, dim=0)[:N]
    y_all_phys = torch.cat(ys, dim=0)[:N]
    return x_all_phys, y_all_phys


# -----------------------------------------------------------------------------
# Collocation / boundary sampling for the PINN
# -----------------------------------------------------------------------------
# All interior/IC/RAR points must exclude the well holes because the PDE is not defined inside the holes.
# Coordinates returned to the model are dimensionless: x/L_ref, y/L_ref, t/T_ref.
def sample_interior(N):
    x, y = _rejection_sample_xy(N)
    t = torch.rand(N, 1, device=DEVICE, dtype=torch.get_default_dtype()) * _T
    xt = (x / _L).requires_grad_(True)
    yt = (y / _L).requires_grad_(True)
    tt = (t / _T).requires_grad_(True)
    return xt, yt, tt


def sample_initial(N):
    x, y = _rejection_sample_xy(N)
    t = torch.zeros(N, 1, device=DEVICE, dtype=torch.get_default_dtype())
    return x / _L, y / _L, t / _T


def sample_injection_arc(N):
    theta = _GEOM_EPS + torch.rand(N, 1, device=DEVICE, dtype=torch.get_default_dtype()) * (0.5 * math.pi - 2.0 * _GEOM_EPS)
    x = _inj_cx + _R * torch.cos(theta)
    y = _inj_cy + _R * torch.sin(theta)
    t = torch.rand(N, 1, device=DEVICE, dtype=torch.get_default_dtype()) * _T
    xt = (x / _L).requires_grad_(True)
    yt = (y / _L).requires_grad_(True)
    tt = (t / _T).requires_grad_(True)
    nx = torch.cos(theta)
    ny = torch.sin(theta)
    return xt, yt, tt, nx, ny


def sample_outlet_arc(N):
    theta = math.pi + _GEOM_EPS + torch.rand(N, 1, device=DEVICE, dtype=torch.get_default_dtype()) * (0.5 * math.pi - 2.0 * _GEOM_EPS)
    x = _out_cx + _R * torch.cos(theta)
    y = _out_cy + _R * torch.sin(theta)
    t = torch.rand(N, 1, device=DEVICE, dtype=torch.get_default_dtype()) * _T
    return x / _L, y / _L, t / _T


def sample_outer_boundary(N_each=600):
    """No-flow boundary on the square edges, skipping only the segments covered by well arcs."""
    dtype = torch.get_default_dtype()

    y1 = (_R + _GEOM_EPS) + torch.rand(N_each, 1, device=DEVICE, dtype=dtype) * (_L - _R - 2.0 * _GEOM_EPS)
    x1 = torch.zeros_like(y1)
    n1 = torch.tensor([-1.0, 0.0], device=DEVICE, dtype=dtype).view(1, 2).repeat(N_each, 1)

    x2 = (_R + _GEOM_EPS) + torch.rand(N_each, 1, device=DEVICE, dtype=dtype) * (_L - _R - 2.0 * _GEOM_EPS)
    y2 = torch.zeros_like(x2)
    n2 = torch.tensor([0.0, -1.0], device=DEVICE, dtype=dtype).view(1, 2).repeat(N_each, 1)

    y3 = torch.rand(N_each, 1, device=DEVICE, dtype=dtype) * (_L - _R - _GEOM_EPS)
    x3 = torch.full_like(y3, _L)
    n3 = torch.tensor([1.0, 0.0], device=DEVICE, dtype=dtype).view(1, 2).repeat(N_each, 1)

    x4 = torch.rand(N_each, 1, device=DEVICE, dtype=dtype) * (_L - _R - _GEOM_EPS)
    y4 = torch.full_like(x4, _L)
    n4 = torch.tensor([0.0, 1.0], device=DEVICE, dtype=dtype).view(1, 2).repeat(N_each, 1)

    x = torch.cat([x1, x2, x3, x4], dim=0)
    y = torch.cat([y1, y2, y3, y4], dim=0)
    n = torch.cat([n1, n2, n3, n4], dim=0)

    t = torch.rand(x.shape[0], 1, device=DEVICE, dtype=dtype) * _T

    xt = (x / _L).requires_grad_(True)
    yt = (y / _L).requires_grad_(True)
    tt = (t / _T).requires_grad_(True)
    nx = n[:, 0:1]
    ny = n[:, 1:2]
    return xt, yt, tt, nx, ny


# -----------------------------------------------------------------------------
# RAR (Residual-based Adaptive Refinement)
# -----------------------------------------------------------------------------
# Every few iterations, we sample a pool of candidate points and keep those with the largest PDE residual
# and/or largest saturation gradient magnitude. This tends to focus points near the moving front.
@torch.no_grad()
def _pool_points(N_pool):
    x, y = _rejection_sample_xy(N_pool)
    t = torch.rand(N_pool, 1, device=DEVICE, dtype=torch.get_default_dtype()) * _T
    return x / _L, y / _L, t / _T


def select_rar_points(model, eps,
                      N_pool=60000,
                      N_candidate=16000,
                      N_sel_r=12000,
                      N_sel_g=4000):
    xt_pool, yt_pool, tt_pool = _pool_points(N_pool)
    N_pool_eff = xt_pool.shape[0]
    N_candidate = min(N_candidate, N_pool_eff)
    idx_cand = torch.randperm(N_pool_eff, device=DEVICE)[:N_candidate]

    xt = xt_pool[idx_cand].clone().detach().requires_grad_(True)
    yt = yt_pool[idx_cand].clone().detach().requires_grad_(True)
    tt = tt_pool[idx_cand].clone().detach().requires_grad_(True)

    r_w, r_c, _, S, _, _, _ = pde_residual(model, xt, yt, tt, eps=eps)

    score_r = (r_w.abs() + r_c.abs()).detach().squeeze()
    dSdx = grad(S, xt)
    dSdy = grad(S, yt)
    score_g = torch.sqrt(dSdx ** 2 + dSdy ** 2).detach().squeeze()

    k_r = min(N_sel_r, score_r.numel())
    k_g = min(N_sel_g, score_g.numel())

    idx_r = torch.topk(score_r, k=k_r, largest=True).indices
    idx_g = torch.topk(score_g, k=k_g, largest=True).indices

    idx = torch.unique(torch.cat([idx_r, idx_g], dim=0))

    x_sel = xt[idx].clone().detach().requires_grad_(True)
    y_sel = yt[idx].clone().detach().requires_grad_(True)
    t_sel = tt[idx].clone().detach().requires_grad_(True)

    del xt_pool, yt_pool, tt_pool, xt, yt, tt, r_w, r_c, S, dSdx, dSdy
    return x_sel, y_sel, t_sel


# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------
# The loss is a weighted sum of:
#   - PDE residual (with a warm-up + ramp schedule)
#   - supervised pressure/saturation data loss
#   - injection boundary (flux + saturation)
#   - production boundary (pressure)
#   - outer boundary no-flow
#   - initial condition
# Practical tip: start with a low PDE weight (warm-up) so the network first learns a reasonable scale from data/BC.
def train():
    model = TwoNetPINN(width=128, use_fourier=True).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=2e-4)

    w_data_p = 5.0
    w_data_s = 20.0
    w_inj_flux = 5.0
    w_inj_sat = 20.0
    w_out_p = 5.0
    w_noflow = 2.0
    w_ic = 20.0

    beta_max = 30.0
    warm_steps = 5000.0

    RAR_EVERY = 50
    N_BASE = 20000
    N_POOL = 60000
    N_CAND = 16000
    N_SEL_R = 12000
    N_SEL_G = 4000

    EPS0 = 1e-3
    EPS1 = 1e-6
    DECAY_STEPS = 8000.0

    def eps_schedule(it):
        if it < DECAY_STEPS:
            s = it / DECAY_STEPS
            return EPS0 * (EPS1 / EPS0) ** s
        return EPS1

    WARMUP_STEPS = 2000
    RAMP_STEPS = 4000
    RAR_START = 3500

    def w_pde_schedule(it):
        if it <= WARMUP_STEPS:
            return 0.0
        if it <= WARMUP_STEPS + RAMP_STEPS:
            s = (it - WARMUP_STEPS) / RAMP_STEPS
            return float(s)
        return 1.0

    FRONT_A = 10.0
    FRONT_SIGMA = 0.15

    # --------------------------
    # Adam training
    # --------------------------
    for it in range(1, 200001):
        opt.zero_grad()

        beta = min(beta_max, 1.0 + (it / warm_steps) * (beta_max - 1.0))
        model.set_beta(beta)

        eps = eps_schedule(it)
        w_pde = w_pde_schedule(it)

        if w_pde > 0.0:
            use_rar = (it >= RAR_START) and (it % RAR_EVERY == 0)
            if use_rar:
                xf, yf, tf = select_rar_points(
                    model, eps=eps,
                    N_pool=N_POOL, N_candidate=N_CAND,
                    N_sel_r=N_SEL_R, N_sel_g=N_SEL_G,
                )
                rar_flag = "RAR"
            else:
                xf, yf, tf = sample_interior(N_BASE)
                rar_flag = "rnd"

            r_w, r_c, *_ = pde_residual(model, xf, yf, tf, eps=eps)
            loss_pde = (r_w ** 2).mean() + (r_c ** 2).mean()
        else:
            loss_pde = torch.zeros((), device=DEVICE)
            rar_flag = "off"

        xd, yd, td, ptd_true, Sd_true = sample_data_batch(8192)
        ptd_pred, Sd_pred = model(xd, yd, td)
        loss_data_p = ((ptd_pred - ptd_true) ** 2).mean()

        w_front = 1.0 + FRONT_A * torch.exp(-((Sd_true - 0.5) ** 2) / (2.0 * FRONT_SIGMA ** 2))
        w_front = w_front / (w_front.mean().detach() + 1e-12)
        loss_data_s = (w_front * (Sd_pred - Sd_true) ** 2).mean()

        xi, yi, ti, nxi, nyi = sample_injection_arc(1500)
        _, _, _, S_i, vtx_i, vty_i, _ = pde_residual(model, xi, yi, ti, eps=eps)
        vn_in = vtx_i * nxi + vty_i * nyi
        loss_inj_flux = ((vn_in - 1.0) ** 2).mean()
        loss_inj_sat = ((S_i - (1.0 - Sw_irr)) ** 2).mean()

        xo, yo, to = sample_outlet_arc(1500)
        p_t_o, _ = model(xo, yo, to)
        loss_out_p = ((p_t_o - p_out_tilde) ** 2).mean()

        xb, yb, tb, nxb, nyb = sample_outer_boundary(600)
        _, _, _, _, vtx_b, vty_b, _ = pde_residual(model, xb, yb, tb, eps=eps)
        vn_b = vtx_b * nxb + vty_b * nyb
        loss_noflow = (vn_b ** 2).mean()

        x0, y0, t0 = sample_initial(4000)
        p_t_0, S_0 = model(x0, y0, t0)
        loss_ic = ((p_t_0 - 0.0) ** 2).mean() + ((S_0 - Snr) ** 2).mean()

        loss = (
            w_pde * loss_pde
            + w_data_p * loss_data_p
            + w_data_s * loss_data_s
            + w_inj_flux * loss_inj_flux
            + w_inj_sat * loss_inj_sat
            + w_out_p * loss_out_p
            + w_noflow * loss_noflow
            + w_ic * loss_ic
        )

        loss.backward()
        opt.step()

        if it % 200 == 0:
            print(
                f"it={it:6d} [{rar_flag}] wPDE={w_pde:4.2f} beta={beta:5.2f} eps={eps:.2e} loss={loss.item():.3e} | "
                f"pde={loss_pde.item():.2e} dataP={loss_data_p.item():.2e} dataS={loss_data_s.item():.2e} | "
                f"injF={loss_inj_flux.item():.2e} injS={loss_inj_sat.item():.2e} outP={loss_out_p.item():.2e} "
                f"noflow={loss_noflow.item():.2e} ic={loss_ic.item():.2e} | A_time={A_time:.3f}"
            )

    # -------------------------------------------------------------------------
    # L-BFGS fine-tuning (新增部分；不改你的loss组成/物理逻辑)
    # -------------------------------------------------------------------------
    # 说明：
    # - L-BFGS要求 closure 的目标函数尽量“确定”（每次一样），否则容易震荡或失败。
    # - 因此这里固定一批点（interior/data/bc/ic），在L-BFGS过程中不再重采样。
    # - eps 和 beta 固定为训练末值（eps=EPS1, beta=beta_max），w_pde=1。
    USE_LBFGS = True
    LBFGS_MAX_ITER = 1600          # 可以调大，比如 2000
    LBFGS_HISTORY_SIZE = 50
    LBFGS_LINE_SEARCH = "strong_wolfe"

    if USE_LBFGS:
        print("\n=== Starting L-BFGS fine-tuning (fixed batches) ===")

        # 固定超参为末值（逻辑不变，只是把 schedule 在精修阶段固定下来）
        beta = beta_max
        model.set_beta(beta)
        eps = EPS1
        w_pde = 1.0

        # 固定采样点（一次性采样，L-BFGS期间不变）
        # PDE点：用随机 interior（不使用RAR，保证closure稳定）
        xf, yf, tf = sample_interior(N_BASE)

        # data监督点：固定一批
        xd, yd, td, ptd_true, Sd_true = sample_data_batch(8192)

        # injection arc
        xi, yi, ti, nxi, nyi = sample_injection_arc(1500)

        # outlet arc
        xo, yo, to = sample_outlet_arc(1500)

        # outer boundary
        xb, yb, tb, nxb, nyb = sample_outer_boundary(600)

        # initial condition
        x0, y0, t0 = sample_initial(4000)

        # 预先计算 dataS 的 front 权重（固定监督批次）
        with torch.no_grad():
            w_front = 1.0 + FRONT_A * torch.exp(-((Sd_true - 0.5) ** 2) / (2.0 * FRONT_SIGMA ** 2))
            w_front = w_front / (w_front.mean().detach() + 1e-12)

        lbfgs = torch.optim.LBFGS(
            model.parameters(),
            lr=1.0,
            max_iter=LBFGS_MAX_ITER,
            history_size=LBFGS_HISTORY_SIZE,
            line_search_fn=LBFGS_LINE_SEARCH,
        )

        # 计数器：L-BFGS内部会多次调用closure
        closure_calls = {"n": 0}

        def closure():
            closure_calls["n"] += 1
            lbfgs.zero_grad(set_to_none=True)

            # PDE
            r_w, r_c, *_ = pde_residual(model, xf, yf, tf, eps=eps)
            loss_pde = (r_w ** 2).mean() + (r_c ** 2).mean()

            # data
            ptd_pred, Sd_pred = model(xd, yd, td)
            loss_data_p = ((ptd_pred - ptd_true) ** 2).mean()
            loss_data_s = (w_front * (Sd_pred - Sd_true) ** 2).mean()

            # injection
            _, _, _, S_i, vtx_i, vty_i, _ = pde_residual(model, xi, yi, ti, eps=eps)
            vn_in = vtx_i * nxi + vty_i * nyi
            loss_inj_flux = ((vn_in - 1.0) ** 2).mean()
            loss_inj_sat = ((S_i - (1.0 - Sw_irr)) ** 2).mean()

            # outlet pressure
            p_t_o, _ = model(xo, yo, to)
            loss_out_p = ((p_t_o - p_out_tilde) ** 2).mean()

            # no-flow on outer boundary
            _, _, _, _, vtx_b, vty_b, _ = pde_residual(model, xb, yb, tb, eps=eps)
            vn_b = vtx_b * nxb + vty_b * nyb
            loss_noflow = (vn_b ** 2).mean()

            # initial condition
            p_t_0, S_0 = model(x0, y0, t0)
            loss_ic = ((p_t_0 - 0.0) ** 2).mean() + ((S_0 - Snr) ** 2).mean()

            # total
            loss = (
                w_pde * loss_pde
                + w_data_p * loss_data_p
                + w_data_s * loss_data_s
                + w_inj_flux * loss_inj_flux
                + w_inj_sat * loss_inj_sat
                + w_out_p * loss_out_p
                + w_noflow * loss_noflow
                + w_ic * loss_ic
            )

            loss.backward()

            # 可选：打印 closure 过程（不要太频繁）
            if closure_calls["n"] % 50 == 0:
                print(
                    f"[LBFGS] call={closure_calls['n']:5d} loss={loss.item():.3e} | "
                    f"pde={loss_pde.item():.2e} dataP={loss_data_p.item():.2e} dataS={loss_data_s.item():.2e} | "
                    f"injF={loss_inj_flux.item():.2e} injS={loss_inj_sat.item():.2e} outP={loss_out_p.item():.2e} "
                    f"noflow={loss_noflow.item():.2e} ic={loss_ic.item():.2e}"
                )
            return loss

        lbfgs.step(closure)

        # L-BFGS结束后，再评估一次同批次loss
        with torch.no_grad():
            r_w, r_c, *_ = pde_residual(model, xf, yf, tf, eps=eps)
            loss_pde = (r_w ** 2).mean() + (r_c ** 2).mean()

            ptd_pred, Sd_pred = model(xd, yd, td)
            loss_data_p = ((ptd_pred - ptd_true) ** 2).mean()
            loss_data_s = (w_front * (Sd_pred - Sd_true) ** 2).mean()

            _, _, _, S_i, vtx_i, vty_i, _ = pde_residual(model, xi, yi, ti, eps=eps)
            vn_in = vtx_i * nxi + vty_i * nyi
            loss_inj_flux = ((vn_in - 1.0) ** 2).mean()
            loss_inj_sat = ((S_i - (1.0 - Sw_irr)) ** 2).mean()

            p_t_o, _ = model(xo, yo, to)
            loss_out_p = ((p_t_o - p_out_tilde) ** 2).mean()

            _, _, _, _, vtx_b, vty_b, _ = pde_residual(model, xb, yb, tb, eps=eps)
            vn_b = vtx_b * nxb + vty_b * nyb
            loss_noflow = (vn_b ** 2).mean()

            p_t_0, S_0 = model(x0, y0, t0)
            loss_ic = ((p_t_0 - 0.0) ** 2).mean() + ((S_0 - Snr) ** 2).mean()

            loss = (
                w_pde * loss_pde
                + w_data_p * loss_data_p
                + w_data_s * loss_data_s
                + w_inj_flux * loss_inj_flux
                + w_inj_sat * loss_inj_sat
                + w_out_p * loss_out_p
                + w_noflow * loss_noflow
                + w_ic * loss_ic
            )
            print(
                f"=== L-BFGS done === loss={loss.item():.3e} | "
                f"pde={loss_pde.item():.2e} dataP={loss_data_p.item():.2e} dataS={loss_data_s.item():.2e} | "
                f"injF={loss_inj_flux.item():.2e} injS={loss_inj_sat.item():.2e} outP={loss_out_p.item():.2e} "
                f"noflow={loss_noflow.item():.2e} ic={loss_ic.item():.2e}"
            )

    out_name = "pinn_co2_brine_TwoNet_beta_RAR_eps_frontw_bigdata.pt"
    torch.save(model.state_dict(), out_name)
    print(f"Saved model to {out_name}")
    return model


if __name__ == "__main__":
    print(f"P_ref={P_ref:.3e} Pa, p_out_tilde={p_out_tilde:.3e}, A_time={A_time:.3f}")
    train()
