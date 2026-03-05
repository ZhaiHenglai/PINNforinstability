"""
GPU-adaptive experiment script for 2D brine-CO2 two-phase flow PINN.

This file merges:
1) the experiment-switch skeleton for controlled studies:
   B0 / A1 / A3 / A4 / A5 / A6
2) the GPU-adaptive runtime sizing idea from the uploaded multi-GPU version.

Experiment definitions
----------------------
B0 : minimal working baseline
A1 : B0 + front-weighted saturation loss
A3 : A1 + full RAR (residual + |grad S|)
A4 : A3 + beta curriculum
A5 : A4 + diffusion decay
A6 : A5 + TwoNet

How to run
----------
1) Put your binary dataset next to this script as `tables_cache_tensor.pt` or `tables_cache.pt`,
   or set `PINN_DATA_PT=/abs/path/to/file.pt`
2) Set EXP_NAME below, and optionally GPU_ID / SEED
3) Run:
       python pinn_experiment_gpu_adaptive.py

Notes
-----
- float64 is kept for PINN stability with higher-order derivatives.
- Runtime profile only changes width / batch sizes / RAR sizes / L-BFGS usage.
- The physics and loss structure remain aligned with the original script.
"""

# -----------------------------------------------------------------------------
# Imports and numeric setup
# -----------------------------------------------------------------------------
import os
import glob
import math
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# -----------------------------------------------------------------------------
# User controls
# -----------------------------------------------------------------------------
EXP_NAME = "A5"   # choose from: B0, A1, A3, A4, A5, A6
SEED = 0
GPU_ID = os.environ.get("PINN_GPU_ID", "0")

# dataset
DATA_PT_CANDIDATES = [
    "/home/henglai_pc/pythonTesst/PINN3/tables_cache_tensor.pt",
    "tables_cache.pt",
]
DATA_GLOB = "./tables/*.csv"
EOS_FIT_SAMPLE = 2_000_000

# training length
MAX_ADAM_ITERS = 20000

# experiment switches; overwritten by configure_experiment()
USE_FRONT_WEIGHT = False
USE_RAR = False
USE_BETA_CURRICULUM = False
USE_DIFFUSION_DECAY = False
USE_TWONET = False
USE_FOURIER = False
USE_LBFGS = False

EPS_CONST = 1e-5
BETA_CONST = 1.0

# set GPU visibility before device selection
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(GPU_ID))


# -----------------------------------------------------------------------------
# Helpers: seeds and runtime config
# -----------------------------------------------------------------------------
def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _detect_runtime_config():
    """
    Build a small runtime profile that adapts the heavy training settings to the available device.

    Important choice:
    - Keep float64 everywhere for stability of higher-order derivatives.
    - Only adapt batch sizes / RAR sizes / model width / L-BFGS usage by GPU type.
    """
    cpu_threads = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))

    base = {
        "dtype": torch.float64,
        "cpu_threads": max(1, cpu_threads),
    }

    if not torch.cuda.is_available():
        base.update({
            "profile": "cpu",
            "device": torch.device("cpu"),
            "gpu_name": "CPU",
            "gpu_mem_gb": 0.0,
            "gpu_cc": "n/a",
            "gpu_sms": 0,
            "width": 128,
            "N_BASE": 12000,
            "N_POOL": 36000,
            "N_CAND": 10000,
            "N_SEL_R": 7500,
            "N_SEL_G": 2500,
            "DATA_BATCH": 4096,
            "USE_LBFGS": False,
            "LBFGS_MAX_ITER": 0,
            "LBFGS_HISTORY_SIZE": 50,
            "INJ_ARC_N": 1200,
            "OUTLET_ARC_N": 1200,
            "OUTER_BC_N": 500,
            "INIT_N": 3000,
        })
        return base

    props = torch.cuda.get_device_properties(0)
    gpu_name = props.name
    mem_gb = props.total_memory / (1024 ** 3)
    cc = f"{props.major}.{props.minor}"
    gpu_name_l = gpu_name.lower()

    if "v100" in gpu_name_l:
        base.update({
            "profile": "v100",
            "device": torch.device("cuda"),
            "gpu_name": gpu_name,
            "gpu_mem_gb": mem_gb,
            "gpu_cc": cc,
            "gpu_sms": props.multi_processor_count,
            # balanced V100 profile
            "width": 160,
            "N_BASE": 45000,
            "N_POOL": 120000,
            "N_CAND": 32000,
            "N_SEL_R": 24000,
            "N_SEL_G": 8000,
            "DATA_BATCH": 24576,
            "USE_LBFGS": True,
            "LBFGS_MAX_ITER": 1400,
            "LBFGS_HISTORY_SIZE": 40,
            "INJ_ARC_N": 2600,
            "OUTLET_ARC_N": 2600,
            "OUTER_BC_N": 1100,
            "INIT_N": 9000,
        })
        return base

    if "2080 ti" in gpu_name_l or ("rtx" in gpu_name_l and mem_gb <= 12.5):
        base.update({
            "profile": "2080ti",
            "device": torch.device("cuda"),
            "gpu_name": gpu_name,
            "gpu_mem_gb": mem_gb,
            "gpu_cc": cc,
            "gpu_sms": props.multi_processor_count,
            "width": 96,
            "N_BASE": 8000,
            "N_POOL": 24000,
            "N_CAND": 6000,
            "N_SEL_R": 4500,
            "N_SEL_G": 1500,
            "DATA_BATCH": 2048,
            "USE_LBFGS": False,
            "LBFGS_MAX_ITER": 0,
            "LBFGS_HISTORY_SIZE": 30,
            "INJ_ARC_N": 800,
            "OUTLET_ARC_N": 800,
            "OUTER_BC_N": 300,
            "INIT_N": 2000,
        })
        return base

    # generic fallbacks by memory
    if mem_gb >= 30.0:
        base.update({
            "profile": "gpu_32gb_like",
            "device": torch.device("cuda"),
            "gpu_name": gpu_name,
            "gpu_mem_gb": mem_gb,
            "gpu_cc": cc,
            "gpu_sms": props.multi_processor_count,
            "width": 128,
            "N_BASE": 18000,
            "N_POOL": 54000,
            "N_CAND": 14000,
            "N_SEL_R": 10500,
            "N_SEL_G": 3500,
            "DATA_BATCH": 6144,
            "USE_LBFGS": True,
            "LBFGS_MAX_ITER": 1200,
            "LBFGS_HISTORY_SIZE": 50,
            "INJ_ARC_N": 1400,
            "OUTLET_ARC_N": 1400,
            "OUTER_BC_N": 600,
            "INIT_N": 3500,
        })
        return base

    if mem_gb >= 20.0:
        base.update({
            "profile": "gpu_24gb_like",
            "device": torch.device("cuda"),
            "gpu_name": gpu_name,
            "gpu_mem_gb": mem_gb,
            "gpu_cc": cc,
            "gpu_sms": props.multi_processor_count,
            "width": 128,
            "N_BASE": 14000,
            "N_POOL": 42000,
            "N_CAND": 11000,
            "N_SEL_R": 8250,
            "N_SEL_G": 2750,
            "DATA_BATCH": 4096,
            "USE_LBFGS": True,
            "LBFGS_MAX_ITER": 900,
            "LBFGS_HISTORY_SIZE": 50,
            "INJ_ARC_N": 1200,
            "OUTLET_ARC_N": 1200,
            "OUTER_BC_N": 500,
            "INIT_N": 3000,
        })
        return base

    if mem_gb >= 10.0:
        base.update({
            "profile": "gpu_12gb_like",
            "device": torch.device("cuda"),
            "gpu_name": gpu_name,
            "gpu_mem_gb": mem_gb,
            "gpu_cc": cc,
            "gpu_sms": props.multi_processor_count,
            "width": 96,
            "N_BASE": 9000,
            "N_POOL": 27000,
            "N_CAND": 7000,
            "N_SEL_R": 5250,
            "N_SEL_G": 1750,
            "DATA_BATCH": 2048,
            "USE_LBFGS": False,
            "LBFGS_MAX_ITER": 0,
            "LBFGS_HISTORY_SIZE": 30,
            "INJ_ARC_N": 900,
            "OUTLET_ARC_N": 900,
            "OUTER_BC_N": 350,
            "INIT_N": 2200,
        })
        return base

    base.update({
        "profile": "gpu_small",
        "device": torch.device("cuda"),
        "gpu_name": gpu_name,
        "gpu_mem_gb": mem_gb,
        "gpu_cc": cc,
        "gpu_sms": props.multi_processor_count,
        "width": 64,
        "N_BASE": 5000,
        "N_POOL": 15000,
        "N_CAND": 4000,
        "N_SEL_R": 3000,
        "N_SEL_G": 1000,
        "DATA_BATCH": 1024,
        "USE_LBFGS": False,
        "LBFGS_MAX_ITER": 0,
        "LBFGS_HISTORY_SIZE": 20,
        "INJ_ARC_N": 600,
        "OUTLET_ARC_N": 600,
        "OUTER_BC_N": 250,
        "INIT_N": 1500,
    })
    return base


def configure_experiment(exp_name: str):
    """
    Configure experiment switches.

    B0 : single net, uniform sat loss, no RAR, fixed beta=1, constant eps
    A1 : B0 + front-weighted saturation loss
    A3 : A1 + full RAR
    A4 : A3 + beta curriculum
    A5 : A4 + diffusion decay
    A6 : A5 + TwoNet
    """
    global USE_FRONT_WEIGHT, USE_RAR, USE_BETA_CURRICULUM
    global USE_DIFFUSION_DECAY, USE_TWONET, USE_FOURIER, USE_LBFGS
    global EPS_CONST, BETA_CONST

    # reset defaults
    USE_FRONT_WEIGHT = False
    USE_RAR = False
    USE_BETA_CURRICULUM = False
    USE_DIFFUSION_DECAY = False
    USE_TWONET = False
    USE_FOURIER = False
    USE_LBFGS = False

    EPS_CONST = 1e-5
    BETA_CONST = 1.0

    if exp_name == "B0":
        pass
    elif exp_name == "A1":
        USE_FRONT_WEIGHT = True
    elif exp_name == "A3":
        USE_FRONT_WEIGHT = True
        USE_RAR = True
    elif exp_name == "A4":
        USE_FRONT_WEIGHT = True
        USE_RAR = True
        USE_BETA_CURRICULUM = True
    elif exp_name == "A5":
        USE_FRONT_WEIGHT = True
        USE_RAR = True
        USE_BETA_CURRICULUM = True
        USE_DIFFUSION_DECAY = True
    elif exp_name == "A6":
        USE_FRONT_WEIGHT = True
        USE_RAR = True
        USE_BETA_CURRICULUM = True
        USE_DIFFUSION_DECAY = True
        USE_TWONET = True
    else:
        raise ValueError(f"Unknown EXP_NAME: {exp_name}")


RUNTIME = _detect_runtime_config()
DEVICE = RUNTIME["device"]
DTYPE = RUNTIME["dtype"]

torch.set_default_dtype(DTYPE)
torch.set_num_threads(RUNTIME["cpu_threads"])
try:
    torch.set_num_interop_threads(1)
except Exception:
    pass


def print_runtime_report():
    print(
        f"[Runtime] EXP={EXP_NAME} | profile={RUNTIME['profile']} | device={DEVICE} | dtype={DTYPE} | "
        f"cpu_threads={RUNTIME['cpu_threads']} | visible_cuda={os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}"
    )

    if DEVICE.type == "cuda":
        print(
            f"[GPU] name={RUNTIME.get('gpu_name', 'unknown')} | total_mem={RUNTIME.get('gpu_mem_gb', 0.0):.1f} GB | "
            f"compute_capability={RUNTIME.get('gpu_cc', 'unknown')} | SMs={RUNTIME.get('gpu_sms', 'unknown')} | "
            f"cuda_version={torch.version.cuda}"
        )
    else:
        print("[GPU] CUDA not available; running on CPU.")

    print(
        f"[TrainConfig] width={RUNTIME['width']} | DATA_BATCH={RUNTIME['DATA_BATCH']} | "
        f"N_BASE={RUNTIME['N_BASE']} | N_POOL={RUNTIME['N_POOL']} | N_CAND={RUNTIME['N_CAND']} | "
        f"N_SEL_R={RUNTIME['N_SEL_R']} | N_SEL_G={RUNTIME['N_SEL_G']} | "
        f"USE_LBFGS_RUNTIME={RUNTIME['USE_LBFGS']} | LBFGS_MAX_ITER={RUNTIME['LBFGS_MAX_ITER']}"
    )
    print(
        f"[ExpSwitches] front_weight={USE_FRONT_WEIGHT} | RAR={USE_RAR} | beta_curr={USE_BETA_CURRICULUM} | "
        f"diff_decay={USE_DIFFUSION_DECAY} | twonet={USE_TWONET} | fourier={USE_FOURIER} | lbfgs={USE_LBFGS}"
    )


def get_gpu_memory_report():
    if DEVICE.type != "cuda":
        return "gpu_mem=n/a"
    alloc = torch.cuda.memory_allocated() / (1024 ** 3)
    reserv = torch.cuda.memory_reserved() / (1024 ** 3)
    max_alloc = torch.cuda.max_memory_allocated() / (1024 ** 3)
    max_reserv = torch.cuda.max_memory_reserved() / (1024 ** 3)
    return (
        f"gpu_mem alloc={alloc:.2f}GB reserv={reserv:.2f}GB "
        f"max_alloc={max_alloc:.2f}GB max_reserv={max_reserv:.2f}GB"
    )


# -----------------------------------------------------------------------------
# Physical parameters and nondimensionalization
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# Dataset loading utilities
# -----------------------------------------------------------------------------
REQUIRED_COLS = [
    "X", "Y", "phase1::Pressure", "phase1::Time",
    "phase2::PhaseVolumeFraction", "phase2::Density",
    "phase1::Density", "phase1::Viscosity_0", "phase2::Viscosity_0",
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
def _sample_pair_numpy(p, rho, n_sample: int, seed: int = 0):
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
    dtmp = pd.DataFrame({"p": p_pa, "rho": rho})
    g = dtmp.groupby("p", as_index=False)["rho"].mean().sort_values("p")
    p_u = g["p"].to_numpy(dtype=np.float64)
    rho_u = g["rho"].to_numpy(dtype=np.float64)
    return p_u, rho_u


def fit_rho_cheb(p_pa: np.ndarray, rho: np.ndarray,
                 deg_list=(10, 12, 14, 16, 18, 20),
                 n_val=20000, seed=0):
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


p_s, rho_c_s = _sample_pair_numpy(arrays["p"], arrays["rho_c"], EOS_FIT_SAMPLE, seed=SEED)
cheb_coeffs_np, rho_c_ref, eos_pmin, eos_pmax = fit_rho_cheb(p_s, rho_c_s)


class ChebRhoCO2(nn.Module):
    def __init__(self, coeffs_np, rho_ref, pmin, pmax, eps=1e-12):
        super().__init__()
        self.register_buffer("c", torch.tensor(coeffs_np, dtype=DTYPE))
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
        p_pa = p_pa.to(dtype=DTYPE)
        z = (p_pa - self.pmid) / self.prng
        z = torch.clamp(z, -1.0 + self.eps, 1.0 - self.eps)
        rho_tilde = self._chebval_clenshaw(z)
        return self.rho_ref * rho_tilde


rho_co2_model = ChebRhoCO2(cheb_coeffs_np, rho_c_ref, eos_pmin, eos_pmax).to(DEVICE)


# -----------------------------------------------------------------------------
# Relative permeability model
# -----------------------------------------------------------------------------
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
# Neural networks
# -----------------------------------------------------------------------------
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


class SingleNetPINN(nn.Module):
    def __init__(self, width=128, use_fourier=False):
        super().__init__()
        self.use_fourier = use_fourier
        if use_fourier:
            self.ff = FourierFeatures(3, 32, 6.0)
            in_dim = 64
        else:
            in_dim = 3

        self.backbone = MLP(in_dim, 2, width=width, depth=8)
        self.beta = 1.0

    def set_beta(self, beta: float):
        self.beta = float(beta)

    def forward(self, x_t, y_t, t_t):
        x_in = 2.0 * x_t - 1.0
        y_in = 2.0 * y_t - 1.0
        t_in = 2.0 * t_t - 1.0
        X = torch.cat([x_in, y_in, t_in], dim=1)

        Z = self.ff(X) if self.use_fourier else X
        out = self.backbone(Z)
        p_tilde = out[:, 0:1]
        s_logit = out[:, 1:2]

        S_hat = torch.sigmoid(self.beta * s_logit)
        Sco2 = Snr + (1.0 - Sw_irr - Snr) * S_hat
        return p_tilde, Sco2


class TwoNetPINN(nn.Module):
    def __init__(self, width=128, use_fourier=False):
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
def grad(u, x, create_graph=True, retain_graph=True):
    return torch.autograd.grad(
        u, x,
        grad_outputs=torch.ones_like(u),
        create_graph=create_graph,
        retain_graph=retain_graph,
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
# PDE residual
# -----------------------------------------------------------------------------
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
# Supervised data batching
# -----------------------------------------------------------------------------
def _as_cpu_float32_col(v):
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

p_all = _as_cpu_float32_col(arrays["p"])
Sco2_raw_all = _as_cpu_float32_col(arrays["Sco2"])

N_data = int(x_all.shape[0])


def sample_data_batch(batch_size=None):
    if batch_size is None:
        batch_size = RUNTIME["DATA_BATCH"]

    idx = torch.randint(0, N_data, (batch_size,), device="cpu", dtype=torch.int64)

    xd = x_all.index_select(0, idx).to(DEVICE, dtype=DTYPE)
    yd = y_all.index_select(0, idx).to(DEVICE, dtype=DTYPE)
    td = t_all.index_select(0, idx).to(DEVICE, dtype=DTYPE)

    p_true = p_all.index_select(0, idx).to(DEVICE, dtype=DTYPE)
    ptd = (p_true - p0) / P_ref

    Sco2_raw = Sco2_raw_all.index_select(0, idx).to(DEVICE, dtype=DTYPE)
    Sco2_raw = torch.clamp(Sco2_raw, 0.0, 1.0)
    Sd = Snr + (1.0 - Sw_irr - Snr) * Sco2_raw

    return xd, yd, td, ptd, Sd


# -----------------------------------------------------------------------------
# Geometry helpers
# -----------------------------------------------------------------------------
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
# Collocation / BC / IC sampling
# -----------------------------------------------------------------------------
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
# RAR
# -----------------------------------------------------------------------------
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
    dSdx = grad(S, xt, create_graph=False, retain_graph=True)
    dSdy = grad(S, yt, create_graph=False, retain_graph=True)
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
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return x_sel, y_sel, t_sel


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
def train():
    print_runtime_report()
    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    width = RUNTIME["width"]
    if USE_TWONET:
        model = TwoNetPINN(width=width, use_fourier=USE_FOURIER).to(DEVICE)
    else:
        model = SingleNetPINN(width=width, use_fourier=USE_FOURIER).to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=2e-4)

    # loss weights
    w_data_p = 5.0
    w_data_s = 20.0
    w_inj_flux = 5.0
    w_inj_sat = 20.0
    w_out_p = 5.0
    w_noflow = 2.0
    w_ic = 20.0

    # runtime-dependent sizes
    beta_max = 30.0
    warm_steps = 5000.0

    RAR_EVERY = 50
    N_BASE = RUNTIME["N_BASE"]
    N_POOL = RUNTIME["N_POOL"]
    N_CAND = RUNTIME["N_CAND"]
    N_SEL_R = RUNTIME["N_SEL_R"]
    N_SEL_G = RUNTIME["N_SEL_G"]
    DATA_BATCH = RUNTIME["DATA_BATCH"]
    INJ_ARC_N = RUNTIME["INJ_ARC_N"]
    OUTLET_ARC_N = RUNTIME["OUTLET_ARC_N"]
    OUTER_BC_N = RUNTIME["OUTER_BC_N"]
    INIT_N = RUNTIME["INIT_N"]

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

    for it in range(1, MAX_ADAM_ITERS + 1):
        opt.zero_grad(set_to_none=True)

        # beta
        if USE_BETA_CURRICULUM:
            beta = min(beta_max, 1.0 + (it / warm_steps) * (beta_max - 1.0))
        else:
            beta = BETA_CONST
        model.set_beta(beta)

        # eps
        if USE_DIFFUSION_DECAY:
            eps = eps_schedule(it)
        else:
            eps = EPS_CONST

        w_pde = w_pde_schedule(it)

        # PDE points
        if w_pde > 0.0:
            use_rar_now = USE_RAR and (it >= RAR_START) and (it % RAR_EVERY == 0)
            if use_rar_now:
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

        # supervised data
        xd, yd, td, ptd_true, Sd_true = sample_data_batch(DATA_BATCH)
        ptd_pred, Sd_pred = model(xd, yd, td)
        loss_data_p = ((ptd_pred - ptd_true) ** 2).mean()

        if USE_FRONT_WEIGHT:
            w_front = 1.0 + FRONT_A * torch.exp(-((Sd_true - 0.5) ** 2) / (2.0 * FRONT_SIGMA ** 2))
            w_front = w_front / (w_front.mean().detach() + 1e-12)
            loss_data_s = (w_front * (Sd_pred - Sd_true) ** 2).mean()
        else:
            loss_data_s = ((Sd_pred - Sd_true) ** 2).mean()

        # injection boundary
        xi, yi, ti, nxi, nyi = sample_injection_arc(INJ_ARC_N)
        _, _, _, S_i, vtx_i, vty_i, _ = pde_residual(model, xi, yi, ti, eps=eps)
        vn_in = vtx_i * nxi + vty_i * nyi
        loss_inj_flux = ((vn_in - 1.0) ** 2).mean()
        loss_inj_sat = ((S_i - (1.0 - Sw_irr)) ** 2).mean()

        # outlet pressure
        xo, yo, to = sample_outlet_arc(OUTLET_ARC_N)
        p_t_o, _ = model(xo, yo, to)
        loss_out_p = ((p_t_o - p_out_tilde) ** 2).mean()

        # outer no-flow boundary
        xb, yb, tb, nxb, nyb = sample_outer_boundary(OUTER_BC_N)
        _, _, _, _, vtx_b, vty_b, _ = pde_residual(model, xb, yb, tb, eps=eps)
        vn_b = vtx_b * nxb + vty_b * nyb
        loss_noflow = (vn_b ** 2).mean()

        # initial condition
        x0, y0, t0 = sample_initial(INIT_N)
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
                f"[{EXP_NAME}] it={it:6d} [{rar_flag}] wPDE={w_pde:4.2f} beta={beta:5.2f} eps={eps:.2e} loss={loss.item():.3e} | "
                f"pde={loss_pde.item():.2e} dataP={loss_data_p.item():.2e} dataS={loss_data_s.item():.2e} | "
                f"injF={loss_inj_flux.item():.2e} injS={loss_inj_sat.item():.2e} outP={loss_out_p.item():.2e} "
                f"noflow={loss_noflow.item():.2e} ic={loss_ic.item():.2e} | {get_gpu_memory_report()}"
            )

        # avoid stale cached memory on some GPUs during long runs
        if DEVICE.type == "cuda" and it % 5000 == 0:
            torch.cuda.empty_cache()

    # optional L-BFGS
    use_lbfgs_now = USE_LBFGS and RUNTIME["USE_LBFGS"]
    if use_lbfgs_now:
        print("\n=== Starting L-BFGS fine-tuning (fixed batches) ===")

        beta = beta_max if USE_BETA_CURRICULUM else BETA_CONST
        model.set_beta(beta)
        eps = EPS1 if USE_DIFFUSION_DECAY else EPS_CONST
        w_pde = 1.0

        xf, yf, tf = sample_interior(N_BASE)
        xd, yd, td, ptd_true, Sd_true = sample_data_batch(DATA_BATCH)
        xi, yi, ti, nxi, nyi = sample_injection_arc(INJ_ARC_N)
        xo, yo, to = sample_outlet_arc(OUTLET_ARC_N)
        xb, yb, tb, nxb, nyb = sample_outer_boundary(OUTER_BC_N)
        x0, y0, t0 = sample_initial(INIT_N)

        if USE_FRONT_WEIGHT:
            with torch.no_grad():
                w_front = 1.0 + FRONT_A * torch.exp(-((Sd_true - 0.5) ** 2) / (2.0 * FRONT_SIGMA ** 2))
                w_front = w_front / (w_front.mean().detach() + 1e-12)

        lbfgs = torch.optim.LBFGS(
            model.parameters(),
            lr=1.0,
            max_iter=RUNTIME["LBFGS_MAX_ITER"],
            history_size=RUNTIME["LBFGS_HISTORY_SIZE"],
            line_search_fn="strong_wolfe",
        )

        closure_calls = {"n": 0}

        def closure():
            closure_calls["n"] += 1
            lbfgs.zero_grad(set_to_none=True)

            r_w, r_c, *_ = pde_residual(model, xf, yf, tf, eps=eps)
            loss_pde = (r_w ** 2).mean() + (r_c ** 2).mean()

            ptd_pred, Sd_pred = model(xd, yd, td)
            loss_data_p = ((ptd_pred - ptd_true) ** 2).mean()
            if USE_FRONT_WEIGHT:
                loss_data_s = (w_front * (Sd_pred - Sd_true) ** 2).mean()
            else:
                loss_data_s = ((Sd_pred - Sd_true) ** 2).mean()

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
            loss.backward()

            if closure_calls["n"] % 50 == 0:
                print(f"[LBFGS-{EXP_NAME}] call={closure_calls['n']:5d} loss={loss.item():.3e}")
            return loss

        lbfgs.step(closure)

    out_name = f"model_{EXP_NAME}_seed{SEED}_{RUNTIME['profile']}.pt"
    torch.save(model.state_dict(), out_name)
    print(f"Saved model to {out_name}")
    return model


if __name__ == "__main__":
    configure_experiment(EXP_NAME)

    # default: allow L-BFGS only if both experiment and runtime permit it
    # for current B0/A1/A3/A4/A5/A6 chain we keep it off unless manually enabled later
    USE_LBFGS = False

    set_seed(SEED)

    print(f"Running EXP_NAME = {EXP_NAME}")
    print(f"GPU_ID request = {GPU_ID}")
    print(f"P_ref={P_ref:.3e} Pa, p_out_tilde={p_out_tilde:.3e}, A_time={A_time:.3f}")
    train()


