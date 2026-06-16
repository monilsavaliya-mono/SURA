"""
FK Weight Test — FULLY PARALLELISED CPU VERSION
================================================
What is parallelised now:
  1. FK precompute        (was already parallel)
  2. Dose map build       (now parallel)
  3. Dose slices          (now parallel)
  4. Strategy runs        (all 8 strategies × 60 trajs now parallel)
  5. optimal_dose()       (vectorised numpy scan — no Python loop)

Expected speedup over original:
  Dose map:      ~20x  (was serial, now 24 cores)
  Strategy runs: ~8x   (was serial, now parallel across strategies+trajs)
  optimal_dose:  ~5x   (vectorised c-scan)
  Overall:       ~3-5x wall-clock improvement

Output: figures/fk_weight/run_NNN/
  page1_NNN.png  — value fn, dose map, dose slices, first 5 trajectories
  page2_NNN.png  — last 3 trajectories + full J bar chart
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from multiprocessing import Pool, cpu_count
import os

# ════════════════════════════════════════════════════════════════════
# PARAMETERS
# ════════════════════════════════════════════════════════════════════
K          = 1000
N_CMP      = 60

k1         = 0.733
k3         = k1 * 0.927
k4         = 0.179
k5_per_div = 7.8e-7
F_max      = 3.0

psi_S_max  = k1 - k4
psi_R_max  = k3 - k4
kmax       = 0.672
psiminS    = psi_S_max - kmax
psiminR    = psiminS
kappa      = 1.1
EC50       = 1.01
micS       = EC50 / ((-psiminS / psi_S_max) ** (1.0 / kappa))
MIC_FACTOR = 8
micR       = MIC_FACTOR * micS
nS_eq      = K * (k1 - k4) / k1

# ── utility weights ───────────────────────────────────────────────
w_T   = 1.0
w_N   = 1.0
w_Rr  = 0.0
w_Rt  = 1000.0
w_C   = 0.1    
# ─────────────────────────────────────────────────────────────────

beta       = 1.0
N_immune   = 10
c_base     = micS*3
T_HORIZON  = 100.0
n_fk       = 200          # FK trajectories per grid node
N_CMP      = 60           # comparison trajectories per strategy

V_interp   = None
nS_Start   = int(nS_eq)
nR_Start   = 1

# ── FK grid ───────────────────────────────────────────────────────
GS = np.arange(0, K + 1,      50)
GR = np.arange(0, K // 5 + 1, 10)

# ── dose scan (vectorised) ────────────────────────────────────────
_c_scan = np.linspace(0.0, 12.0 * micS, 200)

# ════════════════════════════════════════════════════════════════════
# PD CURVE  (scalar versions for SSA, vector versions for opt)
# ════════════════════════════════════════════════════════════════════
def alpha_S(c):
    if c <= 0: return 0.0
    u = (c / micS) ** kappa
    return (psi_S_max - psiminS) * u / (u - psiminS / psi_S_max)

def alpha_R(c):
    if c <= 0: return 0.0
    u = (c / micR) ** kappa
    return (psi_R_max - psiminR) * u / (u - psiminR / psi_R_max)

def alpha_S_vec(c_arr):
    """Vectorised alpha_S over array of concentrations."""
    out  = np.zeros_like(c_arr)
    mask = c_arr > 0
    u    = (c_arr[mask] / micS) ** kappa
    out[mask] = (psi_S_max - psiminS) * u / (u - psiminS / psi_S_max)
    return out

def alpha_R_vec(c_arr):
    """Vectorised alpha_R over array of concentrations."""
    out  = np.zeros_like(c_arr)
    mask = c_arr > 0
    u    = (c_arr[mask] / micR) ** kappa
    out[mask] = (psi_R_max - psiminR) * u / (u - psiminR / psi_R_max)
    return out

# precompute alpha arrays over _c_scan once at module level
_aS_scan = alpha_S_vec(_c_scan)
_aR_scan = alpha_R_vec(_c_scan)

_span = psi_S_max - psiminS
_D    = psiminS / psi_S_max

# ════════════════════════════════════════════════════════════════════
# MUTATION
# ════════════════════════════════════════════════════════════════════
def stress_factor(c):
    if c <= 0: return 1.0
    x = c / micS
    return 1.0 + (F_max - 1.0) * (x / (1.0 + x * x)) / 0.5

def mutation_propensity(nS, nR, c, comp):
    return k5_per_div * stress_factor(c) * k1 * nS * comp

# ════════════════════════════════════════════════════════════════════
# UTILITY
# ════════════════════════════════════════════════════════════════════
def U_running(nS, nR, c=0.0):
    return -w_T - w_N * (nS + nR) / K - w_Rr * nR / K - w_C * (c / micS) ** 2

def U_exit(nS, nR):
    return -w_Rt * nR / K

def is_absorbing(nS, nR):
    return (nS + nR) <= N_immune

# ════════════════════════════════════════════════════════════════════
# FK BASELINE SSA
# ════════════════════════════════════════════════════════════════════
_aS_base = alpha_S(c_base)
_aR_base = alpha_R(c_base)
_k2_base = k4 + _aS_base
_k4_base = k4 + _aR_base

def baseline_utility(nS0, nR0, rng, Tmax=T_HORIZON):
    nS, nR, t = float(nS0), float(nR0), 0.0
    U_acc = 0.0
    while t < Tmax:
        if is_absorbing(nS, nR):
            return U_acc + U_exit(nS, nR)
        comp = max(1.0 - (nS + nR) / K, 0.0)
        a0 = k1 * nS * comp
        a1 = nS * (k4 + _aS_base)
        a2 = k3 * nR * comp
        a3 = nR * (k4 + _aR_base)
        a4 = mutation_propensity(nS, nR, c_base, comp)
        a  = a0 + a1 + a2 + a3 + a4
        if a <= 0:
            return U_acc + U_exit(nS, nR)
        dt     = -np.log(rng.random()) / a
        U_acc += U_running(nS, nR, c_base) * dt
        t     += dt
        u = rng.random() * a
        if   u < a0:                  nS += 1
        elif u < a0 + a1:             nS -= 1
        elif u < a0 + a1 + a2:        nR += 1
        elif u < a0 + a1 + a2 + a3:   nR -= 1
        else:                         nS -= 1; nR += 1
        nS = max(0.0, nS); nR = max(0.0, nR)
    return U_acc + U_exit(nS, nR)

def feynman_kac_V(nS0, nR0, n_traj, rng):
    if is_absorbing(nS0, nR0):
        return beta * U_exit(nS0, nR0)
    Us = np.array([baseline_utility(nS0, nR0, rng) for _ in range(n_traj)])
    x  = beta * Us
    m  = x.max()
    if not np.isfinite(m): return -1e6
    s  = np.mean(np.exp(x - m))
    if s <= 0:              return -1e6
    return m + np.log(s)

# ── worker for FK (top-level so pickle works) ─────────────────────
def _fk_node(args):
    i, j, ns, nr, seed = args
    rng = np.random.RandomState(seed)
    return (i, j, feynman_kac_V(ns, nr, n_fk, rng))

def precompute_V_parallel():
    jobs, seed = [], 12345
    for i, ns in enumerate(GS):
        for j, nr in enumerate(GR):
            jobs.append((i, j, ns, nr, seed))
            seed += 1
    Vgrid = np.zeros((len(GS), len(GR)))
    nw = cpu_count()
    print(f"FK on {len(jobs)} nodes across {nw} cores ...")
    with Pool(nw) as pool:
        done = 0
        for i, j, v in pool.imap_unordered(_fk_node, jobs, chunksize=4):
            Vgrid[i, j] = v
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(jobs)} nodes done")
    return Vgrid

def V_at(nS, nR):
    nS = np.clip(nS, GS[0], GS[-1])
    nR = np.clip(nR, GR[0], GR[-1])
    return float(V_interp((nS, nR)))

# ════════════════════════════════════════════════════════════════════
# OPTIMAL DOSE — fully vectorised (no Python loop over c)
# ════════════════════════════════════════════════════════════════════
def kl_term_vec(k_arr, k0):
    """Vectorised KL cost: k·log(k/k0) - k + k0."""
    out  = np.full_like(k_arr, k0)   # default when k<=0
    mask = k_arr > 0
    out[mask] = (k_arr[mask] * np.log(k_arr[mask] / k0)
                 - k_arr[mask] + k0)
    return out

# precompute k2 and k4r arrays over _c_scan
_k2_scan  = k4 + _aS_scan    # shape (200,)
_k4r_scan = k4 + _aR_scan    # shape (200,)
_kl_k2    = kl_term_vec(_k2_scan,  _k2_base)   # shape (200,)
_kl_k4r   = kl_term_vec(_k4r_scan, _k4_base)   # shape (200,)

def optimal_dose(nS, nR):
    """Vectorised two-channel Hamiltonian maximisation."""
    if is_absorbing(nS, nR) or nS < 1:
        return 0.0
    v_here = V_at(nS, nR)
    g_S = (V_at(nS - 1, nR) - v_here) if nS >= 1 else 0.0
    g_R = (V_at(nS, nR - 1) - v_here) if nR >= 1 else 0.0
    if not (np.isfinite(g_S) and np.isfinite(g_R)):
        return 0.0
    # H over all c values at once — shape (200,)
    H = (nS * (_k2_scan  * g_S - _kl_k2)
   + nR * (_k4r_scan * g_R - _kl_k4r)
   - w_C * (_c_scan / micS) ** 2)
    return float(_c_scan[np.argmax(H)])

# ════════════════════════════════════════════════════════════════════
# PARALLEL DOSE MAP  — worker
# ════════════════════════════════════════════════════════════════════
def _dose_node(args):
    ns, nr, cap = args
    return (ns, nr, min(optimal_dose(ns, nr), cap))

def build_dose_map_parallel(DS, DR, cap=10.0):
    jobs = [(ns, nr, cap * micS) for ns in DS for nr in DR]
    cmap = np.zeros((len(DS), len(DR)))
    nw   = cpu_count()
    with Pool(nw) as pool:
        for ns, nr, c in pool.imap_unordered(_dose_node, jobs, chunksize=8):
            i = np.searchsorted(DS, ns)
            j = np.searchsorted(DR, nr)
            cmap[i, j] = c
    return cmap

# ════════════════════════════════════════════════════════════════════
# PARALLEL DOSE SLICES — worker
# ════════════════════════════════════════════════════════════════════
def _dose_slice_node(args):
    ns, nr = args
    return (ns, nr, optimal_dose(ns, nr))

def build_dose_slices_parallel(nS_scan, nR_slices):
    jobs = [(ns, nr) for nr in nR_slices for ns in nS_scan]
    results = {nr: np.zeros(len(nS_scan)) for nr in nR_slices}
    nw = cpu_count()
    with Pool(nw) as pool:
        for ns, nr, c in pool.imap_unordered(_dose_slice_node, jobs, chunksize=16):
            idx = np.searchsorted(nS_scan, ns)
            results[nr][idx] = c
    return results

# ════════════════════════════════════════════════════════════════════
# CONTROLLED SSA
# ════════════════════════════════════════════════════════════════════
def controlled_run(nS0, nR0, dosing_rule, rng, Tmax=T_HORIZON, record_dt=0.2):
    nS, nR, t = float(nS0), float(nR0), 0.0
    J = 0.0; T_exit = None
    ts  = [0.0]; Ss = [nS]; Rs = [nR]; Cs = [dosing_rule(nS, nR, 0.0)]
    next_rec = record_dt
    while t < Tmax:
        if is_absorbing(nS, nR) and T_exit is None:
            T_exit = t; J += w_Rt * nR / K; break
        c    = dosing_rule(nS, nR, t)
        aS   = alpha_S(c); aR = alpha_R(c)
        comp = max(1.0 - (nS + nR) / K, 0.0)
        a0   = k1 * nS * comp
        a1   = nS * (k4 + aS)
        a2   = k3 * nR * comp
        a3   = nR * (k4 + aR)
        a4   = mutation_propensity(nS, nR, c, comp)
        a    = a0 + a1 + a2 + a3 + a4
        if a <= 0: break
        dt    = -np.log(rng.random()) / a
        J_inc = (w_T + w_N * (nS + nR) / K + w_Rr * nR / K + w_C * (c / micS) ** 2) * dt
        if np.isfinite(J_inc): J += J_inc
        t += dt
        u = rng.random() * a
        if   u < a0:                  nS += 1
        elif u < a0 + a1:             nS -= 1
        elif u < a0 + a1 + a2:        nR += 1
        elif u < a0 + a1 + a2 + a3:   nR -= 1
        else:                         nS -= 1; nR += 1
        nS = max(0.0, nS); nR = max(0.0, nR)
        while next_rec <= t and next_rec <= Tmax:
            ts.append(next_rec); Ss.append(nS); Rs.append(nR)
            Cs.append(dosing_rule(nS, nR, next_rec))
            next_rec += record_dt
    if T_exit is None:
        T_exit = Tmax; J += w_Rt * nR / K
    return (np.array(ts), np.array(Ss), np.array(Rs),
            np.array(Cs), J, T_exit)

# ════════════════════════════════════════════════════════════════════
# DOSING RULES
# ════════════════════════════════════════════════════════════════════
def rule_uncontrolled(nS, nR, t): return 0.0
def rule_optimal(nS, nR, t):      return optimal_dose(nS, nR)

def make_constant(c):
    def rule(nS, nR, t): return c
    return rule

def make_interval_optimal(tau):
    state = {"last_check": -1e9, "c": 0.0}
    def rule(nS, nR, t):
        if t - state["last_check"] >= tau:
            state["c"] = optimal_dose(nS, nR)
            state["last_check"] = t
        return state["c"]
    return rule

# ════════════════════════════════════════════════════════════════════
# PARALLEL STRATEGY RUNS
# Each (strategy_idx, traj_idx) pair is one job → all run in parallel
# ════════════════════════════════════════════════════════════════════

# Strategy definitions as plain data (lambda-free for pickle)
# We encode strategies as (type, param) tuples
STRAT_DEFS = [
    ("uncontrolled",  None),
    ("optimal",       None),
    ("interval",      4.0),
    ("interval",      8.0),
    ("interval",      16.0),
    ("constant",      1.0 * micS),   # evaluated at module level - safe
    ("constant",      2.0 * micS),
    ("constant",      4.0 * micS),
]
STRAT_NAMES = [
    "Uncontrolled (c=0)",
    "Optimal FK (continuous)",
    "Interval-optimal q4h",
    "Interval-optimal q8h",
    "Interval-optimal q16h",
    f"Constant 1×MIC ({micS:.1f})",
    f"Constant 2×MIC ({2*micS:.1f})",
    f"Constant 4×MIC ({4*micS:.1f})",
]

def _make_rule(stype, param):
    """Reconstruct rule from picklable (type, param) — called inside worker."""
    if stype == "uncontrolled": return rule_uncontrolled
    if stype == "optimal":      return rule_optimal
    if stype == "interval":     return make_interval_optimal(param)
    if stype == "constant":     return make_constant(param)
    raise ValueError(f"Unknown strategy type: {stype}")

def _strat_traj_worker(args):
    """One trajectory for one strategy — fully independent, safe to pickle."""
    strat_idx, traj_idx, stype, param, seed = args
    rng  = np.random.RandomState(seed)
    rule = _make_rule(stype, param)
    ts, Ss, Rs, Cs, J, Te = controlled_run(nS_Start, nR_Start, rule, rng)
    return (strat_idx, traj_idx, ts, Ss, Rs, Cs, J, Te)

def run_all_strategies_parallel(common_ts):
    """Dispatch ALL (strategy × trajectory) pairs to the pool at once."""
    jobs = []
    seed = 5000
    for si, (stype, param) in enumerate(STRAT_DEFS):
        for ti in range(N_CMP):
            jobs.append((si, ti, stype, param, seed))
            seed += 1

    n_strats = len(STRAT_DEFS)
    nt       = len(common_ts)
    all_Ss   = np.zeros((n_strats, N_CMP, nt))
    all_Rs   = np.zeros((n_strats, N_CMP, nt))
    all_Cs   = np.zeros((n_strats, N_CMP, nt))
    all_Js   = np.zeros((n_strats, N_CMP))
    all_Te   = np.zeros((n_strats, N_CMP))

    nw   = cpu_count()
    total = len(jobs)
    print(f"Strategy runs: {total} jobs across {nw} cores ...")
    with Pool(nw) as pool:
        done = 0
        for si, ti, ts, Ss, Rs, Cs, J, Te in pool.imap_unordered(
                _strat_traj_worker, jobs, chunksize=8):
            all_Ss[si, ti] = np.interp(common_ts, ts, Ss, right=0.0)
            all_Rs[si, ti] = np.interp(common_ts, ts, Rs, right=0.0)
            all_Cs[si, ti] = np.interp(common_ts, ts, Cs, right=0.0)
            all_Js[si, ti] = J
            all_Te[si, ti] = Te
            done += 1
            if done % 100 == 0:
                print(f"  {done}/{total} trajectory jobs done")

    # assemble results dict
    results = {}
    for si, name in enumerate(STRAT_NAMES):
        Js = all_Js[si]
        Te = all_Te[si]
        pc = np.mean(Te < T_HORIZON)
        results[name] = {
            "J_mean":     float(np.mean(Js)),
            "J_std":      float(np.std(Js)),
            "Texit_mean": float(np.mean(Te)),
            "p_clear":    float(pc),
            "avg_Ss":     np.mean(all_Ss[si], axis=0),
            "std_Ss":     np.std(all_Ss[si],  axis=0),
            "avg_Rs":     np.mean(all_Rs[si], axis=0),
            "std_Rs":     np.std(all_Rs[si],  axis=0),
            "avg_Cs":     np.mean(all_Cs[si], axis=0),
        }
        print(f"  {name:32s}  J={np.mean(Js):8.3f} ± {np.std(Js):7.3f}"
              f"  T_exit={np.mean(Te):5.1f}h  P(clear)={pc:.2f}")
    return results

# ════════════════════════════════════════════════════════════════════
# PLOT HELPER
# ════════════════════════════════════════════════════════════════════
def plot_trajectory_panel(ax, common_ts, res, name):
    aS, sS = res["avg_Ss"], res["std_Ss"]
    aR, sR = res["avg_Rs"], res["std_Rs"]
    aC     = res["avg_Cs"]
    ax.fill_between(common_ts, np.maximum(0, aS - sS), aS + sS,
                    color='#2980b9', alpha=0.15)
    ax.fill_between(common_ts, np.maximum(0, aR - sR), aR + sR,
                    color='#e74c3c', alpha=0.15)
    ax.plot(common_ts, aS, color='#2980b9', lw=2.0, label='Mean nS')
    ax.plot(common_ts, aR, color='#e74c3c', lw=2.0, label='Mean nR')
    ax.axhline(N_immune, color='green', ls=':', lw=1.2,
               label=f'N_immune={N_immune}')
    ax.set_title(f"{name}\nJ={res['J_mean']:.2f}±{res['J_std']:.2f}  "
                 f"T={res['Texit_mean']:.1f}h  P(clr)={res['p_clear']:.2f}",
                 fontsize=8)
    ax.set_xlabel("time (h)"); ax.set_ylabel("cells")
    ax.legend(fontsize=6, loc='upper right'); ax.grid(alpha=0.2)
    ax2 = ax.twinx()
    ax2.plot(common_ts, aC, color='gray', lw=1.0, alpha=0.7, ls='--')
    ax2.set_ylabel("avg c (mg/L)", color='gray', fontsize=7)
    ax2.tick_params(axis='y', labelcolor='gray', labelsize=6)

# ════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    print(f"Parameters: micS={micS:.3f}, micR={micR:.1f}, "
          f"nS_eq={nS_eq:.0f}, T_HORIZON={T_HORIZON}")
    print(f"Weights: w_T={w_T}, w_N={w_N}, w_Rr={w_Rr}, w_Rt={w_Rt}, beta={beta}")
    print(f"CPU cores available: {cpu_count()}")

    nS_t, nR_t = int(nS_eq), 1
    U_t = -w_T - w_N * (nS_t + nR_t) / K - w_Rr * nR_t / K
    print(f"U_running at ({nS_t},{nR_t}) = {U_t:.4f}  "
          f"(β·U over horizon ≈ {beta*U_t*T_HORIZON:.1f})")
    if abs(beta * U_t * T_HORIZON) > 500:
        print("WARNING: |β·U·T| > 500 — possible log-sum-exp underflow.")

    # ── 1. FK precompute (parallel) ───────────────────────────────
    Vgrid = precompute_V_parallel()
    print(f"Vgrid: min={Vgrid.min():.3f}, max={Vgrid.max():.3f}, "
          f"bad={(Vgrid < -1e5).sum()} nodes")
    globals()['V_interp'] = RegularGridInterpolator(
        (GS, GR), Vgrid, bounds_error=False, fill_value=None)

    # ── 2. Dose map (parallel) ────────────────────────────────────
    print("\nBuilding dose map (parallel) ...")
    DS = np.arange(0, K + 1,      25)
    DR = np.arange(0, K // 5 + 1, 10)
    cmap_grid = build_dose_map_parallel(DS, DR, cap=10.0)

    # ── 3. Dose slices (parallel) ─────────────────────────────────
    print("Building dose slices (parallel) ...")
    nR_slices = [0, 1, 5, 20, 50]
    nS_scan   = np.arange(0, K + 1, 10)
    dose_slices = build_dose_slices_parallel(nS_scan, nR_slices)

    # ── 4. All strategy runs (parallel) ───────────────────────────
    print("\n(B) Running all strategies in parallel ...")
    common_ts    = np.arange(0, T_HORIZON + 0.2, 0.2)
    strat_results = run_all_strategies_parallel(common_ts)
    strat_names   = list(strat_results.keys())

    # ── 5. Output folder ──────────────────────────────────────────
    base_dir = os.path.join("figures", "fk_weight")
    os.makedirs(base_dir, exist_ok=True)
    existing = [d for d in os.listdir(base_dir)
                if d.startswith("run_") and
                os.path.isdir(os.path.join(base_dir, d))]
    nums = []
    for d in existing:
        try: nums.append(int(d[4:]))
        except ValueError: pass
    run_idx = max(nums) + 1 if nums else 1
    run_dir = os.path.join(base_dir, f"run_{run_idx:03d}")
    os.makedirs(run_dir, exist_ok=True)

    suptitle_base = (
        f"FK Optimal Control — w_T={w_T}  w_N={w_N}  w_Rr={w_Rr}  "
        f"w_Rt={w_Rt}  β={beta}\n"
        f"micS={micS:.2f}  micR={micR:.1f}  nS_eq={nS_eq:.0f}  "
        f"N_immune={N_immune}  T_H={T_HORIZON}h"
    )

    # ════════════════════════════════════════════════════════════════
    # PAGE 1
    # ════════════════════════════════════════════════════════════════
    print("\nPlotting page 1 ...")
    fig1 = plt.figure(figsize=(18, 15))
    gs1  = fig1.add_gridspec(3, 3, hspace=0.42, wspace=0.30)
    fig1.suptitle(suptitle_base + "  |  Page 1 of 2",
                  fontsize=10, fontweight='bold', y=0.99)

    ax = fig1.add_subplot(gs1[0, 0])
    im = ax.imshow(Vgrid.T, origin='lower', aspect='auto',
                   extent=[GS[0], GS[-1], GR[0], GR[-1]], cmap='viridis')
    ax.set_title("Value function V(nS, nR)")
    ax.set_xlabel("nS"); ax.set_ylabel("nR"); plt.colorbar(im, ax=ax)

    ax = fig1.add_subplot(gs1[0, 1])
    im = ax.imshow(cmap_grid.T, origin='lower', aspect='auto',
                   extent=[DS[0], DS[-1], DR[0], DR[-1]], cmap='inferno')
    ax.axvline(nS_eq, color='cyan', ls='--', lw=1, label=f'nS_eq={nS_eq:.0f}')
    ax.set_title("Optimal dose  c*(nS, nR)  [mg/L]")
    ax.set_xlabel("nS"); ax.set_ylabel("nR")
    ax.legend(fontsize=7); plt.colorbar(im, ax=ax)

    ax = fig1.add_subplot(gs1[0, 2])
    colors_sl = plt.cm.plasma(np.linspace(0.1, 0.9, len(nR_slices)))
    for nr_val, col in zip(nR_slices, colors_sl):
        ax.plot(nS_scan, dose_slices[nr_val], lw=1.6,
                label=f"nR={nr_val}", color=col)
    ax.axvline(nS_eq, color='gray', ls=':', lw=1, label='nS_eq')
    ax.axhline(micS,  color='blue', ls='--', lw=1, label='micS')
    ax.axhline(0,     color='black', lw=0.6)
    ax.set_title("Optimal dose c*(nS) at fixed nR")
    ax.set_xlabel("nS"); ax.set_ylabel("c* (mg/L)")
    ax.legend(fontsize=7); ax.grid(alpha=0.25); ax.set_ylim(bottom=0)

    traj_positions = [(1,0),(1,1),(1,2),(2,0),(2,1)]
    for idx, (r, c) in enumerate(traj_positions):
        ax = fig1.add_subplot(gs1[r, c])
        plot_trajectory_panel(ax, common_ts,
                              strat_results[strat_names[idx]],
                              strat_names[idx])
    fig1.add_subplot(gs1[2, 2]).axis('off')

    p1_path = os.path.join(run_dir, f"page1_{run_idx:03d}.png")
    plt.savefig(p1_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {p1_path}")

    # ════════════════════════════════════════════════════════════════
    # PAGE 2
    # ════════════════════════════════════════════════════════════════
    print("Plotting page 2 ...")
    fig2 = plt.figure(figsize=(18, 10))
    gs2  = fig2.add_gridspec(2, 3, hspace=0.42, wspace=0.30,
                              height_ratios=[1, 1.1])
    fig2.suptitle(suptitle_base + "  |  Page 2 of 2",
                  fontsize=10, fontweight='bold', y=0.99)

    for idx, name in enumerate(strat_names[5:]):
        ax = fig2.add_subplot(gs2[0, idx])
        plot_trajectory_panel(ax, common_ts, strat_results[name], name)

    # J comparison bar — all strategies
    ax_bar   = fig2.add_subplot(gs2[1, :])
    Js_all   = [strat_results[n]["J_mean"]  for n in strat_names]
    Jerr_all = [strat_results[n]["J_std"]   for n in strat_names]
    Tc_all   = [strat_results[n]["p_clear"] for n in strat_names]
    bar_colors = ['#7f8c8d','#1a7a2e','#27ae60','#52c47a','#85d6a0',
                  '#2471a3','#5499c7','#85b9de']

    bars = ax_bar.barh(list(range(len(strat_names))), Js_all,
                       xerr=Jerr_all, color=bar_colors,
                       capsize=4, alpha=0.88)
    ax_bar.set_yticks(list(range(len(strat_names))))
    ax_bar.set_yticklabels(strat_names, fontsize=9)
    ax_bar.set_xlabel("Objective J  (lower = better)", fontsize=10)
    ax_bar.set_title(
        f"All-strategy J comparison  ({N_CMP} trajectories each)\n"
        "Numbers = mean J  |  Error bars = ±1 std  |  P(clear) on right",
        fontsize=9)
    ax_bar.grid(alpha=0.3, axis='x')
    ax_bar.invert_yaxis()
    x_max = max(Js_all) * 1.18
    ax_bar.set_xlim(0, x_max)
    for bar, j, jerr, pc in zip(bars, Js_all, Jerr_all, Tc_all):
        ax_bar.text(max(j * 0.97, 2),
                    bar.get_y() + bar.get_height() / 2,
                    f"{j:.1f}", va='center', ha='right',
                    fontsize=8, color='white', fontweight='bold')
        ax_bar.text(j + jerr + x_max * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"P(clr)={pc:.2f}", va='center', ha='left',
                    fontsize=7, color='#333333')

    p2_path = os.path.join(run_dir, f"page2_{run_idx:03d}.png")
    plt.savefig(p2_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {p2_path}")

    # ── Summary ───────────────────────────────────────────────────
    print(f"\nOutput: {run_dir}/")
    print("\n" + "=" * 72)
    print(f"{'Strategy':<32} {'J':>9} {'±std':>8}  {'T_exit':>7}h  {'P(clr)':>7}")
    print("-" * 72)
    for name, res in strat_results.items():
        print(f"{name:<32} {res['J_mean']:>9.3f} "
              f"{res['J_std']:>8.3f}  {res['Texit_mean']:>7.1f}  "
              f"{res['p_clear']:>7.2f}")
    print("\nWhat to look for:")
    print("  1. Value-function panel: GRADIENT present?  (flat = bad)")
    print("  2. Dose-slice panel:     c*(nS) rises with nS?")
    print("  3. Optimal J < all others?  (should be lowest)")
    print("  4. Interval strategies: does q4h ≈ continuous?")
    print("  5. Constant doses: does 4×MIC beat 1×MIC on J?")