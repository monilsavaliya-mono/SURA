"""
FK Weight Test — FULLY PARALLELISED CPU VERSION + PK FOR INTERVAL DOSING
=========================================================================
What changed vs. the previous version (w_C toxicity-cost version):

  Added a real one-compartment PK layer (PK_halflife / PK_lambda) for
  the "Interval-optimal" strategies, via a new dosing-rule contract.

  OLD contract:  dosing_rule(nS, nR, t)        -> c   (instantaneous
                 concentration, used directly by the SSA every step)

  NEW contract:  dosing_rule(nS, nR, t, c_pk)  -> (mode, value)

      mode == "set"    -> c_pk is INSTANTLY set to `value` and held
                          there (no decay) until the next dosing
                          decision. This is the old behaviour —
                          an idealised infusion pump that can track
                          any target concentration with zero lag.
                          Used by: Uncontrolled, Constant *MIC,
                          Optimal FK (continuous).

      mode == "bolus"  -> `value` (a dose, in concentration units) is
                          ADDED to c_pk. Between dosing decisions, c_pk
                          decays as c_pk *= exp(-PK_lambda * dt). If a
                          new bolus arrives before the previous one has
                          fully cleared, it stacks on top (linear PK
                          superposition / additive accumulation).
                          Used by: Interval-optimal q4h/q8h/q16h.

  controlled_run() now carries c_pk as PK state across SSA steps and
  uses it (not the raw dosing_rule output) for alpha_S/alpha_R,
  mutation_propensity, and the w_C toxicity term — so toxicity and
  killing both track *actual drug exposure*, not the controller's
  nominal setpoint/dose.

  NOTE / scope: Vgrid / V_interp / optimal_dose() are UNCHANGED — they
  still live in the no-PK "instantaneous setpoint" world. For "bolus"
  strategies, optimal_dose(nS,nR) is reused as "the dose AMOUNT (in
  concentration units) to administer now", which is a simplification:
  a fully rigorous PK-aware HJB solve would need V(nS, nR, c_pk) on a
  3D grid. This is a reasonable Phase-3 approximation; flag it as a
  Phase-4 extension if you want full rigor.
  FIX 1: optimal no longer quits when nS=0, nR>0 → keeps dosing to kill R
       → optimal should finally beat 4×MIC in resistance-heavy cases

    FIX 2: controller penalizes sub-MIC doses that breed mutants
        → optimal dose map should AVOID the mid-range near micS
        → directly demonstrates your "sub-MIC is dangerous" thesis

    FIX 3: results stay correct if you ever change beta
        → can now do risk-sensitivity experiments safely

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from multiprocessing import Pool, cpu_count
from mpi4py.futures import MPIPoolExecutor
import os

# ════════════════════════════════════════════════════════════════════
# PARAMETERS
# ════════════════════════════════════════════════════════════════════
K          = 2000
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
MIC_FACTOR = 32
micR       = MIC_FACTOR * micS
nS_eq      = K * (k1 - k4) / k1

# ── utility weights ───────────────────────────────────────────────
w_T   = 1.0
w_N   = 1.0
w_Rr  = 100.0
w_Rt  = 1000.0
w_C   = 0.1
# ─────────────────────────────────────────────────────────────────

beta       = 1.0
N_immune   = 10
c_base     = micS * 3
T_HORIZON  = 100.0
n_fk       = 200          # FK trajectories per grid node
N_CMP      = 100          # comparison trajectories per strategy

# ── PK for interval (bolus) dosing ─────────────────────────────────
PK_halflife = 1.0                       # meropenem ~1 h
PK_lambda   = np.log(2) / PK_halflife   # elimination rate
# ─────────────────────────────────────────────────────────────────

V_interp   = None
nS_Start   = int(nS_eq)
nR_Start   = 1

max_dose = 12.0*micS   # max dose allowed for interval strategies
dose_slicing_accuracy = 200 # for plotting dose slices
# ── FK grid ───────────────────────────────────────────────────────
GS = np.arange(0, K + 1,      50)
GR = np.arange(0, K // 5 + 1, 10)

# ── dose scan (vectorised) ────────────────────────────────────────
_c_scan = np.linspace(0.0, max_dose, dose_slicing_accuracy )   # shape (200,)

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
# FK BASELINE SSA  (unchanged — still the no-PK "setpoint" world,
# c_base is a constant reference concentration for the KL baseline)
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
    print(f"Launching FK grid ({len(jobs)} nodes) via MPI across all allocated nodes...")
    with MPIPoolExecutor() as executor:
        results = executor.map(_fk_node, jobs, chunksize=10)
        for i, j, v in results:
            Vgrid[i, j] = v
    return Vgrid

def V_at(nS, nR):
    nS = np.clip(nS, GS[0], GS[-1])
    nR = np.clip(nR, GR[0], GR[-1])
    return float(V_interp((nS, nR)))

# ════════════════════════════════════════════════════════════════════
# OPTIMAL DOSE — fully vectorised (no Python loop over c)
# Still operates in the "instantaneous concentration" world. Reused
# below as a SETPOINT (continuous strategy) or as a BOLUS AMOUNT
# (interval strategy).
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
_stress_scan = 1.0 + (F_max - 1.0) * ((_c_scan/micS) / (1.0 + (_c_scan/micS)**2)) / 0.5
_stress_scan[0] = 1.0   # c=0 → factor 1
_stress_base = stress_factor(c_base)

def optimal_dose(nS, nR, c_floor=0.0):
    if is_absorbing(nS, nR):           # FIX 1: removed nS<1
        return 0.0
    v_here = V_at(nS, nR)
    g_S = (V_at(nS - 1, nR)     - v_here) if nS >= 1 else 0.0
    g_R = (V_at(nS, nR - 1)     - v_here) if nR >= 1 else 0.0
    g_M = (V_at(nS - 1, nR + 1) - v_here) if nS >= 1 else 0.0   # FIX 2
    if not (np.isfinite(g_S) and np.isfinite(g_R) and np.isfinite(g_M)):
        return 0.0

    comp = max(1.0 - (nS + nR) / K, 0.0)
    mut_base = k5_per_div * _stress_base * k1 * nS * comp

    mask = _c_scan >= c_floor
    mut_rates = k5_per_div * _stress_scan[mask] * k1 * nS * comp
    kl_mut    = kl_term_vec(mut_rates, mut_base) if mut_base > 0 else 0.0

    # FIX 3: beta on ALL non-gradient terms (KL + drug cost)
    H = (nS * (_k2_scan[mask]  * g_S - beta * _kl_k2[mask])
       + nR * (_k4r_scan[mask] * g_R - beta * _kl_k4r[mask])
       + (mut_rates * g_M - beta * kl_mut)
       - beta * w_C * (_c_scan[mask] / micS) ** 2)
    return float(_c_scan[mask][np.argmax(H)])

# ════════════════════════════════════════════════════════════════════
# PARALLEL DOSE MAP  — worker  (unchanged)
# ════════════════════════════════════════════════════════════════════
def _dose_node(args):
    ns, nr, cap, vgrid_data = args
    global V_interp
    if V_interp is None and vgrid_data is not None:
        V_interp = RegularGridInterpolator((GS, GR), vgrid_data, bounds_error=False, fill_value=None)
    return (ns, nr, min(optimal_dose(ns, nr), cap))

def build_dose_map_parallel(DS, DR, vgrid_data, cap=10.0):
    jobs = [(ns, nr, cap * micS, vgrid_data) for ns in DS for nr in DR]
    cmap = np.zeros((len(DS), len(DR)))
    with MPIPoolExecutor() as executor:
        results = executor.map(_dose_node, jobs, chunksize=10)
        for ns, nr, c in results:
            i = np.searchsorted(DS, ns)
            j = np.searchsorted(DR, nr)
            cmap[i, j] = c
    return cmap

# ════════════════════════════════════════════════════════════════════
# PARALLEL DOSE SLICES — worker  (unchanged)
# ════════════════════════════════════════════════════════════════════
def _dose_slice_node(args):
    ns, nr, vgrid_data = args
    global V_interp
    if V_interp is None and vgrid_data is not None:
        V_interp = RegularGridInterpolator((GS, GR), vgrid_data, bounds_error=False, fill_value=None)
    return (ns, nr, optimal_dose(ns, nr))

def build_dose_slices_parallel(nS_scan, nR_slices, vgrid_data):
    jobs = [(ns, nr, vgrid_data) for nr in nR_slices for ns in nS_scan]
    results = {nr: np.zeros(len(nS_scan)) for nr in nR_slices}
    with MPIPoolExecutor() as executor:
        res_list = executor.map(_dose_slice_node, jobs, chunksize=10)
        for ns, nr, c in res_list:
            idx = np.searchsorted(nS_scan, ns)
            results[nr][idx] = c
    return results

# ════════════════════════════════════════════════════════════════════
# CONTROLLED SSA — now PK-aware
#
#   c_pk        : current PK state (drug concentration "in the body")
#   cur_mode/   : the dosing regime decided at the START of the
#   cur_val       current inter-reaction interval
#       "set"   -> c_pk held == cur_val for the whole interval
#                  (no decay; idealised infusion-pump tracking)
#       "bolus" -> c_pk decays as c_pk * exp(-PK_lambda * dt) over
#                  the interval, starting from its value at t_old
#
# alpha_S/alpha_R/mutation_propensity/J's toxicity term all use c_pk
# (actual exposure), not the dosing_rule's raw output.
# ════════════════════════════════════════════════════════════════════
def controlled_run(nS0, nR0, dosing_rule, rng, Tmax=T_HORIZON, record_dt=0.2):
    nS, nR, t = float(nS0), float(nR0), 0.0
    J = 0.0; T_exit = None

    # initial dosing decision at t=0
    mode, val = dosing_rule(nS, nR, 0.0, 0.0)
    c_pk = val if mode == "set" else val          # c_pk starts at 0 + val
    cur_mode, cur_val = mode, val

    ts  = [0.0]; Ss = [nS]; Rs = [nR]; Cs = [c_pk]
    next_rec = record_dt

    while t < Tmax:
        if is_absorbing(nS, nR) and T_exit is None:
            T_exit = t; J += w_Rt * nR / K; break

        c    = c_pk                                # exposure for THIS interval
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
        J_inc = (w_T + w_N * (nS + nR) / K + w_Rr * nR / K
                 + w_C * (c / micS) ** 2) * dt
        if np.isfinite(J_inc): J += J_inc

        t_old, c_old = t, c_pk
        t += dt
        if cur_mode == "set":
            c_pk = cur_val                          # held flat, no decay
        else:
            c_pk = c_pk * np.exp(-PK_lambda * dt)   # PK elimination

        # reaction
        u = rng.random() * a
        if   u < a0:                  nS += 1
        elif u < a0 + a1:             nS -= 1
        elif u < a0 + a1 + a2:        nR += 1
        elif u < a0 + a1 + a2 + a3:   nR -= 1
        else:                         nS -= 1; nR += 1
        nS = max(0.0, nS); nR = max(0.0, nR)

        # record — Ss/Rs use post-jump state (matches original
        # behaviour); Cs uses the analytic PK value at next_rec
        while next_rec <= t and next_rec <= Tmax:
            if cur_mode == "set":
                Cval = cur_val
            else:
                Cval = c_old * np.exp(-PK_lambda * (next_rec - t_old))
            ts.append(next_rec); Ss.append(nS); Rs.append(nR); Cs.append(Cval)
            next_rec += record_dt

        # dosing decision for the NEXT interval
        mode, val = dosing_rule(nS, nR, t, c_pk)
        if mode == "set":
            c_pk = val
        else:
            c_pk = c_pk + val
        cur_mode, cur_val = mode, val

    if T_exit is None:
        T_exit = Tmax; J += w_Rt * nR / K
    return (np.array(ts), np.array(Ss), np.array(Rs),
            np.array(Cs), J, T_exit)

# ════════════════════════════════════════════════════════════════════
# DOSING RULES — new (mode, value) contract
# ════════════════════════════════════════════════════════════════════
def rule_uncontrolled(nS, nR, t, c_pk):
    return ("set", 0.0)

def rule_optimal(nS, nR, t, c_pk):
    return ("set", optimal_dose(nS, nR))

def make_constant(c):
    def rule(nS, nR, t, c_pk):
        return ("set", c)
    return rule

def make_interval_bolus(tau):
    """Administer a bolus dose D = optimal_dose(nS, nR) every tau hours
    (first dose at t=0). Between doses c_pk decays as
    exp(-PK_lambda * dt). A dose arriving before the previous one has
    cleared adds on top (linear PK superposition)."""
    state = {"last_dose": -1e9}
    def rule(nS, nR, t, c_pk):
        if t - state["last_dose"] >= tau:
            state["last_dose"] = t
            c_star = optimal_dose(nS, nR, c_floor=c_pk)
            return ("bolus", max(0.0, c_star - c_pk))
        return ("bolus", 0.0)
    return rule

# ════════════════════════════════════════════════════════════════════
# PARALLEL STRATEGY RUNS
# Each (strategy_idx, traj_idx) pair is one job → all run in parallel
# ════════════════════════════════════════════════════════════════════
STRAT_DEFS = [
    ("uncontrolled",  None),
    ("optimal",       None),
    ("interval",      4.0),
    ("interval",      8.0),
    ("interval",      16.0),
    ("constant",      1.0 * micS),
    ("constant",      2.0 * micS),
    ("constant",      4.0 * micS),
]
STRAT_NAMES = [
    "Uncontrolled (c=0)",
    "Optimal FK (continuous)",
    "Bolus q-optimal q4h (PK)",
    "Bolus q-optimal q8h (PK)",
    "Bolus q-optimal q16h (PK)",
    f"Constant 1×MIC ({micS:.1f})",
    f"Constant 2×MIC ({2*micS:.1f})",
    f"Constant 4×MIC ({4*micS:.1f})",
]

def _make_rule(stype, param):
    """Reconstruct rule from picklable (type, param) — called inside worker."""
    if stype == "uncontrolled": return rule_uncontrolled
    if stype == "optimal":      return rule_optimal
    if stype == "interval":     return make_interval_bolus(param)
    if stype == "constant":     return make_constant(param)
    raise ValueError(f"Unknown strategy type: {stype}")

def _strat_traj_worker(args):
    """One trajectory for one strategy — fully independent, safe to pickle."""
    strat_idx, traj_idx, stype, param, seed, vgrid_data = args
    global V_interp
    if V_interp is None and vgrid_data is not None:
        V_interp = RegularGridInterpolator((GS, GR), vgrid_data, bounds_error=False, fill_value=None)
    rng  = np.random.RandomState(seed)
    rule = _make_rule(stype, param)
    ts, Ss, Rs, Cs, J, Te = controlled_run(nS_Start, nR_Start, rule, rng)
    return (strat_idx, traj_idx, ts, Ss, Rs, Cs, J, Te)

def run_all_strategies_parallel(common_ts, vgrid_data):
    """Dispatch ALL (strategy × trajectory) pairs to the pool at once."""
    jobs = []
    seed = 5000
    for si, (stype, param) in enumerate(STRAT_DEFS):
        for ti in range(N_CMP):
            jobs.append((si, ti, stype, param, seed, vgrid_data))
            seed += 1

    n_strats = len(STRAT_DEFS)
    nt       = len(common_ts)
    all_Ss   = np.zeros((n_strats, N_CMP, nt))
    all_Rs   = np.zeros((n_strats, N_CMP, nt))
    all_Cs   = np.zeros((n_strats, N_CMP, nt))
    all_Js   = np.zeros((n_strats, N_CMP))
    all_Te   = np.zeros((n_strats, N_CMP))

    total = len(jobs)
    print(f"Strategy runs: Running {total} trajectory jobs via MPI distributed pool...")
    with MPIPoolExecutor() as executor:
        results = executor.map(_strat_traj_worker, jobs, chunksize=5)
        for si, ti, ts, Ss, Rs, Cs, J, Te in results:
            all_Ss[si, ti] = np.interp(common_ts, ts, Ss, right=0.0)
            all_Rs[si, ti] = np.interp(common_ts, ts, Rs, right=0.0)
            all_Cs[si, ti] = np.interp(common_ts, ts, Cs, right=0.0)
            all_Js[si, ti] = J
            all_Te[si, ti] = Te

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
    print(f"Weights: w_T={w_T}, w_N={w_N}, w_Rr={w_Rr}, w_Rt={w_Rt}, "
          f"w_C={w_C}, beta={beta}")
    print(f"PK: half-life={PK_halflife}h  lambda={PK_lambda:.3f}/h")
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
    cmap_grid = build_dose_map_parallel(DS, DR, Vgrid, cap=10.0)

    # ── 3. Dose slices (parallel) ─────────────────────────────────
    print("Building dose slices (parallel) ...")
    nR_slices = [0, 1, 5, 20, 50]
    nS_scan   = np.arange(0, K + 1, 10)
    dose_slices = build_dose_slices_parallel(nS_scan, nR_slices, Vgrid)

    # ── 4. All strategy runs (parallel) ───────────────────────────
    print("\n(B) Running all strategies in parallel ...")
    common_ts    = np.arange(0, T_HORIZON + 0.2, 0.2)
    strat_results = run_all_strategies_parallel(common_ts, Vgrid)
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
        f"w_Rt={w_Rt}  w_C={w_C}  β={beta}\n"
        f"micS={micS:.2f}  micR={micR:.1f}  nS_eq={nS_eq:.0f}  "
        f"N_immune={N_immune}  T_H={T_HORIZON}h  PK_t½={PK_halflife}h"
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
    print("  3. Bolus q*h panels: avg-c (gray dashed) should show a")
    print("     SAWTOOTH — spikes at each dose, decaying toward zero")
    print("     between doses (peaks may stack if tau < ~half-life).")
    print("  4. Constant/Optimal-continuous panels: avg-c still shows")
    print("     the step-down-to-zero pattern from population averaging")
    print("     (see earlier discussion) — that's unrelated to PK.")
    print("  5. Optimal J < all others?  (should be lowest)")   