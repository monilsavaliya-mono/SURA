"""
FK Weight Test — stripped version for testing utility weights.
MSW curve commented out (add back later after good results).
gs[0,2] replaced with c*(nS) slices — more useful for weight testing.

CURRENT WEIGHTS (change these to experiment):
  w_T  = 0.0   → no time penalty (test: does controller still clear?)
  w_N  = 1.0   → load penalty (carrying K cells for 1h = 1 cost unit)
  w_Rr = 100.0 → resistance running (10 R cells for 1h = 1 cost unit)
  w_Rt = 1000.0→ resistance terminal (1 R cell left = 1 cost unit)

NOTE on w_T=0: controller ignores time-to-clearance. It will minimize
load+resistance exposure but may not rush to clear. Set w_T=1.0 to
add time pressure back.
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
k3         = k1 * 0.927        # 0.679 (7.3% resistance cost, Vanacker 2023)
k4         = 0.179
k5_per_div = 7.8e-7             # per-division mutation rate (Luria-Delbrück)
F_max      = 3.0                # sub-MIC stress fold-increase (Gutiérrez 2013)

psi_S_max  = k1 - k4           # 0.554
psi_R_max  = k3 - k4           # 0.500
kmax       = 0.672
psiminS    = psi_S_max - kmax  # -0.118
psiminR    = psiminS
kappa      = 1.1
EC50       = 1.01
micS       = EC50 / ((-psiminS / psi_S_max) ** (1.0 / kappa))   # ~4.12
MIC_FACTOR = 8
micR       = MIC_FACTOR * micS  # ~33

nS_eq      = K * (k1 - k4) / k1  # ~756

# ── utility weights (CHANGE THESE to experiment) ──────────────────
w_T   = 1.0      # time penalty per hour
w_N   = 1.0      # load penalty (÷K normalised)
w_Rr  = 1.0    # running resistance penalty (÷K normalised)
w_Rt  = 1000.0   # terminal resistance penalty (÷K normalised)
# ──────────────────────────────────────────────────────────────────

beta       = 1.0
N_immune   = 10
c_base     = micS               # FK reference dose (net S growth = 0)
T_HORIZON  = 100.0              # treatment window (h)

PK_halflife = 1.0
PK_lambda   = np.log(2) / PK_halflife

V_interp   = None               # filled in main after parallel FK solve

nS_Start   = int(nS_eq)         # starting state (standing variation)
nR_Start   = 1

# ════════════════════════════════════════════════════════════════════
# PD CURVE
# ════════════════════════════════════════════════════════════════════
def alpha_S(c):
    if c <= 0: return 0.0
    u = (c / micS) ** kappa
    return (psi_S_max - psiminS) * u / (u - psiminS / psi_S_max)

def alpha_R(c):
    if c <= 0: return 0.0
    u = (c / micR) ** kappa
    return (psi_R_max - psiminR) * u / (u - psiminR / psi_R_max)

_span = psi_S_max - psiminS
_D    = psiminS / psi_S_max

def invert_alpha_S(alpha):
    if alpha <= 0:      return 0.0
    if alpha >= _span:  return 10.0 * micS   # cap instead of inf
    u = alpha * _D / (alpha - _span)
    if u <= 0: return 0.0
    return micS * u ** (1.0 / kappa)

# ════════════════════════════════════════════════════════════════════
# MUTATION (birth-coupled + sub-MIC stress, Luria-Delbrück + Gutiérrez 2013)
# ════════════════════════════════════════════════════════════════════
def stress_factor(c):
    """1 at c=0, peaks ~F_max at c=micS, falls at high c."""
    if c <= 0: return 1.0
    x = c / micS
    shape = x / (1.0 + x * x)   # peaks 0.5 at x=1
    return 1.0 + (F_max - 1.0) * (shape / 0.5)

def mutation_propensity(nS, nR, c, comp):
    """a_mut tied to S-birth events (Luria-Delbrück: mutations ∝ divisions)."""
    birth_S = k1 * nS * comp
    return k5_per_div * stress_factor(c) * birth_S

# ════════════════════════════════════════════════════════════════════
# UTILITY (normalised by K so all terms are order-1)
# ════════════════════════════════════════════════════════════════════
def U_running(nS, nR):
    return -w_T - w_N * (nS + nR) / K - w_Rr * nR / K

def U_exit(nS, nR):
    return -w_Rt * nR / K

def is_absorbing(nS, nR):
    return (nS + nR) <= N_immune

# ════════════════════════════════════════════════════════════════════
# FK BASELINE SSA (reference drug c_base = micS so trajectories clear)
# ════════════════════════════════════════════════════════════════════
_aS_base = alpha_S(c_base)
_aR_base = alpha_R(c_base)

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
        U_acc += U_running(nS, nR) * dt
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

def _fk_node(args):
    i, j, ns, nr, seed = args
    rng = np.random.default_rng(seed)
    return (i, j, feynman_kac_V(ns, nr, n_fk, rng))

# ════════════════════════════════════════════════════════════════════
# FK GRID (module-level so workers can read n_fk)
# ════════════════════════════════════════════════════════════════════
GS  = np.arange(0, K + 1,     50)   # nS grid  0..1000 step 50
GR  = np.arange(0, K // 5 + 1, 10)  # nR grid  0..200  step 10
n_fk = 200                           # trajectories per FK node

def precompute_V_parallel():
    jobs, seed = [], 12345
    for i, ns in enumerate(GS):
        for j, nr in enumerate(GR):
            jobs.append((i, j, ns, nr, seed))
            seed += 1
    Vgrid = np.zeros((len(GS), len(GR)))
    nw    = cpu_count()
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
# OPTIMAL DOSE (PRX Eq 32)
# k2_opt = k2_base * exp(V(nS-1,nR) - V(nS,nR))
# alpha_needed = k2_opt - k4  →  invert PD → c*
# ════════════════════════════════════════════════════════════════════
def kl_term(k, k0):
    """KL cost for one reaction: k·log(k/k0) − k + k0."""
    if k <= 0 or k0 <= 0:
        return 0.0
    return k * np.log(k / k0) - k + k0

# precompute baselines once (module level, near _aS_base):
_k2_base = k4 + _aS_base     # baseline S-death coeff
_k4_base = k4 + _aR_base     # baseline R-death coeff

# scan grid for c (module level):
_c_scan = np.linspace(0.0, 12.0 * micS, 200)

def optimal_dose(nS, nR):
    """Two-channel optimal dose: maximize H(c) over both S-death and
    R-death value gains. Reduces to single-channel when nR=0."""
    if is_absorbing(nS, nR) or nS < 1:
        return 0.0

    v_here = V_at(nS, nR)
    g_S = V_at(nS - 1, nR) - v_here if nS >= 1 else 0.0
    g_R = V_at(nS, nR - 1) - v_here if nR >= 1 else 0.0

    if not (np.isfinite(g_S) and np.isfinite(g_R)):
        return 0.0

    best_c, best_H = 0.0, -np.inf
    for c in _c_scan:
        aS = alpha_S(c)
        aR = alpha_R(c)
        k2 = k4 + aS
        k4r = k4 + aR
        # Hamiltonian: value gain minus KL control cost, both channels
        H = ( nS * (k2  * g_S - kl_term(k2,  _k2_base))
            + nR * (k4r * g_R - kl_term(k4r, _k4_base)) )
        if H > best_H:
            best_H, best_c = H, c
    return best_c

# ════════════════════════════════════════════════════════════════════
# CONTROLLED SSA (TRUE dynamics under any dosing_rule(nS,nR,t)->c)
# ════════════════════════════════════════════════════════════════════
def controlled_run(nS0, nR0, dosing_rule, rng, Tmax=T_HORIZON, record_dt=0.2):
    nS, nR, t = float(nS0), float(nR0), 0.0
    J      = 0.0
    T_exit = None
    ts  = [0.0]; Ss = [nS]; Rs = [nR]; Cs = [dosing_rule(nS, nR, 0.0)]
    next_rec = record_dt

    while t < Tmax:
        if is_absorbing(nS, nR) and T_exit is None:
            T_exit = t
            J += w_Rt * nR / K   # terminal cost
            break

        c   = dosing_rule(nS, nR, t)
        aS  = alpha_S(c); aR = alpha_R(c)
        comp = max(1.0 - (nS + nR) / K, 0.0)
        a0  = k1 * nS * comp
        a1  = nS * (k4 + aS)
        a2  = k3 * nR * comp
        a3  = nR * (k4 + aR)
        a4  = mutation_propensity(nS, nR, c, comp)
        a   = a0 + a1 + a2 + a3 + a4
        if a <= 0: break

        dt    = -np.log(rng.random()) / a
        J_inc = (w_T + w_N * (nS + nR) / K + w_Rr * nR / K) * dt
        if np.isfinite(J_inc):
            J += J_inc
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
        T_exit = Tmax
        J += w_Rt * nR / K

    return (np.array(ts), np.array(Ss), np.array(Rs),
            np.array(Cs), J, T_exit)

# ════════════════════════════════════════════════════════════════════
# DOSING RULES
# ════════════════════════════════════════════════════════════════════
def rule_uncontrolled(nS, nR, t): return 0.0
def rule_optimal(nS, nR, t):      return optimal_dose(nS, nR)

# ── extension point (uncomment to add to strategies dict below) ───
def make_constant(c):
    def rule(nS, nR, t): return c
    return rule

def make_interval_optimal(tau):
    """Re-evaluate c* every tau hours, hold constant between checks."""
    state = {"last_check": -1e9, "c": 0.0}
    def rule(nS, nR, t):
        if t - state["last_check"] >= tau:
            state["c"]          = optimal_dose(nS, nR)
            state["last_check"] = t
        return state["c"]
    return rule
# ─────────────────────────────────────────────────────────────────

# ════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    # ── 1. FK precompute ──────────────────────────────────────────
    print(f"Parameters: micS={micS:.3f}, micR={micR:.1f}, "
          f"nS_eq={nS_eq:.0f}, T_HORIZON={T_HORIZON}")
    print(f"Weights: w_T={w_T}, w_N={w_N}, w_Rr={w_Rr}, w_Rt={w_Rt}, beta={beta}")

    # quick utility scale check before spending time on FK
    nS_t, nR_t = int(nS_eq), 1
    comp_t = max(1 - (nS_t + nR_t) / K, 0)
    U_t = -w_T - w_N * (nS_t + nR_t) / K - w_Rr * nR_t / K
    print(f"U_running at ({nS_t},{nR_t}) = {U_t:.4f}  "
          f"(β·U over horizon ≈ {beta*U_t*T_HORIZON:.1f})")
    if abs(beta * U_t * T_HORIZON) > 500:
        print("WARNING: |β·U·T| > 500 — possible log-sum-exp underflow. "
              "Consider reducing beta or T_HORIZON.")

    Vgrid = precompute_V_parallel()
    print(f"Vgrid: min={Vgrid.min():.3f}, max={Vgrid.max():.3f}, "
          f"bad={(Vgrid < -1e5).sum()} nodes")

    globals()['V_interp'] = RegularGridInterpolator(
        (GS, GR), Vgrid, bounds_error=False, fill_value=None)

    # ── 2. Optimal dose map ───────────────────────────────────────
    print("\nBuilding dose map ...")
    DS = np.arange(0, K + 1,      25)
    DR = np.arange(0, K // 5 + 1, 10)
    cmap_grid = np.zeros((len(DS), len(DR)))
    for i, ns in enumerate(DS):
        for j, nr in enumerate(DR):
            cmap_grid[i, j] = min(optimal_dose(ns, nr), 10 * micS)

    # ── 3. Dose slices (replaces MSW for weight testing) ──────────
    # c*(nS) at fixed nR values — shows exactly how the controller
    # responds to population size for different resistance burdens.
    nR_slices = [0, 1, 5, 20, 50]
    dose_slices = {}
    nS_scan = np.arange(0, K + 1, 10)
    for nr_val in nR_slices:
        dose_slices[nr_val] = np.array([optimal_dose(ns, nr_val)
                                        for ns in nS_scan])

    # ── 4. MSW CURVE (commented out — uncomment after good results) ──
    # print("(A) MSW curve ...")
    # msw_concs = np.concatenate([[0],
    #              np.logspace(np.log10(0.05*micS), np.log10(5*micS), 18)])
    # N_MSW = 150
    # p_surv = []
    # for c in msw_concs:
    #     rng = np.random.default_rng(7)
    #     surv = sum(controlled_run(int(nS_eq), 1, make_constant(c), rng)[2][-1] > 0
    #                for _ in range(N_MSW))
    #     p_surv.append(surv / N_MSW)
    # p_surv = np.array(p_surv)

    # ── 5. Strategy comparison ────────────────────────────────────
    print("\n(B) Running strategies ...")
    strategies = {
        "Uncontrolled (c=0)":  rule_uncontrolled,
        "Optimal FK feedback": rule_optimal,
        # uncomment below after fixing controller:
        # f"Constant ({4}xMIC)":        make_constant(4 * micS),
        # "Interval-optimal (q8h)":  make_interval_optimal(8.0),
    }

    common_ts  = np.arange(0, T_HORIZON + 0.2, 0.2)
    strat_results = {}

    for name, rule in strategies.items():
        rng    = np.random.default_rng(99)
        Js, Texits = [], []
        all_Ss = np.zeros((N_CMP, len(common_ts)))
        all_Rs = np.zeros((N_CMP, len(common_ts)))
        all_Cs = np.zeros((N_CMP, len(common_ts)))

        for kk in range(N_CMP):
            ts, Ss, Rs, Cs, J, Te = controlled_run(
                nS_Start, nR_Start, rule, rng)
            Js.append(J); Texits.append(Te)
            all_Ss[kk] = np.interp(common_ts, ts, Ss, right=0.0)
            all_Rs[kk] = np.interp(common_ts, ts, Rs, right=0.0)
            all_Cs[kk] = np.interp(common_ts, ts, Cs, right=0.0)

        strat_results[name] = {
            "J_mean":      np.mean(Js),
            "J_std":       np.std(Js),
            "Texit_mean":  np.mean(Texits),
            "avg_Ss":      np.mean(all_Ss, axis=0),
            "std_Ss":      np.std(all_Ss,  axis=0),
            "avg_Rs":      np.mean(all_Rs, axis=0),
            "std_Rs":      np.std(all_Rs,  axis=0),
            "avg_Cs":      np.mean(all_Cs, axis=0),
        }
        p_clear = np.mean(np.array(Texits) < T_HORIZON)
        print(f"  {name:24s}  J={np.mean(Js):8.3f} ± {np.std(Js):6.3f}"
              f"  T_exit={np.mean(Texits):5.1f}h  P(clear)={p_clear:.2f}")

    # ── 6. PLOTS ──────────────────────────────────────────────────
    print("\nPlotting ...")
    fig = plt.figure(figsize=(17, 11))
    gs  = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.30)
    fig.suptitle(
        f"FK Optimal Control — w_T={w_T}  w_N={w_N}  w_Rr={w_Rr}  "
        f"w_Rt={w_Rt}  β={beta}\n"
        f"micS={micS:.2f}  micR={micR:.1f}  nS_eq={nS_eq:.0f}  "
        f"N_immune={N_immune}  T_H={T_HORIZON}h  c_base=micS",
        fontsize=11, fontweight='bold', y=0.99)

    # Panel (0,0): Value function
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(Vgrid.T, origin='lower', aspect='auto',
                   extent=[GS[0], GS[-1], GR[0], GR[-1]], cmap='viridis')
    ax.set_title("Value function V(nS, nR)")
    ax.set_xlabel("nS"); ax.set_ylabel("nR")
    plt.colorbar(im, ax=ax)

    # Panel (0,1): Optimal dose map
    ax = fig.add_subplot(gs[0, 1])
    im = ax.imshow(cmap_grid.T, origin='lower', aspect='auto',
                   extent=[DS[0], DS[-1], DR[0], DR[-1]], cmap='inferno')
    ax.axvline(nS_eq, color='cyan', ls='--', lw=1, label=f'nS_eq={nS_eq:.0f}')
    ax.set_title("Optimal dose  c*(nS, nR)  [mg/L]")
    ax.set_xlabel("nS"); ax.set_ylabel("nR")
    ax.legend(fontsize=7); plt.colorbar(im, ax=ax)

    # Panel (0,2): Dose slices — c*(nS) at fixed nR values
    # (replaces MSW; shows controller logic directly)
    ax = fig.add_subplot(gs[0, 2])
    colors_sl = plt.cm.plasma(np.linspace(0.1, 0.9, len(nR_slices)))
    for nr_val, col in zip(nR_slices, colors_sl):
        ax.plot(nS_scan, dose_slices[nr_val], lw=1.6,
                label=f"nR={nr_val}", color=col)
    ax.axvline(nS_eq, color='gray', ls=':', lw=1, label=f'nS_eq')
    ax.axhline(micS,  color='blue', ls='--', lw=1, label=f'micS')
    ax.axhline(0,     color='black', lw=0.6)
    ax.set_title("Optimal dose  c*(nS)  at fixed nR\n"
                 "(controller prescription vs population size)")
    ax.set_xlabel("nS"); ax.set_ylabel("c* (mg/L)")
    ax.legend(fontsize=7); ax.grid(alpha=0.25)
    ax.set_ylim(bottom=0)

    # Panels (1,0) and (1,1): trajectories per strategy
    for col, (name, res) in enumerate(strat_results.items()):
        ax = fig.add_subplot(gs[1, col])
        t_g = common_ts
        aS, sS = res["avg_Ss"], res["std_Ss"]
        aR, sR = res["avg_Rs"], res["std_Rs"]
        aC      = res["avg_Cs"]

        ax.fill_between(t_g, np.maximum(0, aS - sS), aS + sS,
                        color='#2980b9', alpha=0.15)
        ax.fill_between(t_g, np.maximum(0, aR - sR), aR + sR,
                        color='#e74c3c', alpha=0.15)
        ax.plot(t_g, aS, color='#2980b9', lw=2.0, label='Mean nS')
        ax.plot(t_g, aR, color='#e74c3c', lw=2.0, label='Mean nR')
        ax.axhline(N_immune, color='green', ls=':', lw=1.2,
                   label=f'N_immune={N_immune}')
        ax.set_title(f"{name}\n"
                     f"J={res['J_mean']:.3f} ± {res['J_std']:.3f},"
                     f"  T_exit={res['Texit_mean']:.1f}h",
                     fontsize=9)
        ax.set_xlabel("time (h)"); ax.set_ylabel("cells")
        ax.legend(fontsize=7, loc='upper right'); ax.grid(alpha=0.2)
        ax2 = ax.twinx()
        ax2.plot(t_g, aC, color='gray', lw=1.0, alpha=0.7, ls='--')
        ax2.set_ylabel("avg c (mg/L)", color='gray', fontsize=8)
        ax2.tick_params(axis='y', labelcolor='gray', labelsize=7)

    # Panel (1,2): objective comparison bar
    ax = fig.add_subplot(gs[1, 2])
    names = list(strat_results.keys())
    Js    = [strat_results[n]["J_mean"] for n in names]
    Jerr  = [strat_results[n]["J_std"]  for n in names]
    bars  = ax.barh(range(len(names)), Js, xerr=Jerr,
                    color=['#7f8c8d', '#27ae60'], capsize=4)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([n.replace(' ', '\n', 1) for n in names], fontsize=8)
    ax.set_xlabel("Objective J  (lower = better)")
    ax.set_title("Strategy comparison\n(mean ± std, 60 trajectories)")
    ax.grid(alpha=0.3, axis='x')
    for bar, j in zip(bars, Js):
        ax.text(j * 0.98, bar.get_y() + bar.get_height() / 2,
                f"{j:.3f}", va='center', ha='right',
                fontsize=7, color='white', fontweight='bold')

    # ── 7. Save (sequential numbering) ───────────────────────────
    out_dir = "figures"
    os.makedirs(out_dir, exist_ok=True)
    prefix  = "fk_weight_"
    nums    = []
    for fname in os.listdir(out_dir):
        if fname.startswith(prefix) and fname.endswith(".png"):
            try: nums.append(int(fname[len(prefix):-4]))
            except ValueError: pass
    next_idx = max(nums) + 1 if nums else 1
    out_path = os.path.join(out_dir, f"{prefix}{next_idx:03d}.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved → {out_path}")

    print("\n" + "=" * 62)
    print(f"{'Strategy':<24} {'J':>9} {'±std':>7}  {'T_exit':>7}h")
    print("-" * 62)
    for name, res in strat_results.items():
        print(f"{name:<24} {res['J_mean']:>9.3f} "
              f"{res['J_std']:>7.3f}  {res['Texit_mean']:>7.1f}")
    print("\nWhat to look for:")
    print("  1. Value-function panel: GRADIENT present?  (flat = bad)")
    print("  2. Dose-slice panel:     c*(nS) rises with nS? (makes sense)")
    print("  3. Optimal J < Uncontrolled J?  (controller actually helping?)")
    print("  4. Optimal trajectory: S actually declines?  (not pinned at 756?)")