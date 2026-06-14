
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
import os

# ════════════════════════════════════════════════════════════════════
# PARAMETERS  (E. coli + meropenem, hours)
# ════════════════════════════════════════════════════════════════════
k1 = 2.5
k3 = 2.25        # 0.679
k4 = 0.5
K     = 2000
gamma = k1

k5_per_div = 7.8e-7      # baseline per-division S->R mutation probability
F_max      = 3.0         # max sub-MIC stress fold-increase (Gutierrez 2013)

psi_S_max = k1 - k4      # 0.554
psi_R_max = k3 - k4      # 0.500
kmax      = 0.672
psiminS   = np.log(10)*(-6.5) * 24.;  # -0.118
psiminR   = psiminS*24
kappa     = 1.1
EC50      = 1.01
micS      = 0.017
MIC_FACTOR = 24
micR      = MIC_FACTOR * micS

nS_eq = K * (k1 - k4) / k1

nS_Start = 2
nR_Start = 1
w_T  = 1.0
w_N  = 1.0
w_Rr = 100.0
w_Rt = 1000.0
beta = 1.0
N_immune = 0

c_base = 0.00 # FK baseline reference dose (FIX 2)
N_SIMS = 200
out_dir = "figures"

T_HORIZON = 100.0

PK_halflife = 1.0
PK_lambda   = np.log(2) / PK_halflife

V_interp = None

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

def net_growth_S(c): return psi_S_max - alpha_S(c)
def net_growth_R(c): return psi_R_max - alpha_R(c)

_span = psi_S_max - psiminS
_D    = psiminS / psi_S_max
def invert_alpha_S(alpha):
    if alpha <= 0:     return 0.0
    if alpha >= _span: return np.inf
    u = alpha * _D / (alpha - _span)
    if u <= 0: return 0.0
    return micS * u ** (1.0 / kappa)

# ════════════════════════════════════════════════════════════════════
# MUTATION (FIX 1)
# ════════════════════════════════════════════════════════════════════
def stress_factor(c):
    """=1 at c=0, peaks ~F_max near c=micS, falls at high c."""
    if c <= 0:
        return 1.0
    x = c / micS
    shape = x / (1.0 + x * x)          # peaks (=0.5) at x=1
    return 1.0 + (F_max - 1.0) * (shape / 0.5)

def mutation_propensity(nS, nR, c, comp):
    birth_S = k1 * nS * comp
    return k5_per_div * stress_factor(c) * birth_S

# ════════════════════════════════════════════════════════════════════
# UTILITY
# ════════════════════════════════════════════════════════════════════
def U_running(nS, nR):  return -w_T - w_N * (nS + nR)/K - w_Rr * nR/K
def U_exit(nS, nR):     return -w_Rt * nR / K   
def is_absorbing(nS, nR): return (nS + nR) <= N_immune

# ════════════════════════════════════════════════════════════════════
# FK BASELINE SSA (FIX 2)
# ════════════════════════════════════════════════════════════════════
_aS_base = alpha_S(c_base)
_aR_base = alpha_R(c_base)


# ════════════════════════════════════════════════════════════════════
# FK GRID
# ════════════════════════════════════════════════════════════════════
GS = np.arange(0, 1001, 50)
GR = np.arange(0, 201, 10)
n_fk = 200


# ════════════════════════════════════════════════════════════════════
# OPTIMAL DOSE
# ════════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════════
# CONTROLLED SSA
# ════════════════════════════════════════════════════════════════════
def controlled_run(nS0, nR0, dosing_rule, rng, Tmax=T_HORIZON, record_dt=0.2):
    nS, nR, t = float(nS0), float(nR0), 0.0
    J = 0.0
    T_exit = None
    ts, Ss, Rs, Cs = [0.0], [nS], [nR], [dosing_rule(nS, nR, 0.0)]
    next_rec = record_dt
    while t < Tmax:
        if is_absorbing(nS, nR) and T_exit is None:
            T_exit = t
            J += w_Rt * nR / K
            break
        c = dosing_rule(nS, nR, t)
        aS = alpha_S(c); aR = alpha_R(c)
        comp = max(1.0 - (nS + nR) / K, 0.0)
        a0 = k1 * nS * comp
        a1 = nS * (k4 + aS)
        a2 = k3 * nR * comp
        a3 = nR * (k4 + aR)
        a4 = mutation_propensity(nS, nR, c, comp)
        a = a0 + a1 + a2 + a3 + a4
        if a <= 0: break
        dt = -np.log(rng.random()) / a
        J_inc = (w_T + w_N * (nS + nR)/K + w_Rr * nR/K) * dt
        if  np.isfinite(J_inc):
            J += J_inc
        else: 
            print(f"NaN J at nS={nS}, nR={nR}, c={c}, dt={dt}")
            break
        t += dt
        u = rng.random() * a
        if   u < a0:                 nS += 1
        elif u < a0 + a1:            nS -= 1
        elif u < a0 + a1 + a2:       nR += 1
        elif u < a0 + a1 + a2 + a3:  nR -= 1
        else:                        nS -= 1; nR += 1
        nS = max(0.0, nS); nR = max(0.0, nR)
        while next_rec <= t and next_rec <= Tmax:
            ts.append(next_rec); Ss.append(nS); Rs.append(nR)
            Cs.append(dosing_rule(nS, nR, next_rec))
            next_rec += record_dt
    if T_exit is None:
        T_exit = Tmax
        J += w_Rt * nR / K 
    return (np.array(ts), np.array(Ss), np.array(Rs), np.array(Cs), J, T_exit)

# ════════════════════════════════════════════════════════════════════
# DOSING RULES  (extension point)
# ════════════════════════════════════════════════════════════════════
def rule_uncontrolled(nS, nR, t):
    return c_base

# FUTURE (unused now):
def make_constant(c):
    def rule(nS, nR, t): return c
    return rule


rng = np.random.default_rng(99)

# Create a uniform time grid for averaging
record_dt = 0.2
common_ts = np.arange(0, T_HORIZON + record_dt, record_dt)

# Matrices to store interpolated trajectories
all_Ss = np.zeros((N_SIMS, len(common_ts)))
all_Rs = np.zeros((N_SIMS, len(common_ts)))
all_Cs = np.zeros((N_SIMS, len(common_ts)))

total_J = 0.0
total_Te = 0.0

for i in range(N_SIMS):
    nS0, nR0 = nS_Start, nR_Start
    ts, Ss, Rs, Cs, J, Te = controlled_run(nS0, nR0, rule_uncontrolled, rng, Tmax=T_HORIZON, record_dt=record_dt)
    print(f"  Run {i+1}/{N_SIMS} complete: J={J:.2f}, T_exit={Te:.2f}h")
    # Interpolate this run onto the common time grid.
    # right=0.0 means if the run exited early (cleared), bacteria count stays at 0.
    all_Ss[i] = np.interp(common_ts, ts, Ss, right=0.0)
    all_Rs[i] = np.interp(common_ts, ts, Rs, right=0.0)
    
    # For the uncontrolled case, drug stays constant, but we interpolate just in case
    all_Cs[i] = np.interp(common_ts, ts, Cs, right=c_base) 
    
    total_J += J
    total_Te += Te

# Compute macroscopic statistics
avg_Ss = np.mean(all_Ss, axis=0)
avg_Rs = np.mean(all_Rs, axis=0)
std_Ss = np.std(all_Ss, axis=0)
std_Rs = np.std(all_Rs, axis=0)
avg_Cs = np.mean(all_Cs, axis=0)

avg_J = total_J / N_SIMS
avg_Te = total_Te / N_SIMS
print(f"  Ensemble run complete: Avg J={avg_J:.2f}, Avg T_exit={avg_Te:.2f}h")

# Plotting the ensemble average dynamics
print("\nPlotting ensemble average dynamics...")
fig, ax = plt.subplots(figsize=(10, 5))

# Plot standard deviations as shaded regions to visualize variance
ax.fill_between(common_ts, np.maximum(0, avg_Ss - std_Ss), avg_Ss + std_Ss, color='#2980b9', alpha=0.2)
ax.fill_between(common_ts, np.maximum(0, avg_Rs - std_Rs), avg_Rs + std_Rs, color='#e74c3c', alpha=0.2)

# Plot the mean trajectories
ax.plot(common_ts, avg_Ss, color='#2980b9', lw=2, label='Expected nS')
ax.plot(common_ts, avg_Rs, color='#e74c3c', lw=2, label='Expected nR')

ax.axhline(N_immune, color='green', ls=':', lw=1.5, label='N_immune')
ax.set_xlabel('time (h)')
ax.set_ylabel('cells')
ax.set_title(f'Ensemble Average Uncontrolled ns = 1  nr =  1 ({N_SIMS} runs) — Avg J={avg_J:.1f}, Avg T_exit={avg_Te:.1f}h')
ax.legend(fontsize=9, loc='upper right')
ax.grid(alpha=0.3)

# Overlay the drug concentration
ax2 = ax.twinx()
ax2.plot(common_ts, avg_Cs, color='gray', lw=1.5, alpha=0.8, ls='--')
ax2.set_ylabel('c (mg/L)', color='gray')

# Ensure output directory exists and save sequentially
os.makedirs(out_dir, exist_ok=True)
existing = [f for f in os.listdir(out_dir) if f.startswith("fk_ensemble_") and f.endswith(".png")]
nums = []
for fname in existing:
    try:
        num = int(fname.rsplit('_', 1)[-1].split('.png')[0])
        nums.append(num)
    except Exception:
        continue
next_idx = max(nums) + 1 if nums else 1
out_name = f"fk_ensemble_{next_idx:03d}.png"
out_path = os.path.join(out_dir, out_name)
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved {out_path}")
