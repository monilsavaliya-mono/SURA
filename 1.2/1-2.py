import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

# ── parameters ────────────────────────────────────
s_max = 20
r_max = 40
k1    = 0.8
k2    = 1.5
k3    = 0.4
k4    = 0.15
k5    = 0.002
beta  = 0.10

ns0    = 10
nr0    = 1
Tmax   = 40.0
N_RUNS = 500

np.random.seed(42)

# ══════════════════════════════════════════════════
# FUNCTIONS
# ══════════════════════════════════════════════════

def ssa_uncontrolled(ns0, nr0, k1, k2, k3, k4, k5, Tmax):
    ns, nr, t = ns0, nr0, 0
    nSs, nRs, ts = [ns], [nr], [t]
    while (ns + nr > 0 and t < Tmax):
        a1 = k1*ns; a2 = k2*ns; a3 = k3*nr
        a4 = k4*nr; a5 = k5*ns
        a0 = a1+a2+a3+a4+a5
        if a0 < 1e-15: break
        tau = -np.log(np.random.random()) / a0
        t += tau
        if t > Tmax: break
        u = np.random.random() * a0
        if   u < a1:          ns += 1
        elif u < a1+a2:       ns -= 1
        elif u < a1+a2+a3:    nr += 1
        elif u < a1+a2+a3+a4: nr -= 1
        else:                 ns -= 1; nr += 1
        ns = max(ns, 0); nr = max(nr, 0)
        nSs.append(ns); nRs.append(nr); ts.append(t)
    return ts, nSs, nRs


def solve_z(s_max, r_max, k1, k2, k3, k4, k5, beta):
    N = (s_max+1) * (r_max+1)
    A = lil_matrix((N, N))
    b = np.zeros(N)

    def idx(ns, nr):
        return ns*(r_max+1) + nr

    def is_boundary(ns, nr):
        return (ns == 0 and nr == 0)

    def is_truncation(ns, nr):
        return ns == s_max or nr == r_max

    for ns in range(s_max+1):
        for nr in range(r_max+1):
            i = idx(ns, nr)
            if is_boundary(ns, nr):
                A[i,i] = 1.0; b[i] = 1.0; continue
            if is_truncation(ns, nr):
                A[i,i] = 1.0; b[i] = 1e-6; continue

            D = beta + (k1+k2+k5)*ns + (k3+k4)*nr
            A[i,i] = D

            def add(ns2, nr2, coef):
                if 0 <= ns2 <= s_max and 0 <= nr2 <= r_max:
                    if is_boundary(ns2, nr2):
                        b[i] += coef * 1.0
                    elif not is_truncation(ns2, nr2):
                        A[i, idx(ns2, nr2)] -= coef

            add(ns+1, nr,   k1*ns)
            add(ns-1, nr,   k2*ns)
            add(ns,   nr+1, k3*nr)
            add(ns,   nr-1, k4*nr)
            add(ns-1, nr+1, k5*ns)

    z_flat = spsolve(A.tocsr(), b)
    if not np.all(np.isfinite(z_flat)) or np.any(z_flat <= 0):
        print("Z solve failed — try smaller beta or larger grid")
        return None
    return z_flat.reshape(s_max+1, r_max+1)


def ssa_opt(ns0, nr0, s_max, r_max, k1, k2, k3, k4, k5, Z, Tmax):
    def Zl(ns, nr):
        return max(Z[max(0,min(ns,s_max)), max(0,min(nr,r_max))], 1e-300)

    ns, nr, t = ns0, nr0, 0.0
    nSs, nRs, ts = [ns], [nr], [t]
    while (ns + nr > 0 and t < Tmax):
        zn = Zl(ns, nr)
        k1t = k1
        k2t = k2 * Zl(ns-1, nr)   / zn
        k3t = k3 
        k4t = k4 
        k5t = k5 
        a1 = k1t*ns; a2 = k2t*ns; a3 = k3t*nr
        a4 = k4t*nr; a5 = k5t*ns
        a0 = a1+a2+a3+a4+a5
        if a0 < 1e-15: break
        tau = -np.log(np.random.random()) / a0
        t += tau
        if t > Tmax: break
        u = np.random.random() * a0
        if   u < a1:          ns += 1
        elif u < a1+a2:       ns -= 1
        elif u < a1+a2+a3:    nr += 1
        elif u < a1+a2+a3+a4: nr -= 1
        else:                 ns -= 1; nr += 1
        ns = max(ns, 0); nr = max(nr, 0)
        nSs.append(ns); nRs.append(nr); ts.append(t)
    return ts, nSs, nRs


# ══════════════════════════════════════════════════
# STEP 1 — Solve Z
# ══════════════════════════════════════════════════
print("Solving Z...")
Z = solve_z(s_max, r_max, k1, k2, k3, k4, k5, beta)
if Z is None:
    exit()
print(f"Z solved. Shape={Z.shape}  Z=[{Z.min():.6f}, {Z.max():.4f}]")

def Z_safe(ns, nr):
    return max(Z[max(0,min(ns,s_max)), max(0,min(nr,r_max))], 1e-300)


# ══════════════════════════════════════════════════
# STEP 2 — Run simulations
# ══════════════════════════════════════════════════
print(f"\nRunning {N_RUNS} uncontrolled runs...")
unc_ts, unc_nS, unc_nR, unc_sum = [], [], [], []

for i in range(N_RUNS):
    ts, nSs, nRs = ssa_uncontrolled(ns0, nr0, k1, k2, k3, k4, k5, Tmax)
    unc_ts.append(ts); unc_nS.append(nSs); unc_nR.append(nRs)
    unc_sum.append({
        "final_t":   ts[-1],   "final_nS": nSs[-1], "final_nR": nRs[-1],
        "max_nR":    max(nRs), "max_nS":   max(nSs),
        "eradicated": int(nSs[-1]==0 and nRs[-1]==0),
        "resistance": int(max(nRs) > 5),
        "steps":      len(ts)-1,
    })
    if (i+1) % 100 == 0: print(f"  unc {i+1}/{N_RUNS}")

print(f"\nRunning {N_RUNS} optimal runs...")
opt_ts, opt_nS, opt_nR, opt_sum = [], [], [], []

for i in range(N_RUNS):
    ts, nSs, nRs = ssa_opt(ns0, nr0, s_max, r_max, k1, k2, k3, k4, k5, Z, Tmax)
    kl_total = 0.0
    for ns, nr in zip(nSs, nRs):
        zn  = Z_safe(ns, nr)
        k2t = k2 * Z_safe(ns-1, nr) / zn
        if k2t > 1e-15:
            ratio = k2t / k2
            kl_total += ns * (k2t * np.log(ratio) - k2t + k2)
    opt_ts.append(ts); opt_nS.append(nSs); opt_nR.append(nRs)
    opt_sum.append({
        "final_t":    ts[-1],   "final_nS": nSs[-1], "final_nR": nRs[-1],
        "max_nR":     max(nRs), "max_nS":   max(nSs),
        "eradicated": int(nSs[-1]==0 and nRs[-1]==0),
        "resistance": int(max(nRs) > 5),
        "steps":      len(ts)-1,
        "kl_cost":    round(kl_total, 6),
    })
    if (i+1) % 100 == 0: print(f"  opt {i+1}/{N_RUNS}")


# ══════════════════════════════════════════════════
# STEP 3 — Interpolate onto common grid
# ══════════════════════════════════════════════════
t_grid = np.linspace(0, Tmax, 300)

def interp(ts, vals):
    return np.interp(t_grid, ts, vals, left=vals[0], right=vals[-1])

unc_nS_mean = np.array([interp(unc_ts[i], unc_nS[i]) for i in range(N_RUNS)]).mean(0)
unc_nR_mean = np.array([interp(unc_ts[i], unc_nR[i]) for i in range(N_RUNS)]).mean(0)
unc_nS_std  = np.array([interp(unc_ts[i], unc_nS[i]) for i in range(N_RUNS)]).std(0)
unc_nR_std  = np.array([interp(unc_ts[i], unc_nR[i]) for i in range(N_RUNS)]).std(0)

opt_nS_mean = np.array([interp(opt_ts[i], opt_nS[i]) for i in range(N_RUNS)]).mean(0)
opt_nR_mean = np.array([interp(opt_ts[i], opt_nR[i]) for i in range(N_RUNS)]).mean(0)
opt_nS_std  = np.array([interp(opt_ts[i], opt_nS[i]) for i in range(N_RUNS)]).std(0)
opt_nR_std  = np.array([interp(opt_ts[i], opt_nR[i]) for i in range(N_RUNS)]).std(0)

unc_maxnR = [s["max_nR"]    for s in unc_sum]
opt_maxnR = [s["max_nR"]    for s in opt_sum]
unc_dur   = [s["final_t"]   for s in unc_sum]
opt_dur   = [s["final_t"]   for s in opt_sum]
unc_steps = [s["steps"]     for s in unc_sum]
opt_steps = [s["steps"]     for s in opt_sum]
opt_kl    = [s["kl_cost"]   for s in opt_sum]
u_erad    = sum(s["eradicated"] for s in unc_sum)
o_erad    = sum(s["eradicated"] for s in opt_sum)
u_res     = sum(s["resistance"] for s in unc_sum)
o_res     = sum(s["resistance"] for s in opt_sum)


# ══════════════════════════════════════════════════
# STEP 4 — Plots
# ══════════════════════════════════════════════════
BG     = '#0f1117'; PANEL  = '#1e293b'; BORDER = '#334155'
TEXT   = '#e2e8f0'; MUTED  = '#94a3b8'; C_GRAY = '#64748b'
C_BLUE = '#4361ee'; C_RED  = '#ef4444'
C_GREEN= '#10b981'; C_AMBER= '#f59e0b'

def style(ax, title, xl, yl):
    ax.set_facecolor(PANEL)
    ax.set_title(title, fontsize=11, fontweight='bold', color=TEXT, pad=8)
    ax.set_xlabel(xl, fontsize=9, color=MUTED)
    ax.set_ylabel(yl, fontsize=9, color=MUTED)
    ax.tick_params(colors=C_GRAY, labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor(BORDER)
    ax.grid(alpha=0.15, color=BORDER)

# ── page 1 ─────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.patch.set_facecolor(BG)
fig.suptitle(
    f'Population Dynamics — {N_RUNS} runs each\n'
    f'k1={k1} k2={k2} k3={k3} k4={k4} k5={k5} β={beta}  '
    f'ns0={ns0} nr0={nr0}  s_max={s_max} r_max={r_max}',
    fontsize=11, fontweight='bold', color=TEXT, y=1.01)

ax = axes[0,0]
ax.plot(t_grid, unc_nS_mean, color=C_GRAY, lw=2, label='Uncontrolled')
ax.fill_between(t_grid, unc_nS_mean-unc_nS_std, unc_nS_mean+unc_nS_std,
                alpha=0.18, color=C_GRAY)
ax.plot(t_grid, opt_nS_mean, color=C_BLUE, lw=2, label='Optimal')
ax.fill_between(t_grid, opt_nS_mean-opt_nS_std, opt_nS_mean+opt_nS_std,
                alpha=0.18, color=C_BLUE)
ax.legend(fontsize=9, facecolor=PANEL, labelcolor=TEXT, edgecolor=BORDER)
style(ax, f'Mean nS ±1std [{N_RUNS} runs]', 'time (h)', 'nS')

ax = axes[0,1]
ax.plot(t_grid, np.maximum(unc_nR_mean, 0.1), color=C_RED,   lw=2, label='Uncontrolled')
ax.plot(t_grid, np.maximum(opt_nR_mean, 0.1), color=C_GREEN, lw=2, label='Optimal')
ax.set_yscale('log')
ax.legend(fontsize=9, facecolor=PANEL, labelcolor=TEXT, edgecolor=BORDER)
style(ax, f'Mean nR log scale [{N_RUNS} runs]', 'time (h)', 'nR (log)')

ax = axes[1,0]
for i in range(N_RUNS):
    ax.plot(unc_ts[i], unc_nS[i], color=C_GRAY,  alpha=0.15, lw=0.6)
for i in range(N_RUNS):
    ax.plot(opt_ts[i], opt_nS[i], color=C_BLUE,  alpha=0.15, lw=0.6)
ax.plot([], [], color=C_GRAY,  label='Uncontrolled')
ax.plot([], [], color=C_BLUE,  label='Optimal')
ax.legend(fontsize=9, facecolor=PANEL, labelcolor=TEXT, edgecolor=BORDER)
style(ax, f'All nS trajectories', 'time (h)', 'nS')

ax = axes[1,1]
for i in range(N_RUNS):
    ax.plot(unc_ts[i], [max(v,0.1) for v in unc_nR[i]], color=C_RED,   alpha=0.15, lw=0.6)
for i in range(N_RUNS):
    ax.plot(opt_ts[i], [max(v,0.1) for v in opt_nR[i]], color=C_GREEN, alpha=0.15, lw=0.6)
ax.set_yscale('log')
ax.plot([], [], color=C_RED,   label='Uncontrolled')
ax.plot([], [], color=C_GREEN, label='Optimal')
ax.legend(fontsize=9, facecolor=PANEL, labelcolor=TEXT, edgecolor=BORDER)
style(ax, f'All nR trajectories (log)', 'time (h)', 'nR (log)')

plt.tight_layout(pad=2.5)
fig.savefig('page1_dynamics.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close(); print("Saved: page1_dynamics.png")

# ── page 2 ─────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.patch.set_facecolor(BG)
fig.suptitle(f'Outcome Distributions — {N_RUNS} runs',
             fontsize=12, fontweight='bold', color=TEXT, y=1.01)
bins = 20

ax = axes[0,0]
ax.hist(unc_maxnR, bins=bins, color=C_RED,   alpha=0.7, label='Uncontrolled', edgecolor=BORDER)
ax.hist(opt_maxnR, bins=bins, color=C_GREEN, alpha=0.7, label='Optimal',      edgecolor=BORDER)
ax.axvline(np.mean(unc_maxnR), color=C_RED,   ls='--', lw=1.5, label=f'unc={np.mean(unc_maxnR):.1f}')
ax.axvline(np.mean(opt_maxnR), color=C_GREEN, ls='--', lw=1.5, label=f'opt={np.mean(opt_maxnR):.1f}')
ax.legend(fontsize=8, facecolor=PANEL, labelcolor=TEXT, edgecolor=BORDER)
style(ax, 'max nR distribution', 'max nR', 'count')

ax = axes[0,1]
ax.hist(unc_dur, bins=bins, color=C_GRAY, alpha=0.7, label='Uncontrolled', edgecolor=BORDER)
ax.hist(opt_dur, bins=bins, color=C_BLUE, alpha=0.7, label='Optimal',      edgecolor=BORDER)
ax.axvline(np.mean(unc_dur), color=C_GRAY, ls='--', lw=1.5)
ax.axvline(np.mean(opt_dur), color=C_BLUE, ls='--', lw=1.5)
ax.legend(fontsize=8, facecolor=PANEL, labelcolor=TEXT, edgecolor=BORDER)
style(ax, 'Duration distribution', 'duration (h)', 'count')

ax = axes[1,0]
ax.hist(unc_steps, bins=bins, color=C_GRAY, alpha=0.7, label='Uncontrolled', edgecolor=BORDER)
ax.hist(opt_steps, bins=bins, color=C_BLUE, alpha=0.7, label='Optimal',      edgecolor=BORDER)
ax.legend(fontsize=8, facecolor=PANEL, labelcolor=TEXT, edgecolor=BORDER)
style(ax, 'Total SSA steps', 'steps', 'count')

ax = axes[1,1]
ax.hist(opt_kl, bins=bins, color=C_AMBER, alpha=0.8, edgecolor=BORDER)
ax.axvline(np.mean(opt_kl), color='white', ls='--', lw=1.5,
           label=f'mean={np.mean(opt_kl):.3f}')
ax.legend(fontsize=9, facecolor=PANEL, labelcolor=TEXT, edgecolor=BORDER)
style(ax, 'KL cost (optimal only)', 'KL cost', 'count')

plt.tight_layout(pad=2.5)
fig.savefig('page2_distributions.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close(); print("Saved: page2_distributions.png")

# ── page 3 ─────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 6))
fig.patch.set_facecolor(BG)
fig.suptitle('Summary Comparison — Uncontrolled vs Optimal',
             fontsize=12, fontweight='bold', color=TEXT, y=1.01)

def bar2(ax, u, o, title, yl, fmt='.2f'):
    bars = ax.bar(['Uncontrolled','Optimal'], [u,o],
                  color=[C_RED,C_GREEN], edgecolor=BORDER, width=0.5)
    for bar, val in zip(bars,[u,o]):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.02,
                f'{val:{fmt}}', ha='center', va='bottom',
                color=TEXT, fontsize=11, fontweight='bold', fontfamily='monospace')
    style(ax, title, '', yl)
    ax.set_xticklabels(['Uncontrolled','Optimal'], color=TEXT, fontsize=10)

bar2(axes[0], np.mean(unc_maxnR), np.mean(opt_maxnR), 'Mean max nR',  'mean max nR', '.1f')
bar2(axes[1], u_res,  o_res,  f'Resistance emerged\n(/{N_RUNS})', 'count', 'd')
bar2(axes[2], u_erad, o_erad, f'Eradicated\n(/{N_RUNS})',         'count', 'd')

plt.tight_layout(pad=2.5)
fig.savefig('page3_comparison.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close(); print("Saved: page3_comparison.png")

# ── page 4 — text summary ──────────────────────
fig, ax = plt.subplots(figsize=(10, 8))
fig.patch.set_facecolor(BG); ax.set_facecolor(PANEL); ax.axis('off')

lines = [
  ("SIMULATION RESULTS SUMMARY",                                          TEXT,  14, True),
  ("",                                                                     TEXT,  10, False),
  (f"Parameters:  k1={k1}  k2={k2}  k3={k3}  k4={k4}  k5={k5}  β={beta}", MUTED, 10, False),
  (f"Initial:     ns0={ns0}  nr0={nr0}  Tmax={Tmax}h  N_runs={N_RUNS}",    MUTED, 10, False),
  (f"Grid:        s_max={s_max}  r_max={r_max}  seed=42",                  MUTED, 10, False),
  ("",                                                                     TEXT,  10, False),
  ("─"*55,                                                                 BORDER, 9, False),
  (f"{'Metric':<32} {'Uncontrolled':>12} {'Optimal':>12}",                 MUTED, 10, False),
  ("─"*55,                                                                 BORDER, 9, False),
  (f"{'Eradicated runs':<32} {u_erad:>12} {o_erad:>12}",                   TEXT,  10, False),
  (f"{'Resistance emerged':<32} {u_res:>12} {o_res:>12}",                  TEXT,  10, False),
  (f"{'Mean max nR':<32} {np.mean(unc_maxnR):>12.2f} {np.mean(opt_maxnR):>12.2f}", TEXT, 10, False),
  (f"{'Std max nR':<32} {np.std(unc_maxnR):>12.2f} {np.std(opt_maxnR):>12.2f}",    TEXT, 10, False),
  (f"{'Median max nR':<32} {np.median(unc_maxnR):>12.2f} {np.median(opt_maxnR):>12.2f}", TEXT, 10, False),
  (f"{'Mean duration (h)':<32} {np.mean(unc_dur):>12.3f} {np.mean(opt_dur):>12.3f}",     TEXT, 10, False),
  (f"{'Mean total steps':<32} {np.mean(unc_steps):>12.1f} {np.mean(opt_steps):>12.1f}",  TEXT, 10, False),
  (f"{'Mean KL cost':<32} {'0.0':>12} {np.mean(opt_kl):>12.4f}",           TEXT,  10, False),
  ("─"*55,                                                                 BORDER, 9, False),
  ("",                                                                     TEXT,  10, False),
  (f"  S net = k1-k2 = {k1}-{k2} = {round(k1-k2,3)}  ({'grows' if k1>k2 else 'decays'} without control)", C_AMBER, 10, False),
  (f"  R net = k3-k4 = {k3}-{k4} = {round(k3-k4,3)}  ({'grows' if k3>k4 else 'decays'} always)",          C_RED if k3>k4 else C_GREEN, 10, False),
]

y = 0.97
for txt, col, sz, bold in lines:
    ax.text(0.04, y, txt, transform=ax.transAxes, fontsize=sz, color=col,
            fontfamily='monospace', fontweight='bold' if bold else 'normal', va='top')
    y -= 0.048
for sp in ax.spines.values(): sp.set_edgecolor(BORDER); sp.set_visible(True)

fig.savefig('page4_summary.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close(); print("Saved: page4_summary.png")

# ══════════════════════════════════════════════════
# STEP 5 — Terminal summary
# ══════════════════════════════════════════════════
print("\n" + "═"*52)
print(f"  RESULTS  ({N_RUNS} runs, ns0={ns0} nr0={nr0})")
print("═"*52)
print(f"  {'Metric':<28} {'Unc':>10} {'Opt':>10}")
print(f"  {'-'*48}")
print(f"  {'Eradicated':<28} {u_erad:>10} {o_erad:>10}")
print(f"  {'Resistance emerged':<28} {u_res:>10} {o_res:>10}")
print(f"  {'Mean max nR':<28} {np.mean(unc_maxnR):>10.2f} {np.mean(opt_maxnR):>10.2f}")
print(f"  {'Mean duration (h)':<28} {np.mean(unc_dur):>10.3f} {np.mean(opt_dur):>10.3f}")
print(f"  {'Mean KL cost':<28} {'0.0':>10} {np.mean(opt_kl):>10.4f}")
print("═"*52)