import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import csv, os

k1   = 0.733
k3   = 0.733 * 0.927
k4   = 0.179
k5   = 7.8e-7
kmax = 0.672
EC50 = 1.01
C0   = 30.0
lam  = 0.693
dose_interval = 8

t_last = 0.0
t_next = dose_interval
C_peak = C0
C_dose = C0
C_curr = C0
total_drug = C0



def k2_from_c(c):
    if c <= 0:
        return 0.0
    return kmax * c / (c + EC50)


def reset_pk_state():
    global t_last, t_next, C_peak, C_curr, total_drug
    t_last = 0.0
    t_next = dose_interval
    C_peak = C0
    C_curr = C0
    total_drug = C0


def k2_standard_at_t(t):
    global t_last, t_next, C_peak, C_curr, total_drug

    while t >= t_next:
        C_peak = C_peak * np.exp(-lam * dose_interval) + C_dose
        t_last = t_next
        t_next += dose_interval
        total_drug += C_dose

    C_now = C_peak * np.exp(-lam * (t - t_last))
    C_curr = C_now
    return k2_from_c(C_now)

def k_avg(n_cycles=8, n_points=200):
    global t_last, t_next, C_peak, C_curr, total_drug
    reset_pk_state()          # start fresh
    all_k2 = []

    for cycle in range(n_cycles):
        t_start = t_last      # start of this cycle
        t_end   = t_next      # end of this cycle (= next dose time)

        # sample within this cycle WITHOUT triggering the dose
        # compute C directly so while loop doesn't fire
        for t in np.linspace(t_start, t_end - 0.001, n_points):
            C_now = C_peak * np.exp(-lam * (t - t_last))
            all_k2.append(k2_from_c(C_now))

        k2_standard_at_t(t_end)

    reset_pk_state()          # clean up globals for next caller
    return float(np.mean(all_k2))

def ssa_standard(ns0 , nr0 , Tmax , seed , pop_limit = 500):
    global total_drug 
    np.random.seed(seed)
    ns,nr,t = ns0,nr0,0.0
    ts = [0.0]
    nSs = [ns]
    nRr =  [nr]
    Cs = [C_dose]
    total_drug = C_dose
    k2s = [k2_standard_at_t(0.0)]

    while((ns+nr>0) and (ns + nr < pop_limit) and t < Tmax):
        k2 = k2_standard_at_t(t)
        a1 = k1*ns
        a2 = k2*ns
        a3 = k3*nr
        a4 = k4*nr
        a5 = k5*ns
        a6 = k4*ns   #to check if this corrrect like not sure this is used for natuall rate anyway 

        a0 = a1+a2+a3+a4+a5+a6
        if(a0< 1e-15):
            break

        tau = -np.log(np.random.random())/a0
        t+=tau
        if(t>=Tmax):
            break

        u = np.random.random()*a0
        if(u<a1):
            ns+=1
        elif(u<a1+a2):
            ns-=1
        elif(u<a1+a2+a3):
            nr+=1
        elif(u<a1+a2+a3+a4):
            nr-=1
        elif(u<a1+a2+a3+a4+a5):
            ns-=1
            nr+=1
        else:
            ns-=1

        ns = max(ns,0)
        nr = max(nr,0)
        nSs.append(ns)
        nRr.append(nr)
        Cs.append(C_curr)
        k2s.append(k2)
        ts.append(t)
    n_doses = int(total_drug / C0)
    summary = {
        "final_time": t,
        "final_ns": ns,
        "final_nr": nr,
        "n_doses": n_doses,
        "total_drug": total_drug
    }
    return summary,(ts,nSs,nRr,Cs,k2s)



def uncontrolled_ssa(ns0 , nr0 , Tmax , seed , pop_limit = 500):

    np.random.seed(seed)
    ns,nr,t = ns0,nr0,0.0
    ts = [0.0]
    nSs = [ns]
    nRr =  [nr]

    while((ns+nr>0) and (ns + nr < pop_limit) and t < Tmax):
        a1 = k1*ns
        a2 = 0
        a3 = k3*nr
        a4 = k4*nr
        a5 = k5*ns
        a6 = k4*ns   #to check if this corrrect like not sure this is used for natuall rate anyway 

        a0 = a1+a2+a3+a4+a5+a6
        if(a0< 1e-15):
            break

        tau = -np.log(np.random.random())/a0
        t+=tau
        if(t>=Tmax):
            break

        u = np.random.random()*a0
        if(u<a1):
            ns+=1
        elif(u<a1+a2):
            ns-=1
        elif(u<a1+a2+a3):
            nr+=1
        elif(u<a1+a2+a3+a4):
            nr-=1
        elif(u<a1+a2+a3+a4+a5):
            ns-=1
            nr+=1
        else:
            ns-=1

        ns = max(ns,0)
        nr = max(nr,0)
        nSs.append(ns)
        nRr.append(nr)
        ts.append(t)

    summary = {
        "final_time": t,
        "final_ns": ns,
        "final_nr": nr,
    }
    return summary,(ts,nSs,nRr)


if __name__ == "__main__":
    seed  = 42
    ns0   = 20
    nr0   = 1
    Tmax  = 48.0
    k2a = k_avg()             # ← compute FIRST, resets internally
    print(f"k2_avg = {k2a:.4f} /h")
    print(f"breakeven = {k1+k4:.4f} /h")
    reset_pk_state()
    summ_u, traj_u = uncontrolled_ssa(ns0, nr0, Tmax, seed)
    reset_pk_state()
    summ_s, traj_s = ssa_standard(ns0, nr0, Tmax, seed)
    ts_u, nSs_u, nRs_u = traj_u
    ts_s, nSs_s, nRs_s, Cs_s, k2s_s = traj_s

    print("── Uncontrolled ──────────────────────────────")
    for k, v in summ_u.items():
        print(f"  {k:<15} = {v}")

    print("\n── Standard dosing ───────────────────────────")
    for k, v in summ_s.items():
        print(f"  {k:<15} = {v}")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        f'E. coli + Meropenem — seed={seed}  '
        f'ns0={ns0}  nr0={nr0}  Tmax={Tmax}h',
        fontsize=12, fontweight='bold')

    ax = axes[0, 0]
    ax.plot(ts_u, nSs_u, color='gray', lw=1.5, label='Uncontrolled')
    ax.plot(ts_s, nSs_s, color='#4361ee', lw=1.5, label='Standard dosing')
    ax.set_xlabel('time (h)'); ax.set_ylabel('nS count')
    ax.set_title('Sensitive bacteria nS(t)')
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(ts_u, nRs_u, color='gray', lw=1.5, label='Uncontrolled')
    ax.plot(ts_s, nRs_s, color='#ef4444', lw=1.5, label='Standard dosing')
    ax.set_xlabel('time (h)'); ax.set_ylabel('nR count')
    ax.set_title('Resistant bacteria nR(t)')
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(ts_s, k2s_s, color='#f59e0b', lw=1.2)
    ax.axhline(kmax, color='red', ls='--', lw=1, label=f'kmax={kmax}')
    ax.axhline(k1+k4, color='gray', ls=':', lw=1, label=f'breakeven={k1+k4:.3f}')
    ax.set_xlabel('time (h)'); ax.set_ylabel('k2 (/h)')
    ax.set_title('Drug killing rate k2(t) — sawtooth PK')
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(ts_s, Cs_s, color='#3b82f6', lw=1.2)
    ax.axhline(C0, color='gray', ls='--', lw=1, label=f'C0={C0}')
    ax.set_xlabel('time (h)'); ax.set_ylabel('C (mg/L)')
    ax.set_title('Drug concentration C(t)')
    ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    
    output_dir = "step 1 time varying constant dosing"
    os.makedirs(output_dir, exist_ok=True)
    filename = f"dosing_interval_{dose_interval}h_dynamics.png"
    filepath = os.path.join(output_dir, filename)
    
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\nSaved: {filepath}")
