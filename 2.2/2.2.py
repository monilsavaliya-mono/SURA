import numpy as np 
from scipy .sparse import lil_matrix 
from scipy .sparse .linalg import spsolve 
import matplotlib .pyplot as plt 
import csv ,os 

k1 =0.733 
k3 =0.733 *0.927 
k4 =0.179 
k5 =7.8e-7 
kmax =0.672 
EC50 =1.01 
C0 =30.0 
lam =0.693 
dose_interval =8.0 
s_max =40 
r_max =20 
beta =0.05 
t_last =0.0 
t_next =dose_interval 
C_peak =C0 
C_dose =C0 
C_curr =C0 
total_drug =C0 


def k2_from_c (c ):
    if c <=0 :
        return 0.0 
    return kmax *c /(c +EC50 )


def reset_pk_state ():
    global t_last ,t_next ,C_peak ,C_curr ,total_drug 
    t_last =0.0 
    t_next =dose_interval 
    C_peak =C0 
    C_curr =C0 
    total_drug =C0 


def solve_z (s_max ,r_max ,k1 ,k2 ,k3 ,k4 ,k5 ,beta ):
    N =(s_max +1 )*(r_max +1 )
    A =lil_matrix ((N ,N ))
    b =np .zeros (N )
    def index (ns ,nr ):
        return (r_max +1 )*ns +nr 
    def is_boundry (ns ,nr ):
        return (ns ==0 and nr ==0 )or ns ==s_max or nr ==r_max 

    for ns in range (s_max +1 ):
        for nr in range (r_max +1 ):
            i =index (ns ,nr )
            if (is_boundry (ns ,nr )):
                A [i ,i ]=1.0 
                b [i ]=1.0 
                continue 
            D =(k1 +k2 +k4 +k5 )*ns +(k3 +k4 )*nr +beta 
            A [i ,i ]=D 
            def add_nb (ns2 ,nr2 ,coef ,_i =i ):
                if ns2 <0 or nr2 <0 :
                    return 
                if ns2 >s_max or nr2 >r_max :
                    b [_i ]+=coef 
                    return 
                if is_boundry (ns2 ,nr2 ):
                    b [_i ]+=coef 
                else :
                    A [_i ,index (ns2 ,nr2 )]-=coef 
            add_nb (ns +1 ,nr ,k1 *ns )
            add_nb (ns -1 ,nr ,(k2 +k4 )*ns )
            add_nb (ns ,nr +1 ,k3 *nr )
            add_nb (ns ,nr -1 ,k4 *nr )
            add_nb (ns -1 ,nr +1 ,k5 *ns )
    z_flat =spsolve (A .tocsr (),b )
    if not (np .all (np .isfinite (z_flat ))and np .all (z_flat >0 )):
        print ("Warning: z has non-positive or infinite values, check the parameters.")
        return None 
    return z_flat .reshape ((s_max +1 ,r_max +1 ))





def k2_standard_at_t (t ):
    global t_last ,t_next ,C_peak ,C_curr ,total_drug 

    while t >=t_next :
        C_peak =C_peak *np .exp (-lam *dose_interval )+C_dose 
        t_last =t_next 
        t_next +=dose_interval 
        total_drug +=C_dose 

    C_now =C_peak *np .exp (-lam *(t -t_last ))
    C_curr =C_now 
    return k2_from_c (C_now )


def k_avg (n_cycles =8 ,n_points =200 ):
    global t_last ,t_next ,C_peak ,C_curr ,total_drug 
    reset_pk_state ()
    all_k2 =[]

    for cycle in range (n_cycles ):
        t_start =t_last 
        t_end =t_next 

        for t in np .linspace (t_start ,t_end -0.001 ,n_points ):
            C_now =C_peak *np .exp (-lam *(t -t_last ))
            all_k2 .append (k2_from_c (C_now ))

        k2_standard_at_t (t_end )

    reset_pk_state ()
    return float (np .mean (all_k2 ))


def dose_prx (ns ,nr ,C_trough ,Z ,k2a ,s_max ,r_max ):
    ns_c =max (0 ,min (ns ,s_max ))
    nr_c =max (0 ,min (nr ,r_max ))
    ns1 =max (0 ,min (ns -1 ,s_max ))

    zn =max (Z [ns_c ,nr_c ],1e-300 )
    z_nb =max (Z [ns1 ,nr_c ],1e-300 )

    ratio =z_nb /zn 
    k2_opt =k2a *ratio 

    if k2_opt <=0 :
        C_opt =0.0 
    elif k2_opt >=kmax :
        C_opt =3.0 *C0 
    else :
        C_opt =k2_opt *EC50 /(kmax -k2_opt )

    D =max (0.0 ,C_opt -C_trough )
    D =min (D ,3.0 *C0 -C_trough )
    return D 

def ssa_optimal (ns0 ,nr0 ,Z ,k2a ,Tmax ,seed ,pop_limit =5000 ):
    global t_last ,t_next ,C_peak ,C_curr ,total_drug 
    np .random .seed (seed )
    ns ,nr ,t =ns0 ,nr0 ,0.0 

    total_drug =C0 
    ts =[0. ];nSs =[ns ];nRs =[nr ];Cs =[C0 ]
    k2s =[k2_standard_at_t (0.0 )]

    while (ns +nr >0 )and (ns +nr <pop_limit )and t <Tmax :

        while t >=t_next :
            C_trough =C_peak *np .exp (-lam *(t_next -t_last ))
            D =dose_prx (ns ,nr ,C_trough ,Z ,k2a ,s_max ,r_max )
            C_peak =C_trough +D 
            t_last =t_next 
            total_drug +=D 
            t_next +=dose_interval 

        C_now =C_peak *np .exp (-lam *(t -t_last ))
        C_curr =C_now 
        k2 =k2_from_c (C_now )

        a1 =k1 *ns 
        a2 =k2 *ns 
        a3 =k3 *nr 
        a4 =k4 *nr 
        a5 =k5 *ns 
        a6 =k4 *ns 
        a0 =a1 +a2 +a3 +a4 +a5 +a6 

        if a0 <1e-15 :
            break 

        tau =-np .log (np .random .random ())/a0 
        t +=tau 
        if t >=Tmax :
            break 

        u =np .random .random ()*a0 
        if u <a1 :
            ns +=1 
        elif u <a1 +a2 :
            ns -=1 
        elif u <a1 +a2 +a3 :
            nr +=1 
        elif u <a1 +a2 +a3 +a4 :
            nr -=1 
        elif u <a1 +a2 +a3 +a4 +a5 :
            ns -=1 ;nr +=1 
        else :
            ns -=1 

        ns =max (ns ,0 );nr =max (nr ,0 )
        ts .append (t );nSs .append (ns )
        nRs .append (nr );Cs .append (C_now )
        k2s .append (k2 )

    summary ={
    "final_time":t ,
    "final_ns":ns ,
    "final_nr":nr ,
    "max_nR":max (nRs ),
    "n_doses":int (total_drug /C0 ),
    "total_drug":round (total_drug ,2 ),
    "eradicated":int (ns ==0 and nr ==0 ),
    "resistance":int (max (nRs )>5 ),
    }
    return summary ,(ts ,nSs ,nRs ,Cs ,k2s )


def ssa_standard (ns0 ,nr0 ,Tmax ,seed ,pop_limit =5000 ):
    global total_drug 
    np .random .seed (seed )
    ns ,nr ,t =ns0 ,nr0 ,0.0 
    ts =[0.0 ]
    nSs =[ns ]
    nRr =[nr ]
    Cs =[C_dose ]
    total_drug =C_dose 
    k2s =[k2_standard_at_t (0.0 )]

    while ((ns +nr >0 )and (ns +nr <pop_limit )and t <Tmax ):
        k2 =k2_standard_at_t (t )
        a1 =k1 *ns 
        a2 =k2 *ns 
        a3 =k3 *nr 
        a4 =k4 *nr 
        a5 =k5 *ns 
        a6 =k4 *ns 

        a0 =a1 +a2 +a3 +a4 +a5 +a6 
        if (a0 <1e-15 ):
            break 

        tau =-np .log (np .random .random ())/a0 
        t +=tau 
        if (t >=Tmax ):
            break 

        u =np .random .random ()*a0 
        if (u <a1 ):
            ns +=1 
        elif (u <a1 +a2 ):
            ns -=1 
        elif (u <a1 +a2 +a3 ):
            nr +=1 
        elif (u <a1 +a2 +a3 +a4 ):
            nr -=1 
        elif (u <a1 +a2 +a3 +a4 +a5 ):
            ns -=1 
            nr +=1 
        else :
            ns -=1 

        ns =max (ns ,0 )
        nr =max (nr ,0 )
        nSs .append (ns )
        nRr .append (nr )
        C_recorded = C_peak * np.exp(-lam * (t - t_last))
        Cs.append(C_recorded)   
        k2s .append (k2 )
        ts .append (t )
    n_doses =int (total_drug /C0 )
    summary ={
    "final_time":t ,
    "final_ns":ns ,
    "final_nr":nr ,
    "n_doses":n_doses ,
    "total_drug":total_drug ,
    "max_nR":max (nRr ),
    "eradicated":int (ns ==0 and nr ==0 ),
    "resistance":int (max (nRr )>5 ),
    }
    return summary ,(ts ,nSs ,nRr ,Cs ,k2s )


def uncontrolled_ssa (ns0 ,nr0 ,Tmax ,seed ,pop_limit =5000 ):

    np .random .seed (seed )
    ns ,nr ,t =ns0 ,nr0 ,0.0 
    ts =[0.0 ]
    nSs =[ns ]
    nRr =[nr ]

    while ((ns +nr >0 )and (ns +nr <pop_limit )and t <Tmax ):
        a1 =k1 *ns 
        a2 =0 
        a3 =k3 *nr 
        a4 =k4 *nr 
        a5 =k5 *ns 
        a6 =k4 *ns 

        a0 =a1 +a2 +a3 +a4 +a5 +a6 
        if (a0 <1e-15 ):
            break 

        tau =-np .log (np .random .random ())/a0 
        t +=tau 
        if (t >=Tmax ):
            break 

        u =np .random .random ()*a0 
        if (u <a1 ):
            ns +=1 
        elif (u <a1 +a2 ):
            ns -=1 
        elif (u <a1 +a2 +a3 ):
            nr +=1 
        elif (u <a1 +a2 +a3 +a4 ):
            nr -=1 
        elif (u <a1 +a2 +a3 +a4 +a5 ):
            ns -=1 
            nr +=1 
        else :
            ns -=1 

        ns =max (ns ,0 )
        nr =max (nr ,0 )
        nSs .append (ns )
        nRr .append (nr )
        ts .append (t )

    summary ={
    "final_time":t ,
    "final_ns":ns ,
    "final_nr":nr ,    "max_nR":max (nRr ),
    "eradicated":int (ns ==0 and nr ==0 ),
    "resistance":int (max (nRr )>5 ),    }
    return summary ,(ts ,nSs ,nRr )

if __name__ =="__main__":

    seed =42 
    ns0 ,nr0 ,Tmax =20 ,1 ,48.0 


    reset_pk_state ()
    k2a =k_avg ()
    print (f"k2_avg    = {k2a:.4f} /h")
    print (f"breakeven = {k1+k4:.4f} /h")
    print (f"kmax      = {kmax:.4f} /h")
    print (f"kmax > breakeven? {kmax > k1+k4}")


    print ("\nSolving Z...")
    reset_pk_state ()
    Z =solve_z (s_max ,r_max ,k1 ,k2a ,k3 ,k4 ,k5 ,beta )
    if Z is None :
        print ("Z failed — try smaller beta")
        exit ()
    print (f"Z shape : {Z.shape}")
    print (f"Z range : [{Z.min():.4f}, {Z.max():.4f}]")
    print (f"Z(0,0)  = {Z[0,0]:.4f}  ← must be 1.0")
    print (f"Z(1,0)  = {Z[1,0]:.4f}")
    print (f"Z(20,1) = {Z[min(20,s_max), min(1,r_max)]:.4f}")


    print ("\nRunning uncontrolled...")
    reset_pk_state ()
    summ_u ,traj_u =uncontrolled_ssa (ns0 ,nr0 ,Tmax ,seed )

    print ("Running standard dosing...")
    reset_pk_state ()
    summ_s ,traj_s =ssa_standard (ns0 ,nr0 ,Tmax ,seed )

    print ("Running optimal dosing...")
    reset_pk_state ()
    summ_o ,traj_o =ssa_optimal (ns0 ,nr0 ,Z ,k2a ,Tmax ,seed )


    ts_u ,nSs_u ,nRs_u =traj_u 
    ts_s ,nSs_s ,nRs_s ,Cs_s ,k2s_s =traj_s 
    ts_o ,nSs_o ,nRs_o ,Cs_o ,k2s_o =traj_o 


    print (f"\n{'Metric':<18} {'Uncontrolled':>14} "
    f"{'Standard':>12} {'Optimal':>12}")
    print ("-"*58 )
    for key in ['final_ns','final_nr','max_nR',
    'eradicated','resistance']:
        vu =summ_u .get (key ,'N/A')
        vs =summ_s .get (key ,'N/A')
        vo =summ_o .get (key ,'N/A')
        print (f"  {key:<16} {str(vu):>14} {str(vs):>12} {str(vo):>12}")
    print (f"  {'total_drug':<16} {'0':>14} "
    f"{str(summ_s.get('total_drug','N/A')):>12} "
    f"{str(summ_o.get('total_drug','N/A')):>12}")


    fig ,axes =plt .subplots (2 ,3 ,figsize =(16 ,10 ))
    fig .suptitle (
    f'E. coli + Meropenem  seed={seed}  '
    f'ns0={ns0}  nr0={nr0}  Tmax={Tmax}h  '
    f'dose_interval={dose_interval}h  β={beta}',
    fontsize =11 ,fontweight ='bold')


    ax =axes [0 ,0 ]
    ax .plot (ts_u ,nSs_u ,color ='gray',lw =1.5 ,
    label ='Uncontrolled')
    ax .plot (ts_s ,nSs_s ,color ='#4361ee',lw =1.5 ,
    label ='Standard')
    ax .plot (ts_o ,nSs_o ,color ='#10b981',lw =1.5 ,
    label ='Optimal')
    ax .set_xlabel ('time (h)');ax .set_ylabel ('nS count')
    ax .set_title ('Sensitive bacteria nS(t)')
    ax .legend (fontsize =9 );ax .grid (alpha =0.3 )


    ax =axes [0 ,1 ]
    ax .plot (ts_u ,nRs_u ,color ='gray',lw =1.5 ,
    label =f"Unc  maxNR={summ_u['max_nR']}")
    ax .plot (ts_s ,nRs_s ,color ='#ef4444',lw =1.5 ,
    label =f"Std  maxNR={summ_s['max_nR']}")
    ax .plot (ts_o ,nRs_o ,color ='#f59e0b',lw =1.5 ,
    label =f"Opt  maxNR={summ_o['max_nR']}")
    ax .set_xlabel ('time (h)');ax .set_ylabel ('nR count')
    ax .set_title ('Resistant bacteria nR(t)  ← KEY RESULT')
    ax .legend (fontsize =9 );ax .grid (alpha =0.3 )


    ax =axes [0 ,2 ]
    im =ax .imshow (Z ,origin ='lower',aspect ='auto',
    extent =[0 ,r_max ,0 ,s_max ],cmap ='viridis')
    plt .colorbar (im ,ax =ax )
    ax .scatter ([nr0 ],[ns0 ],color ='red',s =80 ,zorder =5 ,
    label =f'start ({ns0},{nr0})')
    ax .set_xlabel ('nR');ax .set_ylabel ('nS')
    ax .set_title ('Z(nS,nR) — value function')
    ax .legend (fontsize =9 )


    ax =axes [1 ,0 ]
    ax .plot (ts_s ,k2s_s ,color ='#4361ee',lw =1.0 ,
    alpha =0.8 ,label ='Standard k2(t)')
    ax .plot (ts_o ,k2s_o ,color ='#10b981',lw =1.0 ,
    alpha =0.8 ,label ='Optimal k2†(t)')
    ax .axhline (k1 +k4 ,color ='red',ls ='--',lw =1.2 ,
    label =f'breakeven={k1+k4:.3f}')
    ax .axhline (kmax ,color ='gray',ls =':',lw =1.0 ,
    label =f'kmax={kmax}')
    ax .set_xlabel ('time (h)');ax .set_ylabel ('k2 (/h)')
    ax .set_title ('Killing rate — standard vs optimal')
    ax .legend (fontsize =8 );ax .grid (alpha =0.3 )


    ax =axes [1 ,1 ]
    ax .plot (ts_s ,Cs_s ,color ='#4361ee',lw =1.0 ,
    alpha =0.8 ,label ='Standard C(t)')
    ax .plot (ts_o ,Cs_o ,color ='#10b981',lw =1.0 ,
    alpha =0.8 ,label ='Optimal C(t)')
    ax .axhline (C0 ,color ='gray',ls ='--',lw =1 ,
    label =f'C0={C0} mg/L')
    ax .set_xlabel ('time (h)');ax .set_ylabel ('C (mg/L)')
    ax .set_title ('Drug concentration C(t)')
    ax .legend (fontsize =9 );ax .grid (alpha =0.3 )


    ax =axes [1 ,2 ]
    ax .axis ('off')
    lines =[
    "── Summary ──────────────────────",
    f"",
    f"{'Metric':<18}{'Unc':>6}{'Std':>8}{'Opt':>8}",
    f"{'─'*40}",
    f"{'max nR':<18}"
    f"{summ_u['max_nR']:>6}"
    f"{summ_s['max_nR']:>8}"
    f"{summ_o['max_nR']:>8}",
    f"{'final nR':<18}"
    f"{summ_u['final_nr']:>6}"
    f"{summ_s['final_nr']:>8}"
    f"{summ_o['final_nr']:>8}",
    f"{'eradicated':<18}"
    f"{summ_u['eradicated']:>6}"
    f"{summ_s['eradicated']:>8}"
    f"{summ_o['eradicated']:>8}",
    f"{'resistance>5':<18}"
    f"{summ_u['resistance']:>6}"
    f"{summ_s['resistance']:>8}"
    f"{summ_o['resistance']:>8}",
    f"",
    f"{'total drug':<18}"
    f"{'N/A':>6}"
    f"{summ_s['total_drug']:>8.1f}"
    f"{summ_o['total_drug']:>8.1f}",
    f"",
    f"k2_avg    = {k2a:.4f} /h",
    f"breakeven = {k1+k4:.4f} /h",
    f"kmax      = {kmax:.4f} /h",
    f"beta      = {beta}",
    f"grid      = {s_max}×{r_max}",
    ]
    ax .text (0.05 ,0.95 ,"\n".join (lines ),
    transform =ax .transAxes ,
    fontsize =9 ,va ='top',
    fontfamily ='monospace',
    bbox =dict (boxstyle ='round',
    facecolor ='#f0f4ff',
    edgecolor ='#4361ee',
    alpha =0.85 ))

    plt .tight_layout ()

    output_dir ="step 2 folder time varying dosing"
    os .makedirs (output_dir ,exist_ok =True )
    filename =(f"all_three_seed{seed}"
    f"_interval{dose_interval}h.png")
    filepath =os .path .join (output_dir ,filename )
    plt .savefig (filepath ,dpi =150 ,bbox_inches ='tight')
    plt .show ()
    print (f"\nSaved: {filepath}")