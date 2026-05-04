import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import time, io, sys, traceback

st.set_page_config(page_title="Systolic Array Simulator", layout="wide", page_icon="⬛")

st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    .title { font-family: monospace; font-size: 22px; font-weight: 600; color: #e2e8f0; letter-spacing:.05em }
    .subtitle { font-family: monospace; font-size: 12px; color: #718096; margin-top:-8px }
    .mode-desc { font-family: monospace; font-size: 12px; color: #a0aec0; font-style:italic; padding:8px 0 }
    div[data-testid="metric-container"] { background:#1a202c; border-radius:8px; padding:8px; border:0.5px solid #2d3748 }
    .step-log { font-family:monospace; font-size:11px; background:#1a202c; padding:8px; border-radius:6px; color:#a0aec0; max-height:160px; overflow-y:auto }
    .terminal-wrap { background:#0d1117; border:1px solid #30363d; border-radius:8px; overflow:hidden }
    .terminal-bar { background:#161b22; padding:8px 14px; font-family:monospace; font-size:11px; color:#8b949e; border-bottom:1px solid #30363d }
    .terminal-body { background:#0d1117; padding:12px 14px; font-family:'Courier New',monospace; font-size:12px; color:#e6edf3; min-height:100px; white-space:pre-wrap; word-break:break-all; line-height:1.6 }
    .t-prompt { color:#388bfd } .t-out { color:#7ee787 } .t-err { color:#ff7b72 }
    .checker-pass    { background:#1c3a2a; border:1px solid #2ea043; border-radius:8px; padding:12px 16px; font-family:monospace; font-size:13px; color:#7ee787 }
    .checker-fail    { background:#3a1c1c; border:1px solid #da3633; border-radius:8px; padding:12px 16px; font-family:monospace; font-size:13px; color:#ff7b72 }
    .checker-pending { background:#1c2333; border:1px solid #30363d; border-radius:8px; padding:12px 16px; font-family:monospace; font-size:13px; color:#8b949e }
    .dim-note { font-family:monospace; font-size:11px; color:#4a9eff; background:#0d1f33; border:1px solid #1a3a5c; border-radius:6px; padding:6px 10px; margin:4px 0 }
</style>
""", unsafe_allow_html=True)

# ── State init ────────────────────────────────────────────────────────────────
def init_state():
    defaults = dict(
        mode="weight_stationary", N=3, M=3, P=3,
        sim_steps=[], current_step=-1, C_result=None,
        log=[], running=False,
        terminal_history=[], python_C=None,
        checker_result=None, checker_detail="",
        term_ns={},
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()
if "np" not in st.session_state.term_ns:
    st.session_state.term_ns["np"] = np

# ── Modes ─────────────────────────────────────────────────────────────────────
MODE_DESC = {
    "weight_stationary": "Weights (B) stay fixed in each PE. Input rows (A) stream east, outputs accumulate. TPU v1 style.",
    "input_stationary":  "Inputs (A) stay fixed per PE. Weight columns (B) stream south, partials flow east.",
    "output_stationary": "Partial sums stay in each PE. A streams east, B streams south simultaneously.",
    "row_stationary":    "Each PE row holds one row of A. Weights stream south, results drain east.",
}
MODE_LABELS = {k: k.replace("_", " ").title() for k in MODE_DESC}

def fmt_val(v):
    f = float(v)
    return str(int(f)) if f == int(f) else f"{f:.2g}"

# ── Simulation engine ─────────────────────────────────────────────────────────
# A is M×N (M west-input streams, each of length N = inner dim)
# B is N×P (P north-input streams, each of length N)
# Output C is M×P
# The array is N×N. We tile: each PE(i,j) sees row i of A and col j of B.
# For M>N or P>N, we map multiple logical rows/cols across the array in passes,
# but the simpler and more useful model: we keep the full streaming wavefront,
# just with M input streams on the west and P on the north. PEs are re-used
# across logical rows/cols via the diagonal wavefront with appropriate offsets.
#
# Concretely: the array computes C[m,p] = sum_k A[m,k]*B[k,p] for all m,p.
# Each PE(i,j) in the N×N array handles output C[i,j] for i<M, j<P.
# For i>=M the row is inactive; for j>=P the col is inactive.
# Inner dimension K is always N (the array size) — that's what the array is built for.

def build_sim_steps(A_full, B_full, pe_overrides, mode, N, M, P):
    """
    A_full: M×N, B_full: N×P
    Streams A rows west→east and B cols north→south through the N×N array.
    PEs (i,j) for i<M and j<P accumulate C[i,j].
    """
    # Pad A to N×N (add zero rows if M<N), pad B to N×N (add zero cols if P<N)
    # If M>N or P>N, we only show N streams at a time (first N); user sees the concept.
    A = np.zeros((N, N))
    B = np.zeros((N, N))
    A[:min(M,N), :] = A_full[:min(M,N), :N]
    B[:, :min(P,N)] = B_full[:N, :min(P,N)]

    acc   = np.zeros((N, N))
    steps = []
    total = 3 * N - 1

    for t in range(total):
        a_regs = [[None]*N for _ in range(N)]
        b_regs = [[None]*N for _ in range(N)]
        fired  = []

        for i in range(N):
            for j in range(N):
                k = t - i - j
                active_i = (i < M)
                active_j = (j < P)
                av = float(A[i][k]) if (0 <= k < N and active_i) else (
                     0.0            if (0 <= k < N and not active_i) else None)
                bv = float(B[k][j]) if (0 <= k < N and active_j) else (
                     0.0            if (0 <= k < N and not active_j) else None)
                a_regs[i][j] = av if active_i else None
                b_regs[i][j] = bv if active_j else None

                if av is not None and bv is not None:
                    ov = pe_overrides[i][j]
                    if mode == "weight_stationary":
                        mult = av * (ov if ov is not None else bv)
                    elif mode == "input_stationary":
                        mult = (ov if ov is not None else av) * bv
                    elif mode == "output_stationary":
                        mult = av * bv
                    else:
                        mult = (ov if ov is not None else av) * bv
                    acc[i][j] += mult
                    fired.append(dict(i=i, j=j, av=av, bv=bv, mult=mult,
                                      acc=acc[i][j],
                                      inactive=(not active_i or not active_j)))

        steps.append(dict(t=t,
                          a_regs=[r[:] for r in a_regs],
                          b_regs=[r[:] for r in b_regs],
                          acc=acc.copy(), fired=fired,
                          M=M, P=P))

    # Build result matrix (only M×P portion is meaningful)
    C = acc[:min(M,N), :min(P,N)].copy()
    return steps, acc.copy(), C

# ── Drawing ───────────────────────────────────────────────────────────────────
def draw_array(state, N, pe_overrides, mode_name, M, P):
    DARK="#0e1117"; CELL="#1a202c"; BORD="#2d3748"; DIM="#4a5568"
    A_COL="#4299e1"; B_COL="#ed8936"; ACC_COL="#48bb78"
    STAT_COL="#fc8181"; ACT_CELL="#1c3a2a"; TXT="#e2e8f0"
    INK_CELL="#141820"; INK_BORD="#1e2635"

    if   N<=4:  cell,gap,lpad=1.00,0.22,1.20; fpe,fv,fa=6.5,7.0,8.0
    elif N<=6:  cell,gap,lpad=0.80,0.18,1.10; fpe,fv,fa=5.5,6.0,7.0
    elif N<=8:  cell,gap,lpad=0.62,0.14,1.00; fpe,fv,fa=4.5,5.0,6.0
    elif N<=11: cell,gap,lpad=0.50,0.11,0.90; fpe,fv,fa=4.0,4.5,5.0
    else:       cell,gap,lpad=0.38,0.09,0.80; fpe,fv,fa=3.0,3.5,4.0

    stp=cell+gap
    tw=N*stp-gap+lpad*2; th=N*stp-gap+lpad*2
    fig,ax=plt.subplots(figsize=(min(max(tw*1.05,5),16), min(max(th*1.05,4),16)))
    fig.patch.set_facecolor(DARK); ax.set_facecolor(DARK)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_xlim(-0.5,tw+0.3); ax.set_ylim(-0.5,th+0.3)

    for i in range(N):
        for j in range(N):
            cx=lpad+j*stp; cy=lpad+(N-1-i)*stp
            av  = state["a_regs"][i][j] if state else None
            bv  = state["b_regs"][i][j] if state else None
            acc_v = state["acc"][i][j]  if state else 0.0
            active = state and any(f["i"]==i and f["j"]==j and not f.get("inactive")
                                   for f in state["fired"])
            inactive = (i >= min(M,N)) or (j >= min(P,N))

            fc = ACT_CELL if (active and not inactive) else (INK_CELL if inactive else CELL)
            ec = ACC_COL  if (active and not inactive) else (INK_BORD if inactive else BORD)
            lw = 1.2      if (active and not inactive) else 0.5
            alpha = 0.4   if inactive else 1.0

            rect=mpatches.FancyBboxPatch((cx-cell/2,cy-cell/2),cell,cell,
                boxstyle="round,pad=0.03",linewidth=lw,
                edgecolor=ec,facecolor=fc,zorder=2,alpha=alpha)
            ax.add_patch(rect)

            if inactive:
                hatch=mpatches.FancyBboxPatch((cx-cell/2,cy-cell/2),cell,cell,
                    boxstyle="round,pad=0.03",linewidth=0,
                    edgecolor="#2d3748",facecolor="none",
                    hatch="////",zorder=2,alpha=0.2)
                ax.add_patch(hatch)
                if N<=10:
                    ax.text(cx,cy,"—",ha="center",va="center",fontsize=fa,
                            color="#252a36",fontfamily="monospace",zorder=3)

            if N<=10 and not inactive:
                ax.text(cx,cy+cell/2-0.09,f"PE{i},{j}",ha="center",va="top",
                        fontsize=fpe,color=DIM,fontfamily="monospace",zorder=3)
            elif N<=10:
                ax.text(cx,cy+cell/2-0.09,f"PE{i},{j}",ha="center",va="top",
                        fontsize=fpe,color="#252a36",fontfamily="monospace",zorder=3)

            if N<=7 and not inactive:
                if av is not None and av!=0.0:
                    ax.text(cx-cell/2+0.05,cy+cell*0.15,f"a:{fmt_val(av)}",
                            ha="left",va="center",fontsize=fv,color=A_COL,
                            fontfamily="monospace",zorder=3)
                if bv is not None and bv!=0.0:
                    ax.text(cx+cell/2-0.05,cy+cell*0.15,f"b:{fmt_val(bv)}",
                            ha="right",va="center",fontsize=fv,color=B_COL,
                            fontfamily="monospace",zorder=3)

            dms=max(1.5,5-N//3)
            if av is not None and i<min(M,N):
                ax.plot(cx-cell/2,cy,"o",ms=dms,color=A_COL,alpha=0.85,zorder=4)
            if bv is not None and j<min(P,N):
                ax.plot(cx,cy+cell/2,"o",ms=dms,color=B_COL,alpha=0.85,zorder=4)

            if not inactive:
                lbl=("Σ" if N<=10 else "")+fmt_val(acc_v)
                ax.text(cx,cy-cell/2+0.10,lbl,ha="center",va="bottom",
                        fontsize=fa,fontweight="bold",
                        color=ACC_COL if active else DIM,
                        fontfamily="monospace",zorder=3)

            ov=pe_overrides[i][j]
            if ov is not None and N<=8 and not inactive:
                ax.text(cx+cell/2-0.04,cy+cell/2-0.07,"★",ha="right",va="top",
                        fontsize=fpe,color=STAT_COL,fontfamily="monospace",zorder=3)

    # West arrows (A rows) — show M labels, dim the rest
    for i in range(N):
        cy=lpad+(N-1-i)*stp
        active=(i<min(M,N))
        col=A_COL if active else "#2d3748"
        ax.annotate("",xy=(lpad-cell/2-0.04,cy),xytext=(lpad-cell/2-0.32,cy),
                    arrowprops=dict(arrowstyle="->",color=col,lw=0.8,
                                   alpha=1.0 if active else 0.25))
        if N<=13:
            lbl=f"A[{i}]" if active else f"—"
            ax.text(lpad-cell/2-0.35,cy,lbl,ha="right",va="center",
                    fontsize=max(4.5,fpe),color=col,fontfamily="monospace",
                    alpha=1.0 if active else 0.3)

    # North arrows (B cols)
    for j in range(N):
        cx=lpad+j*stp; top=lpad+(N-1)*stp
        active=(j<min(P,N))
        col=B_COL if active else "#2d3748"
        ax.annotate("",xy=(cx,top+cell/2+0.04),xytext=(cx,top+cell/2+0.30),
                    arrowprops=dict(arrowstyle="->",color=col,lw=0.8,
                                   alpha=1.0 if active else 0.25))
        if N<=13:
            lbl=f"B[{j}]" if active else f"—"
            ax.text(cx,top+cell/2+0.34,lbl,ha="center",va="bottom",
                    fontsize=max(4.5,fpe),color=col,fontfamily="monospace",
                    alpha=1.0 if active else 0.3)

    # Show dimensions
    dim_str = f"A: {min(M,N)}×{N}  ·  B: {N}×{min(P,N)}  ·  C: {min(M,N)}×{min(P,N)}"
    ax.text((tw)/2, -0.35, dim_str, ha="center", va="top", fontsize=7,
            color="#4a9eff", fontfamily="monospace")

    ax.text(tw+0.1,0,mode_name.replace("_"," "),ha="right",va="bottom",
            fontsize=6,color=DIM,fontfamily="monospace",style="italic")

    legend_items=[
        mpatches.Patch(color=A_COL,   label="A input (east)"),
        mpatches.Patch(color=B_COL,   label="B weight (south)"),
        mpatches.Patch(color=ACC_COL, label="accumulator"),
        mpatches.Patch(color=STAT_COL,label="override"),
    ]
    ax.legend(handles=legend_items,loc="lower right",fontsize=6,
              facecolor="#1a202c",edgecolor="#2d3748",labelcolor=TXT,
              framealpha=1,handlelength=1)
    plt.tight_layout(pad=0.3)
    return fig

# ── Terminal ──────────────────────────────────────────────────────────────────
def run_python(cmd, ns):
    sb=io.StringIO(); eb=io.StringIO()
    so,se=sys.stdout,sys.stderr
    sys.stdout,sys.stderr=sb,eb
    try:
        try:
            r=eval(compile(cmd,"<t>","eval"),ns)
            if r is not None: print(repr(r))
        except SyntaxError:
            exec(compile(cmd,"<t>","exec"),ns)
    except Exception:
        print(traceback.format_exc(),file=sys.stderr)
    finally:
        sys.stdout,sys.stderr=so,se
    return sb.getvalue(),eb.getvalue()

def auto_numpy(A,B):
    ns={"np":np,"A":A.copy(),"B":B.copy()}
    cmd="C = np.dot(A, B)\nprint(C)"
    out,err=run_python(cmd,ns)
    return ns.get("C"),cmd,out

def run_checker(sim_C, py_C):
    if sim_C is None or py_C is None:
        return "pending","Waiting for both results..."
    try:
        if py_C.shape != sim_C.shape:
            py_C2 = np.zeros_like(sim_C)
            r = min(sim_C.shape[0], py_C.shape[0])
            c = min(sim_C.shape[1], py_C.shape[1])
            py_C2[:r,:c] = py_C[:r,:c]
            py_C = py_C2
        if np.allclose(sim_C,py_C,atol=1e-6):
            return "pass",(f"All {sim_C.size} elements match  |  "
                           f"max |err|={np.max(np.abs(sim_C-py_C)):.2e}")
        bad=np.argwhere(~np.isclose(sim_C,py_C,atol=1e-6))
        s=", ".join(f"C[{r},{c}]: sim={sim_C[r,c]:.4g} np={py_C[r,c]:.4g}" for r,c in bad[:4])
        return "fail",f"{len(bad)} mismatch(es) — {s}"
    except Exception as e:
        return "fail",str(e)

# ════════════════════════════════════════════════════════════════════════════
# Sidebar
# ════════════════════════════════════════════════════════════════════════════
def reset_sim():
    for k in ["sim_steps","log","terminal_history"]: st.session_state[k]=[]
    st.session_state.current_step=-1
    for k in ["C_result","python_C","checker_result"]: st.session_state[k]=None
    st.session_state.checker_detail=""
    st.session_state.running=False

with st.sidebar:
    st.markdown("### Configuration")

    mode_key=st.selectbox("Dataflow mode",options=list(MODE_DESC.keys()),
                          format_func=lambda k:MODE_LABELS[k],
                          index=list(MODE_DESC.keys()).index(st.session_state.mode))
    st.session_state.mode=mode_key
    st.markdown(f'<div class="mode-desc">{MODE_DESC[mode_key]}</div>',unsafe_allow_html=True)

    st.markdown("---")

    # Array size
    N=st.selectbox("Array size (N×N — inner dimension)",
                   options=list(range(2,16)),
                   index=list(range(2,16)).index(st.session_state.N),
                   format_func=lambda n:f"{n}×{n}")
    if N!=st.session_state.N:
        st.session_state.N=N
        st.session_state.M=N
        st.session_state.P=N
        reset_sim()
        if "_rand_A" in st.session_state: del st.session_state["_rand_A"]
        if "_rand_B" in st.session_state: del st.session_state["_rand_B"]

    # Input stream counts — independent of N
    st.markdown("**Input stream counts**")
    st.caption("M = west (A rows), P = north (B cols). Independent of array size.")
    c1,c2=st.columns(2)
    M=c1.number_input("M (A rows)",min_value=1,max_value=64,
                       value=st.session_state.M,step=1,key="m_input")
    P=c2.number_input("P (B cols)",min_value=1,max_value=64,
                       value=st.session_state.P,step=1,key="p_input")
    st.session_state.M=int(M); st.session_state.P=int(P)

    Meff=min(int(M),N); Peff=min(int(P),N)
    st.markdown(
        f'<div class="dim-note">A: <b>{Meff}×{N}</b> &nbsp;·&nbsp; '
        f'B: <b>{N}×{Peff}</b> &nbsp;·&nbsp; '
        f'C: <b>{Meff}×{Peff}</b>'
        + (f"<br>⚠ M>{N}: showing first {N} rows" if int(M)>N else "")
        + (f"<br>⚠ P>{N}: showing first {N} cols" if int(P)>N else "")
        + '</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Matrix A input
    def default_A(m,n):
        return np.fromfunction(lambda i,j:(i*n+j+1)%9+1,(m,n),dtype=float)
    def default_B(n,p):
        return np.eye(n,p)+np.fromfunction(lambda i,j:(i+j)%3,(n,p),dtype=float)*0.5

    if "_rand_A" in st.session_state:
        A_init=st.session_state["_rand_A"]
        B_init=st.session_state["_rand_B"]
    else:
        A_init=default_A(Meff,N)
        B_init=default_B(N,Peff)

    st.markdown(f"**Matrix A  ({Meff}×{N})**")
    if Meff<=5 and N<=5:
        A_rows=[]
        for i in range(Meff):
            cols=st.columns(N)
            row=[cols[j].number_input(f"A{i}{j}",value=float(A_init[i,j] if i<A_init.shape[0] and j<A_init.shape[1] else 0),
                                      label_visibility="collapsed",key=f"a_{i}_{j}",step=1.0) for j in range(N)]
            A_rows.append(row)
        A_full=np.array(A_rows,dtype=float)
    else:
        raw=st.text_area(f"A ({Meff} rows × {N} cols, space-separated)",key="a_raw",
                         value="\n".join(" ".join(str(int(A_init[i,j]) if i<A_init.shape[0] and j<A_init.shape[1] else 0)
                                                  for j in range(N)) for i in range(Meff)),
                         height=min(220,Meff*20+40))
        try:
            rows=[list(map(float,r.split())) for r in raw.strip().splitlines() if r.strip()]
            A_full=np.array(rows,dtype=float) if (len(rows)==Meff and all(len(r)==N for r in rows)) else A_init
        except Exception: A_full=A_init

    st.markdown(f"**Matrix B  ({N}×{Peff})**")
    if N<=5 and Peff<=5:
        B_rows=[]
        for i in range(N):
            cols=st.columns(Peff)
            row=[cols[j].number_input(f"B{i}{j}",value=float(B_init[i,j] if i<B_init.shape[0] and j<B_init.shape[1] else 0),
                                      label_visibility="collapsed",key=f"b_{i}_{j}",step=1.0) for j in range(Peff)]
            B_rows.append(row)
        B_full=np.array(B_rows,dtype=float)
    else:
        raw=st.text_area(f"B ({N} rows × {Peff} cols, space-separated)",key="b_raw",
                         value="\n".join(" ".join(str(int(B_init[i,j]) if i<B_init.shape[0] and j<B_init.shape[1] else 0)
                                                  for j in range(Peff)) for i in range(N)),
                         height=min(220,N*20+40))
        try:
            rows=[list(map(float,r.split())) for r in raw.strip().splitlines() if r.strip()]
            B_full=np.array(rows,dtype=float) if (len(rows)==N and all(len(r)==Peff for r in rows)) else B_init
        except Exception: B_full=B_init

    if st.button("🎲 Randomise A & B",use_container_width=True):
        rng=np.random.default_rng()
        st.session_state["_rand_A"]=rng.integers(1,9,(Meff,N)).astype(float)
        st.session_state["_rand_B"]=rng.integers(1,9,(N,Peff)).astype(float)
        st.rerun()

    st.markdown("---")
    st.markdown("**PE Stationary Overrides**")
    st.caption("Pin a value to a PE. 0 = auto.")
    pe_overrides=[[None]*N for _ in range(N)]
    if N<=5:
        for i in range(N):
            cols=st.columns(N)
            for j in range(N):
                v=cols[j].number_input(f"PE{i}{j}",value=0.0,
                                        label_visibility="collapsed",
                                        key=f"pe_{i}_{j}",step=1.0)
                pe_overrides[i][j]=v if v!=0.0 else None
    else:
        st.caption("PE overrides available for N≤5.")

    st.markdown("---")
    speed=st.slider("Sim speed (ms/step)",50,1500,400,step=50)

# Sync terminal ns
st.session_state.term_ns["A"]=A_full.copy()
st.session_state.term_ns["B"]=B_full.copy()
if st.session_state.C_result is not None:
    st.session_state.term_ns["C_sim"]=st.session_state.C_result.copy()

# ════════════════════════════════════════════════════════════════════════════
# Main area
# ════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="title">Systolic Array Simulator</div>',unsafe_allow_html=True)
st.markdown('<div class="subtitle">Interactive matrix multiply · configurable dataflow · variable input streams · Python verifier</div>',unsafe_allow_html=True)
st.markdown("---")

col_vis,col_ctrl=st.columns([3,1])

def do_build():
    steps,acc,C=build_sim_steps(A_full,B_full,pe_overrides,mode_key,N,int(M),int(P))
    st.session_state.sim_steps=steps
    st.session_state.C_result=C
    st.session_state.log=[]
    st.session_state.python_C=None
    st.session_state.checker_result=None
    st.session_state.checker_detail=""
    return steps, C

with col_ctrl:
    st.markdown("### Controls")

    if st.button("▶ Simulate",use_container_width=True,type="primary"):
        do_build()
        st.session_state.current_step=-1
        st.session_state.running=True

    if st.button("→ Step",use_container_width=True):
        if not st.session_state.sim_steps: do_build()
        if st.session_state.current_step<len(st.session_state.sim_steps)-1:
            st.session_state.current_step+=1

    if st.button("⚡ Finish Now",use_container_width=True):
        if not st.session_state.sim_steps: do_build()
        last=len(st.session_state.sim_steps)-1
        st.session_state.current_step=last
        st.session_state.running=False
        full_log=[]
        for s in st.session_state.sim_steps:
            for f in s["fired"]:
                if f.get("inactive"): continue
                full_log.append(f"t={s['t']} | PE({f['i']},{f['j']}): "
                                 f"{fmt_val(f['av'])}×{fmt_val(f['bv'])}={fmt_val(f['mult'])} → Σ={fmt_val(f['acc'])}")
        st.session_state.log=full_log

    if st.button("↺ Reset",use_container_width=True):
        reset_sim()

    st.markdown("---")
    total=len(st.session_state.sim_steps)
    cur=st.session_state.current_step
    st.metric("Step",f"{max(cur,0)}/{total}" if total else "—")
    st.metric("Array",f"{N}×{N}")
    st.metric("A dims",f"{Meff}×{N}")
    st.metric("B dims",f"{N}×{Peff}")

    sim_done=(st.session_state.C_result is not None and total>0 and cur>=total-1)
    if sim_done:
        st.markdown("**Simulated C**")
        C_disp=st.session_state.C_result
        sN=min(C_disp.shape[0],6); sP=min(C_disp.shape[1],6)
        for i in range(sN):
            cols=st.columns(sP)
            for j in range(sP):
                cols[j].markdown(
                    f"<div style='background:#1c3a2a;border-radius:5px;padding:3px;"
                    f"text-align:center;font-family:monospace;font-size:10px;"
                    f"color:#48bb78;font-weight:600'>{fmt_val(C_disp[i,j])}</div>",
                    unsafe_allow_html=True)
        if C_disp.shape[0]>6 or C_disp.shape[1]>6:
            st.caption(f"({sN}×{sP} of {C_disp.shape[0]}×{C_disp.shape[1]})")

# Auto-advance
if st.session_state.running and st.session_state.sim_steps:
    if st.session_state.current_step<len(st.session_state.sim_steps)-1:
        st.session_state.current_step+=1
    else:
        st.session_state.running=False

with col_vis:
    state=None
    if st.session_state.sim_steps and st.session_state.current_step>=0:
        idx=min(st.session_state.current_step,len(st.session_state.sim_steps)-1)
        state=st.session_state.sim_steps[idx]
        for f in state["fired"]:
            if f.get("inactive"): continue
            entry=(f"t={state['t']} | PE({f['i']},{f['j']}): "
                   f"{fmt_val(f['av'])}×{fmt_val(f['bv'])}={fmt_val(f['mult'])} → Σ={fmt_val(f['acc'])}")
            if entry not in st.session_state.log:
                st.session_state.log.append(entry)

    vis_M=state["M"] if state else int(M)
    vis_P=state["P"] if state else int(P)
    fig=draw_array(state,N,pe_overrides,mode_key,vis_M,vis_P)
    st.pyplot(fig,use_container_width=True)
    plt.close(fig)

    if st.session_state.log:
        st.markdown("**Simulation log**")
        cur_t=state["t"] if state else -1
        log_html="<br>".join(
            f'<span class="t-prompt">{l}</span>' if f"t={cur_t} |" in l
            else f'<span style="color:#4a5568">{l}</span>'
            for l in st.session_state.log[-35:])
        st.markdown(f'<div class="step-log">{log_html}</div>',unsafe_allow_html=True)

if st.session_state.running:
    time.sleep(speed/1000)
    st.rerun()

# ════════════════════════════════════════════════════════════════════════════
# Python Terminal + Checker
# ════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("## Python Verifier")
st.caption("Terminal pre-loaded with A, B, C_sim. Checker auto-runs np.dot(A,B) when simulation finishes.")

term_col,check_col=st.columns([3,2])

with term_col:
    st.markdown("""<div class="terminal-wrap">
      <div class="terminal-bar">
        <span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#ff5f57;margin-right:4px"></span>
        <span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#febc2e;margin-right:4px"></span>
        <span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#28c840;margin-right:8px"></span>
        python3 &nbsp;·&nbsp; numpy &nbsp;·&nbsp; systolic_verifier
      </div></div>""",unsafe_allow_html=True)

    lines=[]
    if not st.session_state.terminal_history:
        lines.append('<span class="t-prompt"># A, B loaded from simulator | C_sim available after simulation</span>')
        lines.append('<span class="t-prompt">&gt;&gt;&gt; </span>')
    else:
        for e in st.session_state.terminal_history:
            for ln in e["cmd"].splitlines():
                lines.append(f'<span class="t-prompt">&gt;&gt;&gt; {ln}</span>')
            if e["out"]: lines.append(f'<span class="t-out">{e["out"].rstrip()}</span>')
            if e["err"]: lines.append(f'<span class="t-err">{e["err"].rstrip()}</span>')
        lines.append('<span class="t-prompt">&gt;&gt;&gt; </span>')

    st.markdown(f'<div class="terminal-wrap"><div class="terminal-body">{"<br>".join(lines)}</div></div>',
                unsafe_allow_html=True)

    c1,c2,c3=st.columns([5,1,1])
    cmd_input=c1.text_input("cmd",value="",label_visibility="collapsed",
                             placeholder=">>> type Python here  (np, A, B, C_sim available)",
                             key="term_input")
    run_btn=c2.button("Run ▶",use_container_width=True)
    clr_btn=c3.button("Clear",use_container_width=True)

    if clr_btn:
        st.session_state.terminal_history=[]
        st.rerun()

    def execute_cmd(cmd):
        out,err=run_python(cmd,st.session_state.term_ns)
        st.session_state.terminal_history.append({"cmd":cmd,"out":out,"err":err})
        if "C" in st.session_state.term_ns:
            try:
                cand=np.array(st.session_state.term_ns["C"],dtype=float)
                sim=st.session_state.C_result
                if sim is not None:
                    # trim to matching shape
                    r=min(cand.shape[0],sim.shape[0]); c=min(cand.shape[1],sim.shape[1])
                    st.session_state.python_C=cand[:r,:c]
                    res,det=run_checker(sim,cand)
                    st.session_state.checker_result=res
                    st.session_state.checker_detail=det
            except Exception: pass

    if run_btn and cmd_input.strip():
        execute_cmd(cmd_input); st.rerun()

    st.markdown("**Quick commands:**")
    qcols=st.columns(5)
    quick=[("np.dot(A,B)","C = np.dot(A, B)\nprint(C)"),
           ("A @ B","C = A @ B\nprint(C)"),
           ("print A","print(A)"),
           ("print B","print(B)"),
           ("C_sim","print(C_sim)" if st.session_state.C_result is not None else "print('run simulation first')")]
    for col,(lbl,cmd) in zip(qcols,quick):
        if col.button(lbl,use_container_width=True,key=f"qc_{lbl}"):
            execute_cmd(cmd); st.rerun()

with check_col:
    st.markdown("### ✓ Auto-Checker")
    st.caption("Compares simulated C vs np.dot(A,B). Auto-runs when simulation completes.")

    sim_done2=(st.session_state.C_result is not None and
               len(st.session_state.sim_steps)>0 and
               st.session_state.current_step>=len(st.session_state.sim_steps)-1)

    if sim_done2 and st.session_state.python_C is None:
        py_C,auto_cmd,auto_out=auto_numpy(A_full,B_full)
        if py_C is not None:
            sim=st.session_state.C_result
            r=min(py_C.shape[0],sim.shape[0]); c=min(py_C.shape[1],sim.shape[1])
            st.session_state.python_C=py_C[:r,:c]
            st.session_state.term_ns["C"]=py_C
            st.session_state.terminal_history.append({"cmd":f"# [auto] {auto_cmd}","out":auto_out,"err":""})
            res,det=run_checker(sim,py_C)
            st.session_state.checker_result=res
            st.session_state.checker_detail=det

    if st.button("⟳ Re-run Checker",use_container_width=True):
        py_C,_,_=auto_numpy(A_full,B_full)
        if py_C is not None:
            sim=st.session_state.C_result
            if sim is not None:
                r=min(py_C.shape[0],sim.shape[0]); c=min(py_C.shape[1],sim.shape[1])
                st.session_state.python_C=py_C[:r,:c]
                res,det=run_checker(sim,py_C)
                st.session_state.checker_result=res
                st.session_state.checker_detail=det

    result=st.session_state.checker_result
    detail=st.session_state.checker_detail
    st.markdown("")
    if result=="pass":
        st.markdown(f'<div class="checker-pass">✅ &nbsp;<strong>PASS</strong> — Simulation correct<br>'
                    f'<span style="font-size:11px;opacity:.85">{detail}</span></div>',unsafe_allow_html=True)
    elif result=="fail":
        st.markdown(f'<div class="checker-fail">❌ &nbsp;<strong>FAIL</strong> — Mismatch detected<br>'
                    f'<span style="font-size:11px;opacity:.85">{detail}</span></div>',unsafe_allow_html=True)
    else:
        st.markdown('<div class="checker-pending">⏳ &nbsp;Run the simulation to auto-verify</div>',
                    unsafe_allow_html=True)

    if st.session_state.C_result is not None and st.session_state.python_C is not None:
        st.markdown("")
        c_sim=st.session_state.C_result
        c_py=st.session_state.python_C
        sN=min(c_sim.shape[0],6); sP=min(c_sim.shape[1],6)
        comp=st.columns(2)
        with comp[0]:
            st.markdown("**Simulated**")
            for i in range(sN):
                st.markdown(" | ".join(f"`{fmt_val(c_sim[i,j])}`" for j in range(sP)))
        with comp[1]:
            st.markdown("**numpy dot**")
            for i in range(sN):
                st.markdown(" | ".join(f"`{fmt_val(c_py[i,j])}`" for j in range(sP)))
        if c_sim.shape[0]>6 or c_sim.shape[1]>6:
            st.caption(f"({sN}×{sP} of {c_sim.shape[0]}×{c_sim.shape[1]})")

        diff=np.abs(c_sim[:sN,:sP]-c_py[:sN,:sP])
        fig2,ax2=plt.subplots(figsize=(3.8,3.2))
        fig2.patch.set_facecolor("#0e1117"); ax2.set_facecolor("#0e1117")
        im=ax2.imshow(diff,cmap="RdYlGn_r" if result=="fail" else "Greens_r",
                      aspect="auto",vmin=0)
        ax2.set_title(f"|sim − numpy|  max={diff.max():.2e}",
                      color="#8b949e",fontsize=8,pad=5)
        ax2.tick_params(colors="#4a5568",labelsize=6)
        for sp in ax2.spines.values(): sp.set_edgecolor("#2d3748")
        cbar=plt.colorbar(im,ax=ax2,fraction=0.046)
        cbar.ax.tick_params(colors="#4a5568",labelsize=6)
        plt.tight_layout(pad=0.4)
        st.pyplot(fig2,use_container_width=True)
        plt.close(fig2)