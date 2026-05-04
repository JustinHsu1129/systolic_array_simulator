import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import time
import io
import sys
import traceback

st.set_page_config(page_title="Systolic Array Simulator", layout="wide", page_icon="⬛")

st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    .title { font-family: monospace; font-size: 22px; font-weight: 600; color: #e2e8f0; letter-spacing: 0.05em; }
    .subtitle { font-family: monospace; font-size: 12px; color: #718096; margin-top: -8px; }
    .mode-desc { font-family: monospace; font-size: 12px; color: #a0aec0; font-style: italic; padding: 8px 0; }
    div[data-testid="metric-container"] { background: #1a202c; border-radius: 8px; padding: 8px; border: 0.5px solid #2d3748; }
    .step-log { font-family: monospace; font-size: 11px; background: #1a202c; padding: 8px; border-radius: 6px; color: #a0aec0; max-height: 160px; overflow-y: auto; }
    .terminal-wrap { background: #0d1117; border: 1px solid #30363d; border-radius: 8px; overflow: hidden; }
    .terminal-bar { background: #161b22; padding: 8px 14px; font-family: monospace; font-size: 11px; color: #8b949e; border-bottom: 1px solid #30363d; }
    .terminal-body { background: #0d1117; padding: 12px 14px; font-family: 'Courier New', monospace; font-size: 12px; color: #e6edf3; min-height: 100px; white-space: pre-wrap; word-break: break-all; line-height: 1.6; }
    .t-prompt { color: #388bfd; }
    .t-out    { color: #7ee787; }
    .t-err    { color: #ff7b72; }
    .checker-pass    { background: #1c3a2a; border: 1px solid #2ea043; border-radius: 8px; padding: 12px 16px; font-family: monospace; font-size: 13px; color: #7ee787; }
    .checker-fail    { background: #3a1c1c; border: 1px solid #da3633; border-radius: 8px; padding: 12px 16px; font-family: monospace; font-size: 13px; color: #ff7b72; }
    .checker-pending { background: #1c2333; border: 1px solid #30363d; border-radius: 8px; padding: 12px 16px; font-family: monospace; font-size: 13px; color: #8b949e; }
</style>
""", unsafe_allow_html=True)

# ── State init ───────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "mode": "weight_stationary",
        "N": 3,
        "active_rows": 3,
        "active_cols": 3,
        "sim_steps": [],
        "current_step": -1,
        "C_result": None,
        "log": [],
        "running": False,
        "terminal_history": [],
        "python_C": None,
        "checker_result": None,
        "checker_detail": "",
        "term_ns": {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# Always keep numpy available in terminal namespace
if "np" not in st.session_state.term_ns:
    st.session_state.term_ns["np"] = np

# ── Mode config ──────────────────────────────────────────────────────────────
MODE_DESC = {
    "weight_stationary": "Weights stay fixed in each PE. Input rows stream east, outputs accumulate in place. Classic TPU v1 style.",
    "input_stationary":  "Inputs stay fixed in each PE. Weight columns stream south, partial sums flow east.",
    "output_stationary": "Partial sums stay in each PE. Inputs stream east, weights stream south simultaneously.",
    "row_stationary":    "Each PE row holds one row of A (stationary). Weights stream south, results drain east. Minimises data movement.",
}
MODE_LABELS = {
    "weight_stationary": "Weight Stationary",
    "input_stationary":  "Input Stationary",
    "output_stationary": "Output Stationary",
    "row_stationary":    "Row Stationary",
}

# ── Helpers ──────────────────────────────────────────────────────────────────
def fmt_val(v):
    f = float(v)
    return str(int(f)) if f == int(f) else f"{f:.2g}"

# ── Simulation engine ────────────────────────────────────────────────────────
def build_sim_steps(A, B, pe_overrides, mode, N, active_rows=None, active_cols=None):
    # Pad A rows beyond active_rows with zeros; pad B cols beyond active_cols with zeros
    if active_rows is None: active_rows = N
    if active_cols is None: active_cols = N

    A_padded = A.copy()
    B_padded = B.copy()
    for i in range(N):
        if i >= active_rows:
            A_padded[i, :] = 0.0   # entire row zeroed → no west inputs on this lane
    for j in range(N):
        if j >= active_cols:
            B_padded[:, j] = 0.0   # entire col zeroed → no north inputs on this lane

    acc   = np.zeros((N, N))
    steps = []
    for t in range(3 * N - 1):
        a_regs = [[None]*N for _ in range(N)]
        b_regs = [[None]*N for _ in range(N)]
        fired  = []
        for i in range(N):
            for j in range(N):
                k  = t - i - j
                # Show None (no arrow/dot) for padded-out lanes so the viz stays clean
                av = float(A_padded[i][k]) if (0 <= k < N and i < active_rows) else (
                     0.0                   if (0 <= k < N and i >= active_rows) else None)
                bv = float(B_padded[k][j]) if (0 <= k < N and j < active_cols) else (
                     0.0                   if (0 <= k < N and j >= active_cols) else None)
                # Keep None for display but treat 0-padded as 0 for multiply
                av_vis = av  # used for display
                bv_vis = bv
                a_regs[i][j] = av_vis
                b_regs[i][j] = bv_vis
                if av is not None and bv is not None:
                    ov = pe_overrides[i][j]
                    if mode == "weight_stationary":
                        mult = av * (ov if ov is not None else bv)
                    elif mode == "input_stationary":
                        mult = (ov if ov is not None else av) * bv
                    elif mode == "output_stationary":
                        mult = av * bv
                    else:  # row_stationary
                        mult = (ov if ov is not None else av) * bv
                    acc[i][j] += mult
                    if mult != 0 or (i < active_rows and j < active_cols):
                        fired.append({"i": i, "j": j, "av": av, "bv": bv,
                                      "mult": mult, "acc": acc[i][j],
                                      "padded": (i >= active_rows or j >= active_cols)})
        steps.append({
            "t": t,
            "a_regs": [row[:] for row in a_regs],
            "b_regs": [row[:] for row in b_regs],
            "acc": acc.copy(),
            "fired": fired,
            "active_rows": active_rows,
            "active_cols": active_cols,
        })
    return steps, acc.copy()

# ── Drawing ──────────────────────────────────────────────────────────────────
def draw_array(state, N, pe_overrides, mode_name, active_rows=None, active_cols=None):
    if active_rows is None: active_rows = N
    if active_cols is None: active_cols = N

    DARK = "#0e1117"; CELL = "#1a202c"; BORD = "#2d3748"; DIM = "#4a5568"
    A_COL = "#4299e1"; B_COL = "#ed8936"; ACC_COL = "#48bb78"
    STAT_COL = "#fc8181"; ACT_CELL = "#1c3a2a"; TXT = "#e2e8f0"
    PAD_CELL = "#141414"; PAD_BORD = "#1e2430"   # muted style for zero-padded PEs

    if   N <= 4:  cell, gap, pad = 1.00, 0.22, 1.20; fs_pe,fs_v,fs_a = 6.5,7.0,8.0
    elif N <= 6:  cell, gap, pad = 0.80, 0.18, 1.10; fs_pe,fs_v,fs_a = 5.5,6.0,7.0
    elif N <= 8:  cell, gap, pad = 0.62, 0.14, 1.00; fs_pe,fs_v,fs_a = 4.5,5.0,6.0
    elif N <= 11: cell, gap, pad = 0.50, 0.11, 0.90; fs_pe,fs_v,fs_a = 4.0,4.5,5.0
    else:         cell, gap, pad = 0.38, 0.09, 0.80; fs_pe,fs_v,fs_a = 3.0,3.5,4.0

    step    = cell + gap
    total_w = N * step - gap + pad * 2
    total_h = N * step - gap + pad * 2
    fw = min(max(total_w * 1.05, 5), 16)
    fh = min(max(total_h * 1.05, 4), 16)

    fig, ax = plt.subplots(figsize=(fw, fh))
    fig.patch.set_facecolor(DARK)
    ax.set_facecolor(DARK)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_xlim(-0.5, total_w + 0.3)
    ax.set_ylim(-0.5, total_h + 0.3)

    for i in range(N):
        for j in range(N):
            cx = pad + j * step
            cy = pad + (N - 1 - i) * step
            av  = state["a_regs"][i][j] if state else None
            bv  = state["b_regs"][i][j] if state else None
            acc = state["acc"][i][j]    if state else 0.0
            active = state and any(f["i"]==i and f["j"]==j and not f.get("padded") for f in state["fired"])

            is_padded_row = (i >= active_rows)
            is_padded_col = (j >= active_cols)
            is_padded     = is_padded_row or is_padded_col

            fc = ACT_CELL if (active and not is_padded) else (PAD_CELL if is_padded else CELL)
            ec = ACC_COL  if (active and not is_padded) else (PAD_BORD if is_padded else BORD)
            lw = 1.2      if (active and not is_padded) else 0.5

            rect = mpatches.FancyBboxPatch(
                (cx - cell/2, cy - cell/2), cell, cell,
                boxstyle="round,pad=0.03", linewidth=lw,
                edgecolor=ec, facecolor=fc, zorder=2,
                alpha=0.45 if is_padded else 1.0)
            ax.add_patch(rect)

            # Hatching for padded cells to make it crystal clear
            if is_padded:
                hatch = mpatches.FancyBboxPatch(
                    (cx - cell/2, cy - cell/2), cell, cell,
                    boxstyle="round,pad=0.03", linewidth=0,
                    edgecolor="#2d3748", facecolor="none",
                    hatch="////", zorder=2, alpha=0.25)
                ax.add_patch(hatch)

            # "0" label for padded cells
            if is_padded and N <= 10:
                ax.text(cx, cy, "0",
                        ha="center", va="center", fontsize=fs_a,
                        color="#2d3748", fontfamily="monospace", zorder=3, style="italic")

            if N <= 10 and not is_padded:
                ax.text(cx, cy + cell/2 - 0.09, f"PE{i},{j}",
                        ha="center", va="top", fontsize=fs_pe,
                        color=DIM, fontfamily="monospace", zorder=3)
            elif N <= 10 and is_padded:
                ax.text(cx, cy + cell/2 - 0.09, f"PE{i},{j}",
                        ha="center", va="top", fontsize=fs_pe,
                        color="#252a36", fontfamily="monospace", zorder=3)

            if N <= 7 and not is_padded:
                if av is not None and av != 0.0:
                    ax.text(cx - cell/2 + 0.05, cy + cell*0.15,
                            f"a:{fmt_val(av)}", ha="left", va="center",
                            fontsize=fs_v, color=A_COL, fontfamily="monospace", zorder=3)
                if bv is not None and bv != 0.0:
                    ax.text(cx + cell/2 - 0.05, cy + cell*0.15,
                            f"b:{fmt_val(bv)}", ha="right", va="center",
                            fontsize=fs_v, color=B_COL, fontfamily="monospace", zorder=3)

            dot_ms = max(1.5, 5 - N // 3)
            if av is not None and not is_padded_row:
                ax.plot(cx - cell/2, cy, "o", ms=dot_ms, color=A_COL, alpha=0.85, zorder=4)
            if bv is not None and not is_padded_col:
                ax.plot(cx, cy + cell/2, "o", ms=dot_ms, color=B_COL, alpha=0.85, zorder=4)

            if not is_padded:
                lbl = ("Σ" if N <= 10 else "") + fmt_val(acc)
                ax.text(cx, cy - cell/2 + 0.10, lbl,
                        ha="center", va="bottom", fontsize=fs_a, fontweight="bold",
                        color=ACC_COL if active else DIM,
                        fontfamily="monospace", zorder=3)

            ov = pe_overrides[i][j]
            if ov is not None and N <= 8 and not is_padded:
                ax.text(cx + cell/2 - 0.04, cy + cell/2 - 0.07, "★",
                        ha="right", va="top", fontsize=fs_pe,
                        color=STAT_COL, fontfamily="monospace", zorder=3)

    for i in range(N):
        cy       = pad + (N - 1 - i) * step
        is_pad_r = (i >= active_rows)
        row_col  = "#2d3748" if is_pad_r else A_COL
        ax.annotate("", xy=(pad - cell/2 - 0.04, cy),
                    xytext=(pad - cell/2 - 0.32, cy),
                    arrowprops=dict(arrowstyle="->", color=row_col, lw=0.8,
                                   alpha=0.35 if is_pad_r else 1.0))
        if N <= 13:
            lbl = f"A[{i}]" + (" [0]" if is_pad_r else "")
            ax.text(pad - cell/2 - 0.35, cy, lbl,
                    ha="right", va="center", fontsize=max(4.5, fs_pe),
                    color=row_col, fontfamily="monospace",
                    alpha=0.45 if is_pad_r else 1.0)

    for j in range(N):
        cx        = pad + j * step
        top_cy    = pad + (N - 1) * step
        is_pad_c  = (j >= active_cols)
        col_color = "#2d3748" if is_pad_c else B_COL
        ax.annotate("", xy=(cx, top_cy + cell/2 + 0.04),
                    xytext=(cx, top_cy + cell/2 + 0.30),
                    arrowprops=dict(arrowstyle="->", color=col_color, lw=0.8,
                                   alpha=0.35 if is_pad_c else 1.0))
        if N <= 13:
            lbl = f"B[{j}]" + (" [0]" if is_pad_c else "")
            ax.text(cx, top_cy + cell/2 + 0.34, lbl,
                    ha="center", va="bottom", fontsize=max(4.5, fs_pe),
                    color=col_color, fontfamily="monospace",
                    alpha=0.45 if is_pad_c else 1.0)

    ax.text(total_w + 0.1, 0, mode_name.replace("_", " "),
            ha="right", va="bottom", fontsize=6, color=DIM,
            fontfamily="monospace", style="italic")

    legend_items = [
        mpatches.Patch(color=A_COL,    label="A input (east)"),
        mpatches.Patch(color=B_COL,    label="B weight (south)"),
        mpatches.Patch(color=ACC_COL,  label="accumulator"),
        mpatches.Patch(color=STAT_COL, label="override"),
    ]
    ax.legend(handles=legend_items, loc="lower right", fontsize=6,
              facecolor="#1a202c", edgecolor="#2d3748", labelcolor=TXT,
              framealpha=1, handlelength=1)
    plt.tight_layout(pad=0.3)
    return fig

# ── Terminal executor ────────────────────────────────────────────────────────
def run_python(cmd: str, ns: dict) -> tuple:
    stdout_buf = io.StringIO(); stderr_buf = io.StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = stdout_buf, stderr_buf
    try:
        try:
            result = eval(compile(cmd, "<term>", "eval"), ns)
            if result is not None:
                print(repr(result))
        except SyntaxError:
            exec(compile(cmd, "<term>", "exec"), ns)
    except Exception:
        print(traceback.format_exc(), file=sys.stderr)
    finally:
        sys.stdout, sys.stderr = old_out, old_err
    return stdout_buf.getvalue(), stderr_buf.getvalue()

def auto_numpy(A, B):
    ns  = {"np": np, "A": A.copy(), "B": B.copy()}
    cmd = "C = np.dot(A, B)\nprint(C)"
    out, err = run_python(cmd, ns)
    return ns.get("C"), cmd, out

def run_checker(sim_C, py_C):
    if sim_C is None or py_C is None:
        return "pending", "Waiting for both results..."
    try:
        if np.allclose(sim_C, py_C, atol=1e-6):
            return "pass", (f"All {sim_C.size} elements match  |  "
                            f"max |err| = {np.max(np.abs(sim_C - py_C)):.2e}")
        bad = np.argwhere(~np.isclose(sim_C, py_C, atol=1e-6))
        sample = ", ".join(
            f"C[{r},{c}]: sim={sim_C[r,c]:.4g} np={py_C[r,c]:.4g}"
            for r, c in bad[:4])
        return "fail", f"{len(bad)} mismatch(es) — {sample}"
    except Exception as e:
        return "fail", str(e)

# ════════════════════════════════════════════════════════════════════════════
# Sidebar
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### Configuration")

    mode_key = st.selectbox(
        "Dataflow mode",
        options=list(MODE_DESC.keys()),
        format_func=lambda k: MODE_LABELS[k],
        index=list(MODE_DESC.keys()).index(st.session_state.mode),
    )
    st.session_state.mode = mode_key
    st.markdown(f'<div class="mode-desc">{MODE_DESC[mode_key]}</div>', unsafe_allow_html=True)

    st.markdown("---")
    N = st.selectbox(
        "Array size",
        options=list(range(2, 16)),
        index=list(range(2, 16)).index(st.session_state.N),
        format_func=lambda n: f"{n}×{n}",
    )
    if N != st.session_state.N:
        st.session_state.N = N
        st.session_state.active_rows = N
        st.session_state.active_cols = N
        for k in ["sim_steps","log","terminal_history"]:
            st.session_state[k] = []
        for k in ["current_step"]:
            st.session_state[k] = -1
        for k in ["C_result","python_C","checker_result","checker_detail"]:
            st.session_state[k] = None
        st.session_state.checker_detail = ""
        st.session_state.running = False
        if "_rand_A" in st.session_state: del st.session_state["_rand_A"]
        if "_rand_B" in st.session_state: del st.session_state["_rand_B"]

    st.markdown("**Active inputs (others zero-padded)**")
    c1, c2 = st.columns(2)
    active_rows = c1.slider("A rows →", 1, N, min(st.session_state.active_rows, N), key="ar_slider")
    active_cols = c2.slider("B cols ↓", 1, N, min(st.session_state.active_cols, N), key="ac_slider")
    st.session_state.active_rows = active_rows
    st.session_state.active_cols = active_cols
    if active_rows < N or active_cols < N:
        st.caption(f"🔲 {active_rows} active row(s) west · {active_cols} active col(s) north · rest → 0")

    st.markdown("---")

    def default_A(N):
        return np.fromfunction(lambda i, j: (i * N + j + 1) % 9 + 1, (N, N), dtype=float)
    def default_B(N):
        return (np.eye(N) + np.fromfunction(lambda i, j: (i+j) % 3, (N, N), dtype=float) * 0.5)

    if "_rand_A" in st.session_state:
        A_init = st.session_state["_rand_A"]
        B_init = st.session_state["_rand_B"]
    else:
        A_init = default_A(N)
        B_init = default_B(N)

    st.markdown("**Input matrix A**")
    if N <= 5:
        A_rows = []
        for i in range(N):
            cols = st.columns(N)
            row  = [cols[j].number_input(f"A{i}{j}", value=float(A_init[i,j]),
                                          label_visibility="collapsed",
                                          key=f"a_{i}_{j}", step=1.0) for j in range(N)]
            A_rows.append(row)
        A = np.array(A_rows, dtype=float)
    else:
        raw = st.text_area("A (space-separated rows)", key="a_raw",
                           value="\n".join(" ".join(str(int(A_init[i,j])) for j in range(N)) for i in range(N)),
                           height=min(220, N * 20))
        try:
            rows = [list(map(float, r.split())) for r in raw.strip().splitlines() if r.strip()]
            A = np.array(rows, dtype=float) if (len(rows)==N and all(len(r)==N for r in rows)) else A_init
        except Exception:
            A = A_init

    st.markdown("**Weight matrix B**")
    if N <= 5:
        B_rows = []
        for i in range(N):
            cols = st.columns(N)
            row  = [cols[j].number_input(f"B{i}{j}", value=float(B_init[i,j]),
                                          label_visibility="collapsed",
                                          key=f"b_{i}_{j}", step=1.0) for j in range(N)]
            B_rows.append(row)
        B = np.array(B_rows, dtype=float)
    else:
        raw = st.text_area("B (space-separated rows)", key="b_raw",
                           value="\n".join(" ".join(str(int(B_init[i,j])) for j in range(N)) for i in range(N)),
                           height=min(220, N * 20))
        try:
            rows = [list(map(float, r.split())) for r in raw.strip().splitlines() if r.strip()]
            B = np.array(rows, dtype=float) if (len(rows)==N and all(len(r)==N for r in rows)) else B_init
        except Exception:
            B = B_init

    if st.button("🎲 Randomise A & B", use_container_width=True):
        rng = np.random.default_rng()
        st.session_state["_rand_A"] = rng.integers(1, 9, (N, N)).astype(float)
        st.session_state["_rand_B"] = rng.integers(1, 9, (N, N)).astype(float)
        st.rerun()

    st.markdown("---")
    st.markdown("**PE Stationary Overrides**")
    st.caption("Leave 0 = auto.")
    pe_overrides = [[None]*N for _ in range(N)]
    if N <= 5:
        for i in range(N):
            cols = st.columns(N)
            for j in range(N):
                v = cols[j].number_input(f"PE{i}{j}", value=0.0,
                                          label_visibility="collapsed",
                                          key=f"pe_{i}_{j}", step=1.0)
                pe_overrides[i][j] = v if v != 0.0 else None
    else:
        st.caption("PE overrides available for arrays ≤ 5×5.")

    st.markdown("---")
    speed = st.slider("Sim speed (ms/step)", 50, 1500, 400, step=50)

# Always sync terminal namespace with current A/B
st.session_state.term_ns["A"] = A.copy()
st.session_state.term_ns["B"] = B.copy()
if st.session_state.C_result is not None:
    st.session_state.term_ns["C_sim"] = st.session_state.C_result.copy()

# ════════════════════════════════════════════════════════════════════════════
# Main area
# ════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="title">Systolic Array Simulator</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Interactive matrix multiply · configurable dataflow · PE stationary values · Python verifier</div>', unsafe_allow_html=True)
st.markdown("---")

col_vis, col_ctrl = st.columns([3, 1])

with col_ctrl:
    st.markdown("### Controls")

    if st.button("▶ Simulate", use_container_width=True, type="primary"):
        steps, C = build_sim_steps(A, B, pe_overrides, mode_key, N, active_rows, active_cols)
        st.session_state.sim_steps      = steps
        st.session_state.current_step   = -1
        st.session_state.C_result       = C
        st.session_state.log            = []
        st.session_state.running        = True
        st.session_state.python_C       = None
        st.session_state.checker_result = None
        st.session_state.checker_detail = ""

    if st.button("→ Step", use_container_width=True):
        if not st.session_state.sim_steps:
            steps, C = build_sim_steps(A, B, pe_overrides, mode_key, N, active_rows, active_cols)
            st.session_state.sim_steps  = steps
            st.session_state.C_result   = C
            st.session_state.log        = []
        if st.session_state.current_step < len(st.session_state.sim_steps) - 1:
            st.session_state.current_step += 1

    if st.button("⚡ Finish Now", use_container_width=True):
        if not st.session_state.sim_steps:
            steps, C = build_sim_steps(A, B, pe_overrides, mode_key, N, active_rows, active_cols)
            st.session_state.sim_steps  = steps
            st.session_state.C_result   = C
            st.session_state.log        = []
        # Jump to last step and build full log instantly
        last = len(st.session_state.sim_steps) - 1
        st.session_state.current_step = last
        st.session_state.running      = False
        # Populate the full log in one shot
        full_log = []
        for s in st.session_state.sim_steps:
            for f in s["fired"]:
                full_log.append(
                    f"t={s['t']} | PE({f['i']},{f['j']}): "
                    f"{fmt_val(f['av'])}×{fmt_val(f['bv'])}={fmt_val(f['mult'])} → Σ={fmt_val(f['acc'])}"
                )
        st.session_state.log = full_log

    if st.button("↺ Reset", use_container_width=True):
        for k in ["sim_steps","log","terminal_history"]:
            st.session_state[k] = []
        st.session_state.current_step   = -1
        st.session_state.C_result       = None
        st.session_state.running        = False
        st.session_state.python_C       = None
        st.session_state.checker_result = None
        st.session_state.checker_detail = ""

    st.markdown("---")
    total = len(st.session_state.sim_steps)
    cur   = st.session_state.current_step
    st.metric("Step",  f"{max(cur,0)}/{total}" if total else "—")
    st.metric("Mode",  MODE_LABELS[mode_key])
    st.metric("Array", f"{N}×{N}")
    st.metric("Cycles", f"{3*N-1}")

    sim_done = (st.session_state.C_result is not None and total > 0 and cur >= total - 1)

    if sim_done:
        st.markdown("**Simulated C**")
        C_disp  = st.session_state.C_result
        show_N  = min(N, 7)
        for i in range(show_N):
            cols = st.columns(show_N)
            for j in range(show_N):
                cols[j].markdown(
                    f"<div style='background:#1c3a2a;border-radius:5px;padding:3px;"
                    f"text-align:center;font-family:monospace;font-size:10px;"
                    f"color:#48bb78;font-weight:600'>{fmt_val(C_disp[i,j])}</div>",
                    unsafe_allow_html=True)
        if N > show_N:
            st.caption(f"({show_N}×{show_N} of {N}×{N})")

# ── Auto-advance ──────────────────────────────────────────────────────────────
if st.session_state.running and st.session_state.sim_steps:
    if st.session_state.current_step < len(st.session_state.sim_steps) - 1:
        st.session_state.current_step += 1
    else:
        st.session_state.running = False

with col_vis:
    state = None
    if st.session_state.sim_steps and st.session_state.current_step >= 0:
        idx   = min(st.session_state.current_step, len(st.session_state.sim_steps) - 1)
        state = st.session_state.sim_steps[idx]
        for f in state["fired"]:
            if f.get("padded"):
                continue   # skip zero-padded firings from the log
            entry = (f"t={state['t']} | PE({f['i']},{f['j']}): "
                     f"{fmt_val(f['av'])}×{fmt_val(f['bv'])}={fmt_val(f['mult'])} → Σ={fmt_val(f['acc'])}")
            if entry not in st.session_state.log:
                st.session_state.log.append(entry)

    # Use active_rows/active_cols from current state snapshot if available, else sidebar values
    vis_ar = state["active_rows"] if state and "active_rows" in state else active_rows
    vis_ac = state["active_cols"] if state and "active_cols" in state else active_cols

    fig = draw_array(state, N, pe_overrides, mode_key, vis_ar, vis_ac)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    if st.session_state.log:
        st.markdown("**Simulation log**")
        cur_t    = state["t"] if state else -1
        log_html = "<br>".join(
            f'<span class="t-prompt">{l}</span>' if f"t={cur_t} |" in l
            else f'<span style="color:#4a5568">{l}</span>'
            for l in st.session_state.log[-35:]
        )
        st.markdown(f'<div class="step-log">{log_html}</div>', unsafe_allow_html=True)

if st.session_state.running:
    time.sleep(speed / 1000)
    st.rerun()

# ════════════════════════════════════════════════════════════════════════════
# Python Terminal + Checker
# ════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("## Python Verifier")
st.caption("Interactive Python terminal pre-loaded with A, B, C_sim. "
           "The checker auto-runs `np.dot(A, B)` when the simulation finishes and compares results.")

term_col, check_col = st.columns([3, 2])

with term_col:
    # Terminal chrome
    st.markdown("""
    <div class="terminal-wrap">
      <div class="terminal-bar">
        <span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#ff5f57;margin-right:4px"></span>
        <span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#febc2e;margin-right:4px"></span>
        <span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#28c840;margin-right:8px"></span>
        python3 &nbsp;·&nbsp; numpy &nbsp;·&nbsp; systolic_verifier
      </div>
    </div>""", unsafe_allow_html=True)

    # Build terminal body
    lines = []
    if not st.session_state.terminal_history:
        lines.append('<span class="t-prompt"># A, B are loaded from the simulator  |  C_sim available after simulation</span>')
        lines.append('<span class="t-prompt">&gt;&gt;&gt; </span>')
    else:
        for entry in st.session_state.terminal_history:
            for ln in entry["cmd"].splitlines():
                lines.append(f'<span class="t-prompt">&gt;&gt;&gt; {ln}</span>')
            if entry["out"]:
                lines.append(f'<span class="t-out">{entry["out"].rstrip()}</span>')
            if entry["err"]:
                lines.append(f'<span class="t-err">{entry["err"].rstrip()}</span>')
        lines.append('<span class="t-prompt">&gt;&gt;&gt; </span>')

    st.markdown(
        f'<div class="terminal-wrap"><div class="terminal-body">{"<br>".join(lines)}</div></div>',
        unsafe_allow_html=True)

    # Input row
    c1, c2, c3 = st.columns([5, 1, 1])
    cmd_input = c1.text_input("cmd", value="", label_visibility="collapsed",
                               placeholder=">>> type Python here  (np, A, B, C_sim available)",
                               key="term_input")
    run_btn = c2.button("Run ▶", use_container_width=True)
    clr_btn = c3.button("Clear", use_container_width=True)

    if clr_btn:
        st.session_state.terminal_history = []
        st.rerun()

    def execute_cmd(cmd):
        out, err = run_python(cmd, st.session_state.term_ns)
        st.session_state.terminal_history.append({"cmd": cmd, "out": out, "err": err})
        # If user computed a matrix called C, run the checker against it
        if "C" in st.session_state.term_ns:
            try:
                candidate = np.array(st.session_state.term_ns["C"], dtype=float)
                if candidate.shape == (N, N):
                    st.session_state.python_C = candidate
                    res, det = run_checker(st.session_state.C_result, candidate)
                    st.session_state.checker_result = res
                    st.session_state.checker_detail = det
            except Exception:
                pass

    if run_btn and cmd_input.strip():
        execute_cmd(cmd_input)
        st.rerun()

    # Quick commands
    st.markdown("**Quick commands:**")
    qcols = st.columns(5)
    quick = [
        ("np.dot(A,B)",  "C = np.dot(A, B)\nprint(C)"),
        ("A @ B",        "C = A @ B\nprint(C)"),
        ("print A",      "print(A)"),
        ("print B",      "print(B)"),
        ("C_sim",        "print(C_sim)" if st.session_state.C_result is not None
                         else "print('run simulation first')"),
    ]
    for col, (label, cmd) in zip(qcols, quick):
        if col.button(label, use_container_width=True, key=f"qc_{label}"):
            execute_cmd(cmd)
            st.rerun()

with check_col:
    st.markdown("### ✓ Auto-Checker")
    st.caption("Compares simulated C vs `np.dot(A, B)`. Runs automatically when simulation completes.")

    # Auto-trigger when simulation finishes and checker hasn't run yet
    sim_done = (
        st.session_state.C_result is not None and
        len(st.session_state.sim_steps) > 0 and
        st.session_state.current_step >= len(st.session_state.sim_steps) - 1
    )
    if sim_done and st.session_state.python_C is None:
        py_C, auto_cmd, auto_out = auto_numpy(A, B)
        st.session_state.python_C = py_C
        st.session_state.term_ns["C"] = py_C
        st.session_state.terminal_history.append({
            "cmd": f"# [auto] {auto_cmd}",
            "out": auto_out, "err": ""
        })
        res, det = run_checker(st.session_state.C_result, py_C)
        st.session_state.checker_result = res
        st.session_state.checker_detail = det

    if st.button("⟳ Re-run Checker", use_container_width=True):
        py_C, auto_cmd, auto_out = auto_numpy(A, B)
        st.session_state.python_C = py_C
        st.session_state.term_ns["C"] = py_C
        res, det = run_checker(st.session_state.C_result, py_C)
        st.session_state.checker_result = res
        st.session_state.checker_detail = det

    st.markdown("")
    result = st.session_state.checker_result
    detail = st.session_state.checker_detail

    if result == "pass":
        st.markdown(f"""<div class="checker-pass">
            ✅ &nbsp;<strong>PASS</strong> — Simulation is correct<br>
            <span style="font-size:11px;opacity:0.85">{detail}</span>
        </div>""", unsafe_allow_html=True)
    elif result == "fail":
        st.markdown(f"""<div class="checker-fail">
            ❌ &nbsp;<strong>FAIL</strong> — Mismatch detected<br>
            <span style="font-size:11px;opacity:0.85">{detail}</span>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""<div class="checker-pending">
            ⏳ &nbsp;Run the simulation to auto-verify
        </div>""", unsafe_allow_html=True)

    # Side-by-side matrix comparison
    if st.session_state.C_result is not None and st.session_state.python_C is not None:
        st.markdown("")
        show_N = min(N, 6)
        c_sim = st.session_state.C_result
        c_py  = st.session_state.python_C

        comp_cols = st.columns(2)
        with comp_cols[0]:
            st.markdown("**Simulated**")
            for i in range(show_N):
                row_str = " | ".join(
                    f"`{fmt_val(c_sim[i,j])}`" for j in range(show_N))
                st.markdown(row_str)

        with comp_cols[1]:
            st.markdown("**numpy dot**")
            for i in range(show_N):
                row_str = " | ".join(
                    f"`{fmt_val(c_py[i,j])}`" for j in range(show_N))
                st.markdown(row_str)

        if N > show_N:
            st.caption(f"Showing {show_N}×{show_N} of {N}×{N}")

        # Diff heatmap
        st.markdown("**|sim − numpy| heatmap**")
        diff = np.abs(c_sim - c_py)
        cmap = "RdYlGn_r" if result == "fail" else "Greens_r"
        fig2, ax2 = plt.subplots(figsize=(3.8, 3.2))
        fig2.patch.set_facecolor("#0e1117")
        ax2.set_facecolor("#0e1117")
        im = ax2.imshow(diff, cmap=cmap, aspect="auto", vmin=0)
        ax2.set_title(
            f"max err = {diff.max():.2e}",
            color="#8b949e", fontsize=8, pad=5)
        ax2.tick_params(colors="#4a5568", labelsize=6)
        for sp in ax2.spines.values():
            sp.set_edgecolor("#2d3748")
        cbar = plt.colorbar(im, ax=ax2, fraction=0.046)
        cbar.ax.tick_params(colors="#4a5568", labelsize=6)
        plt.tight_layout(pad=0.4)
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)