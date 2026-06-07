"""
Generate the full ALPS-4B architecture diagram (figures/alps4b_architecture_full.png).

Shows EVERY concept in the architecture — including the not-yet-validated ones —
each tagged with its empirical status so the figure is precise and honest:

    ✓ validated          (a gate / ablation demonstrates it)
    ◐ partial            (works but below target, or a proxy was demonstrated)
    ○ design hypothesis  (implemented but not yet validated end-to-end)

Pure matplotlib (no graphviz dependency). Run:
    python docs/make_architecture_diagram.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

ST = {
    "val":  ("#cfe8cf", "#2e7d32", "✓"),
    "part": ("#ffe6b3", "#e07b00", "◐"),
    "hyp":  ("#ececec", "#8a8a8a", "○"),
    "io":   ("#dbe9f6", "#2b6cb0", ""),
}

fig, ax = plt.subplots(figsize=(20, 13.5))
ax.set_xlim(0, 20); ax.set_ylim(0, 13.5); ax.axis("off")


def band(x, y, w, h, label, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=color, ec="none", alpha=0.30, zorder=0))
    ax.text(x + 0.18, y + h - 0.22, label, fontsize=11, fontweight="bold",
            color="#222", va="top", ha="left", zorder=1)


def box(x, y, w, h, text, status, gate="", fs=8.5):
    fc, ec, tag = ST[status]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.08",
                                fc=fc, ec=ec, lw=1.6, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, fontsize=fs, ha="center", va="center", zorder=3)
    if tag:
        ax.text(x + 0.12, y + h - 0.10, tag, fontsize=10, ha="left", va="top",
                color=ec, fontweight="bold", zorder=3)
    if gate:
        ax.text(x + w - 0.12, y + 0.08, gate, fontsize=7, ha="right", va="bottom",
                color=ec, style="italic", zorder=3)
    return (x + w / 2, y + h / 2, x, y, w, h)  # cx, cy, x, y, w, h


def arrow(p_from, p_to, color="#333", style="-|>", lw=1.6, ls="-", rad=0.0, label="", lf=7.5):
    ax.add_patch(FancyArrowPatch(p_from, p_to, arrowstyle=style, mutation_scale=14, lw=lw,
                                 color=color, ls=ls, connectionstyle=f"arc3,rad={rad}", zorder=1.5))
    if label:
        ax.text((p_from[0] + p_to[0]) / 2, (p_from[1] + p_to[1]) / 2, label, fontsize=lf,
                ha="center", va="center", color=color,
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.85), zorder=4)


# ── Title + legend ──
ax.text(7.0, 13.1, "ALPS-4B — Adaptive Latent Prediction System (Four-Brain)",
        fontsize=17, fontweight="bold", ha="center")
lx = 13.7
for i, (k, lab) in enumerate([("val", "validated (gate / ablation)"),
                              ("part", "partial / proxy demonstrated"),
                              ("hyp", "design hypothesis (not yet validated)")]):
    fc, ec, tag = ST[k]
    ax.add_patch(FancyBboxPatch((lx, 13.0 - i * 0.42), 0.3, 0.28, boxstyle="round,pad=0.02",
                                fc=fc, ec=ec, lw=1.4))
    ax.text(lx + 0.42, 13.14 - i * 0.42, f"{tag}  {lab}", fontsize=8.5, va="center")

# ════════ LEFT PIPELINE (data flow bottom→top) ════════
PW = 12.8
band(0.5, 0.5, PW, 1.6, "INPUT  (vision-primary, O(1) sensors)", "#bcd")
v = box(0.9, 0.7, 3.0, 1.0, "Vision input\n16×224×224\n(always-on)", "io")
au = box(4.4, 0.75, 2.4, 0.9, "Aux sensors\nIMU / LiDAR", "hyp")
mr = box(7.3, 0.75, 3.0, 0.9, "O(1) Modality Router\nselective sensor gating", "hyp")

band(0.5, 2.4, PW, 1.6, "ENCODER  (System-1 features)", "#cdd")
enc = box(0.9, 2.6, 5.4, 1.0, "Vision Encoder — 3D ViT\ntube patches · 90% tube masking · proj head", "val")
sg0 = box(6.6, 2.6, 2.7, 1.0, "SIGReg\ncollapse prevention\n(per-row normalized)", "val")
ax.text(11.0, 3.1, "z_t  (continuous latent)", fontsize=8.5, style="italic", ha="center")

band(0.5, 4.3, PW, 1.8, "OPERATIVE — System 1  (fast, every step)", "#bcd0e8")
opp = box(2.6, 4.5, 4.0, 1.1, "Operative predictor\n(z_t, a_t) → ẑ_{t+1}", "val", "G_op / G2")
dec = box(7.0, 4.5, 2.6, 1.1, "Position decoder\nz → (x, y)", "val", "G1: 0.19 wu")
wb = box(10.0, 4.6, 2.6, 0.9, "Working-state\nbuffer (FIFO)", "hyp")
ax.text(0.95, 5.05, "action a_t →", fontsize=8.5, ha="left", color="#444")

band(0.5, 6.5, PW, 2.0, "TACTICAL — System 2  (mid horizon, sub-goals)", "#ffe2c0")
tap = box(0.85, 6.7, 3.0, 1.25, "Tactical predictor\nh_T + c_T → h_{T+k}", "val", "G_tac")
moe = box(4.05, 6.7, 2.5, 1.25, "Sparse MoE router\ntop-k experts\n(O(1) FLOPs)", "part", "O(1): ○")
rag = box(6.75, 6.7, 2.9, 1.25, "Latent-RAG\nKV cache (Δz)", "val", "G_rag")
sgg = box(9.85, 6.7, 2.7, 1.25, "Goal-conditioned\nsub-goal emission", "part", "G_goals: open")

band(0.5, 9.0, PW, 2.0, "STRATEGIC — System 2  (slow, discrete concepts)", "#e3d4f0")
stp = box(0.85, 9.2, 3.0, 1.25, "Strategic predictor\nc_T → c_{T+K}\n(long horizon)", "part")
vq = box(4.05, 9.2, 2.5, 1.25, "VQ concept\nbottleneck (c_T)\ndiscrete", "part", "G_str: 11 codes")
lg = box(6.75, 9.2, 2.9, 1.25, "Latent transition\ngraph (shortest path)", "val", "ablation 5×")
sem = box(9.85, 9.2, 2.7, 1.25, "Semantic\nvector DB", "hyp")

# data flow + stop-gradient isolation
arrow((v[0], v[3] + v[5]), (enc[0], enc[3]))
arrow((au[0], au[3] + au[5]), (mr[0], mr[3]))
arrow((mr[0], mr[3] + mr[5]), (enc[0] + 2.2, enc[3]))
arrow((enc[0], enc[3] + enc[5]), (opp[0], opp[3]), label="z_t")
arrow((opp[0] + opp[4] / 2 + 0.05, opp[1]), (dec[2], dec[1] + dec[5] / 2))
arrow((opp[0] - 0.7, opp[3] + opp[5]), (tap[0] + 1.0, tap[3]), color="#1565c0",
      label="stop-grad ↑", lf=7)
arrow((tap[0] + 1.2, tap[3] + tap[5]), (stp[0] + 1.0, stp[3]), color="#6a1b9a",
      label="stop-grad ↑", lf=7)
arrow((vq[0], vq[3]), (moe[0], moe[3] + moe[5]), color="#6a1b9a", ls=(0, (4, 2)),
      label="c_T concept", lf=7)
arrow((tap[0], tap[3]), (opp[0] - 0.75, opp[3] + opp[5]), color="#e07b00", ls=(0, (4, 2)),
      label="h_T sub-goal", lf=7, rad=-0.15)
arrow((opp[0] + opp[4], opp[1]), (13.1, opp[1]), label="ẑ_{t+1} / action →")

# ════════ RIGHT COLUMN: coordination + safety + lifelong ════════
RX, RW = 13.7, 5.9
band(RX, 6.6, RW, 4.4, "COORDINATION  (cross-layer)", "#d9d9d9")
im = box(RX + 0.15, 9.5, RW - 0.3, 1.1, "Inverse monitor (efference copy)\nsurprise D>τ → dynamic gating S1→S2", "hyp", "G6: pending")
bc = box(RX + 0.15, 8.2, RW - 0.3, 1.05, "Banach contraction checker\nrefinement loop, Lipschitz L<1", "hyp")
ebm = box(RX + 0.15, 6.9, RW - 0.3, 1.05, "EBM binding\nE = αE_S + βE_T + γE_O", "hyp")

band(RX, 4.3, RW, 2.0, "SAFETY  (out-of-gradient reflex)", "#f3c9c9")
fb = box(RX + 0.15, 4.45, RW - 0.3, 1.4,
         "Fallback watchdog\nNaN/Inf · variance collapse · hypersphere pinning\n→ MRC (Lyapunov-stable) → safe-haven / brake",
         "hyp", "G8 / G9: pending")

band(RX, 0.5, RW, 3.5, "LIFELONG SELF-LEARNING", "#cfe3cf")
sl = box(RX + 0.15, 2.45, RW - 0.3, 1.2,
         "No-retraining loop\nsurprise → write Δz → retrieve → correct\n(zero-shot, surprise-gated)", "val", "G_rag")
box(RX + 0.15, 1.45, (RW - 0.5) / 2, 0.85, "Sleep\nconsolidation", "hyp")
box(RX + 0.35 + (RW - 0.5) / 2, 1.45, (RW - 0.5) / 2, 0.85, "Hive-mind sync\n+ audit / delete", "hyp")
ax.text(RX + RW / 2, 0.85, "3-tier memory:  working (FIFO) · episodic cache · semantic DB",
        fontsize=7.6, ha="center", style="italic", color="#444")

# cross-cutting connectors
arrow((rag[0] + rag[4] / 2, rag[1] - rag[5] / 2 + rag[5]), (sl[2], sl[1] + sl[5] / 2),
      color="#2e7d32", rad=0.12)
arrow((im[2], im[1] + 0.35), (rag[0] + rag[4], rag[1] + rag[5]), color="#888",
      ls=(0, (3, 2)), rad=0.18)
arrow((fb[2], fb[1] + fb[5] / 2), (dec[0] + dec[4] / 2, dec[1] + 0.15), color="#b00020",
      ls=(0, (2, 2)), rad=0.22, label="monitor z (out-of-grad)", lf=6.5)

ax.text(7.0, 0.18,
        "Validated end-to-end on Two Rooms; coordination/safety modules implemented but their gates are pending (see docs/VALIDATION_PLAN.md).",
        fontsize=8, ha="center", style="italic", color="#555")

out = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures", "alps4b_architecture_full.png"))
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
print(f"[saved] {out}")
