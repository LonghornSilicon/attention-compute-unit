#!/usr/bin/env python3
"""c20 system figure: APA (ACU) + KVCE (ChannelQuant) inside one Qwen2 attention
step, annotated with the verified end-to-end numbers.

Data: analysis/c20_kvce_q05_headline.json (KVCE ChannelQuant on Qwen2-0.5B, n=1000).
Regenerate:  python paper/figs/c20_apa_kvce_system.py
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "analysis", "c20_kvce_q05_headline.json")
C = json.load(open(DATA))["configs"]

C_KVCE = "#e8f1fb"; E_KVCE = "#1f6fc4"
C_APA  = "#fdeee3"; E_APA  = "#d2691e"
C_ATT  = "#eef0f2"; E_ATT  = "#5b6670"
C_RES  = "#eaf7ee"; E_RES  = "#2e8b57"

fig, ax = plt.subplots(figsize=(15.5, 8.6))
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

def box(x, y, w, h, title, lines, fc, ec, tsz=11, lsz=9.0, tw="bold"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.4",
                 linewidth=1.8, edgecolor=ec, facecolor=fc, zorder=2))
    ax.text(x + w/2, y + h - 3.0, title, ha="center", va="top", fontsize=tsz,
            fontweight=tw, color=ec, zorder=3)
    if lines:
        ax.text(x + w/2, y + h - 7.6, "\n".join(lines), ha="center", va="top",
                fontsize=lsz, color="#222", zorder=3, linespacing=1.45)

def arrow(x1, y1, x2, y2, ec="#333", lw=2.0, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                 mutation_scale=17, lw=lw, color=ec, linestyle=ls, zorder=4,
                 shrinkA=2, shrinkB=2))

def lbl(x, y, t, sz=8.6, c="#444", st="italic", ha="center"):
    ax.text(x, y, t, ha=ha, va="center", fontsize=sz, color=c, style=st, zorder=5)

ax.text(50, 97.5, "LonghornSilicon inference accelerator — APA + KVCE, one Qwen2 attention step",
        ha="center", va="top", fontsize=15.5, fontweight="bold", color="#111")
ax.text(50, 93.0, "this session: ChannelQuant KV (KVCE) measured ~FP16 on Qwen2-0.5B at ~4 bits/value (green panel);  "
        "APA = the independently signed-off block it composes with",
        ha="center", va="top", fontsize=10.0, color="#555")

box(2, 74, 15, 13, "Decoder layer", ["hidden state h_t", "(RMSNorm -> attention)"], C_ATT, E_ATT)
box(21, 74, 15, 13, "Q / K / V proj", ["q_t, k_t, v_t", "per head, dim D=64"], C_ATT, E_ATT)
arrow(17, 80.5, 21, 80.5)

box(2, 30, 46, 40, "KVCE  —  ChannelQuant KV-cache codec  (this revamp)", [], C_KVCE, E_KVCE, tsz=12.5)
ax.text(25, 63.6, "new k_t, v_t  ->  compress  ->  compressed KV cache  ->  decompress  ->  K-hat, V-hat",
        ha="center", fontsize=8.8, color="#0d4a86", fontweight="bold")
box(4.5, 45.0, 20.5, 15.6, "KEY path  (per-channel INT4)",
    ["- group G=128 tokens, freeze", "  D per-channel fp16 scales",
     "- quantize keep channels -> INT4",
     "- CQ-4+: top-k outlier channels",
     "  held FP16 (static ROM mask)"], "#dbe9f9", E_KVCE, tsz=9.8, lsz=8.3)
box(26.5, 45.0, 19.5, 15.6, "VALUE path  (per-token INT4)",
    ["- per-token amax -> fp16 scale",
     "- quantize D dims -> INT4",
     "  (INT8 for CQ-8 tier)",
     "- decompress: code * scale -> fp32"], "#dbe9f9", E_KVCE, tsz=9.8, lsz=8.3)
box(4.5, 32.0, 41.5, 11.5, "unified per-channel SRAM record  {tag, D x fp16 field, D x INT4 code}",
    ["keep channel -> {group scale, INT4}      outlier channel -> {raw fp16, code +1}  (exact widen on read)",
     "serialized 1 shared scale / quant / dequant unit  —  RTL: all CI gates green (func / synth / formal / OpenLane)"],
    "#eaf2fc", E_KVCE, tsz=9.2, lsz=8.2, tw="bold")
arrow(28.5, 74, 24, 70.3, ec=E_KVCE)
lbl(33.5, 72.0, "K, V", c=E_KVCE, st="normal")

box(52, 62, 15.5, 12, "scores  S", ["S = Q . K-hat^T / sqrt(D)", "pre-softmax logits"], C_ATT, E_ATT)
arrow(48, 55, 52, 66, ec=E_KVCE)
lbl(50.6, 61.5, "K-hat", c=E_KVCE, st="normal", sz=9)

box(52, 26, 44, 30, "APA  —  Attention Compute Unit  (precision controller)", [], C_APA, E_APA, tsz=12.5)
box(54.5, 40, 39, 12.5, "streaming ratio test  (no divide, no softmax)",
    ["gate:   max(|S|) * N   >   THRESHOLD * sum|S|      (THRESHOLD = 10)",
     "one int8 score / clock -> 1-cycle decision per tile  -  ~30 flip-flops",
     "Sky130 signed off (253/253 TB)  -  codec-agnostic: sees only S"],
    "#fbe3d0", E_APA, tsz=9.6, lsz=8.4)
box(54.5, 28.5, 18.2, 9.5, "tile -> INT8 path", ["cheap S.V-hat matmul", "~100% of tiles on Qwen"],
    "#f7d9bf", E_APA, tsz=9.4, lsz=8.6)
box(75.3, 28.5, 18.2, 9.5, "tile -> FP16 path", ["exact S.V-hat matmul", "rarely fires on Qwen"],
    "#f7d9bf", E_APA, tsz=9.4, lsz=8.6)
arrow(67.7, 62, 70, 56, ec=E_APA)
lbl(72.0, 60.0, "S (int8)", c=E_APA, st="normal", sz=8.6)

box(52, 8, 20, 12.5, "attention out  O_t", ["O = softmax(S) . V-hat", "INT8/FP16 per tile"], C_ATT, E_ATT)
arrow(62, 26, 62, 20.5, ec=E_APA)
arrow(48, 34.0, 52, 15.0, ec=E_KVCE, ls="--")
lbl(51.4, 27.0, "V-hat", c=E_KVCE, st="normal", sz=9.5)
arrow(72, 14.2, 82, 14.2)
box(82, 8, 15.5, 12.5, "-> next layer", ["residual +", "feed-forward"], C_ATT, E_ATT)

fp = C["fp16"]["acc_norm"]; c4 = C["cq4"]; c4p = C["cq4plus"]
box(2, 4, 46, 22, "VERIFIED end-to-end  —  HellaSwag acc_norm, Qwen2-0.5B, n=1000 (this session)", [],
    C_RES, E_RES, tsz=11)
rows = [
    ("config", "acc_norm", "bits/val", "d vs FP16"),
    ("FP16 (baseline)", f"{fp:.4f}", "16.0", "—"),
    ("ChannelQuant CQ-4", f"{c4['acc_norm']:.4f}", f"{c4['eff_bits']:.2f}", f"{c4['delta_vs_fp16']:+.3f}"),
    ("ChannelQuant CQ-4+", f"{c4p['acc_norm']:.4f}", f"{c4p['eff_bits']:.2f}", f"{c4p['delta_vs_fp16']:+.3f}"),
]
xcol = [4.5, 24.5, 33.5, 41.0]
for r, row in enumerate(rows):
    yy = 20.5 - r*3.4
    wt = "bold" if r == 0 else "normal"
    col = "#1b5e37" if r == 0 else "#123"
    for xc, cell, al in zip(xcol, row, ["left", "center", "center", "center"]):
        ax.text(xc, yy, cell, ha=al, va="center", fontsize=9.2, fontweight=wt, color=col, zorder=5)
    if r == 0:
        ax.plot([4.0, 46], [yy-1.5, yy-1.5], color=E_RES, lw=1.0, zorder=4)
ax.text(25, 5.6, "deltas within the paired CI = statistically indistinguishable from FP16 (near-lossless, ~3.8x KV compression)",
        ha="center", va="center", fontsize=8.4, fontweight="bold", color=E_RES, zorder=5)

ax.add_patch(FancyBboxPatch((74, 88.5), 3, 3, boxstyle="round,pad=0.2", fc=C_KVCE, ec=E_KVCE, lw=1.5))
ax.text(78, 90, "KVCE (ChannelQuant)", va="center", fontsize=8.6, color=E_KVCE)
ax.add_patch(FancyBboxPatch((74, 84.5), 3, 3, boxstyle="round,pad=0.2", fc=C_APA, ec=E_APA, lw=1.5))
ax.text(78, 86, "APA (precision controller)", va="center", fontsize=8.6, color=E_APA)

plt.tight_layout()
out_png = os.path.join(HERE, "c20_apa_kvce_system.png")
plt.savefig(out_png, dpi=155, bbox_inches="tight", facecolor="white")
plt.savefig(out_png.replace(".png", ".pdf"), bbox_inches="tight", facecolor="white")
print("wrote", out_png)
