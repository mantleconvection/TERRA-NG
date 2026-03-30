#!/usr/bin/env python3
"""
Roofline plot for EpsilonDivDivKerngen performance history.
Parses ncu CSV output and creates a DRAM + L2 roofline chart plus a throughput bar chart.
"""
import csv
import re
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np

# H100 SXM specs
PEAK_FP64_TFLOPS = 34.0   # TFLOP/s
PEAK_HBM_TB_S = 3.35      # TB/s HBM3 bandwidth
PEAK_L2_TB_S = 12.0       # TB/s L2 cache bandwidth (approximate)
PEAK_FP64 = PEAK_FP64_TFLOPS * 1e12  # FLOP/s
PEAK_HBM = PEAK_HBM_TB_S * 1e12     # B/s
PEAK_L2 = PEAK_L2_TB_S * 1e12       # B/s

# ---------- parse ncu csv ----------
ncu_file = "ncu_data.csv"

kernels = defaultdict(dict)
with open(ncu_file) as f:
    reader = csv.DictReader(f)
    for row in reader:
        kid = row["ID"]
        kname = row["Kernel Name"]
        metric = row["Metric Name"]
        val_str = row["Metric Value"].replace(",", "")
        val = float(val_str)
        kernels[(kid, kname)][metric] = val

# ---------- compute roofline quantities ----------
version_data = {}

# Ordered labels and descriptions for clean naming
VERSION_NAMES = {
    "EpsilonDivDivSimple":  "v00a simple",
    "EpsilonDivDiv":        "v00b fused",
    "V01Initial":       "v01 initial",
    "V02SplitDimij":    "v02 split dimij",
    "V02bSingleQuadpoint": "v02b 1 quadpt",
    "V03TeamsPrecomp":  "v03 teams+precomp",
    "V04ShmemCoords":   "v04 shmem coords",
    "V05ShmemSrcK":     "v05 shmem src+k",
    "V06XyTiling":      "v06 xy tiling",
    "V07SplitPaths":    "v07 split paths",
    "V08ScalarCoalesced": "v08 scalar coal.",
    "V09SeparateScatter": "v09 sep. scatter",
    "V10SeqRpasses":    "v10 seq rpasses",
}

def version_label(kname):
    # Check specific versioned names first (longest match first)
    for tag, label in VERSION_NAMES.items():
        if tag in kname:
            # Disambiguate EpsilonDivDiv vs EpsilonDivDivKerngen vs EpsilonDivDivSimple
            if tag == "EpsilonDivDiv":
                # Must not match EpsilonDivDivKerngen or EpsilonDivDivSimple
                if "Kerngen" in kname or "Simple" in kname:
                    continue
                return label
            return label
    # Skip EpsilonDivDivKerngen (v11 current) - no longer plotted
    if "EpsilonDivDivKerngen<" in kname:
        return None
    return kname[:30]

for (kid, kname), metrics in kernels.items():
    label = version_label(kname)
    if label is None:
        continue
    dur = metrics.get("gpu__time_duration.sum", 0)
    if label not in version_data or dur > version_data[label]["duration_ns"]:
        dadd = metrics.get("smsp__sass_thread_inst_executed_op_dadd_pred_on.sum", 0)
        dmul = metrics.get("smsp__sass_thread_inst_executed_op_dmul_pred_on.sum", 0)
        dfma = metrics.get("smsp__sass_thread_inst_executed_op_dfma_pred_on.sum", 0)
        flops = dadd + dmul + 2 * dfma
        dram_bytes = metrics.get("dram__bytes.sum", 0)
        l2_bytes = metrics.get("lts__t_bytes.sum", 0)
        regs = metrics.get("launch__registers_per_thread", 0)
        shmem = metrics.get("launch__shared_mem_per_block_allocated", 0)

        oi_dram = flops / dram_bytes if dram_bytes > 0 else 0
        oi_l2 = flops / l2_bytes if l2_bytes > 0 else 0
        perf = flops / (dur * 1e-9) if dur > 0 else 0
        bw_dram = dram_bytes / (dur * 1e-9) if dur > 0 else 0
        bw_l2 = l2_bytes / (dur * 1e-9) if dur > 0 else 0

        # NCU-reported Speed of Light percentages
        sol_dram = metrics.get("dram__throughput.avg.pct_of_peak_sustained_elapsed", 0)
        sol_l2 = metrics.get("lts__throughput.avg.pct_of_peak_sustained_elapsed", 0)
        sol_sm = metrics.get("sm__throughput.avg.pct_of_peak_sustained_elapsed", 0)

        version_data[label] = {
            "duration_ns": dur,
            "flops": flops,
            "dram_bytes": dram_bytes,
            "l2_bytes": l2_bytes,
            "oi_dram": oi_dram,
            "oi_l2": oi_l2,
            "perf_flops": perf,
            "bw_dram": bw_dram,
            "bw_l2": bw_l2,
            "regs": int(regs),
            "shmem": int(shmem),
            "sol_dram_pct": sol_dram,
            "sol_l2_pct": sol_l2,
            "sol_sm_pct": sol_sm,
        }

sorted_versions = sorted(version_data.items(), key=lambda x: x[0])

# ---------- throughput data from noprof benchmark (job 289783, level 8, 10 executions) ----------
THROUGHPUT_GDOFS = {
    "v00a simple":      0.0117,
    "v00b fused":       0.0188,
    "v01 initial":      0.0443,
    "v02 split dimij":  0.107,
    "v03 teams+precomp": 1.713,
    "v04 shmem coords": 4.304,
    "v05 shmem src+k":  5.413,
    "v06 xy tiling":    4.866,
    "v07 split paths":  4.852,
    "v08 scalar coal.":  6.439,
    "v09 sep. scatter": 7.693,
    "v10 seq rpasses":  7.841,
}

# ---------- print table ----------
print(f"{'Version':<25} {'Time(ms)':>10} {'GFLOP/s':>10} {'DRAM BW':>12} {'L2 BW':>12} "
      f"{'OI(DRAM)':>10} {'OI(L2)':>10} {'Regs':>6} {'Shmem':>8} {'Gdofs/s':>10}")
print("-" * 125)
for label, d in sorted_versions:
    gdofs = THROUGHPUT_GDOFS.get(label, 0)
    print(f"{label:<25} {d['duration_ns']/1e6:10.2f} {d['perf_flops']/1e9:10.1f} "
          f"{d['bw_dram']/1e9:10.1f} GB/s {d['bw_l2']/1e9:10.1f} GB/s "
          f"{d['oi_dram']:10.2f} {d['oi_l2']:10.2f} {d['regs']:6d} {d['shmem']:8d} {gdofs:10.3f}")

# ---------- plot setup ----------
MARKERS = ["s", "D", "o", "^", "v", "p", "h", "P", "*", "X", "d", "H", "8", ">"]
cmap = plt.cm.plasma
n = len(sorted_versions)
colors = [cmap(0.15 + 0.75 * i / max(n - 1, 1)) for i in range(n)]

fig = plt.figure(figsize=(24, 20))
ax_dram = fig.add_subplot(2, 2, 1)
ax_l2 = fig.add_subplot(2, 2, 2)
ax_tp = fig.add_subplot(2, 2, 3)
ax_pct = fig.add_subplot(2, 2, 4)

oi_range = np.logspace(-2, 2.5, 500)

# --- roofline panels ---
panels = [
    (ax_dram, PEAK_HBM, f"HBM BW = {PEAK_HBM_TB_S} TB/s", PEAK_HBM_TB_S, "oi_dram", "DRAM"),
    (ax_l2,   PEAK_L2,  f"L2 BW = {PEAK_L2_TB_S} TB/s",   PEAK_L2_TB_S,  "oi_l2",   "L2"),
]

for ax, peak_bw, bw_text, bw_tb, oi_key, subtitle in panels:
    # roofline ceiling
    bw_line = peak_bw * oi_range
    compute_line = np.full_like(oi_range, PEAK_FP64)
    roofline = np.minimum(bw_line, compute_line)

    ax.loglog(oi_range, roofline / 1e12, "k-", linewidth=2.5)
    ax.fill_between(oi_range, roofline / 1e12, 200, alpha=0.04, color="black")

    # annotations on roofline
    ridge_oi = PEAK_FP64 / peak_bw
    bw_annot_x = min(0.15, ridge_oi * 0.3)
    ax.text(bw_annot_x, peak_bw * bw_annot_x / 1e12 * 0.55, bw_text,
            fontsize=11, rotation=38, color="gray", fontstyle="italic")
    ax.text(ridge_oi * 5, PEAK_FP64 / 1e12 * 1.08, f"FP64 peak = {PEAK_FP64_TFLOPS} TFLOP/s",
            fontsize=11, color="gray", fontstyle="italic", ha="center")

    # plot each version
    for i, (label, d) in enumerate(sorted_versions):
        m = MARKERS[i % len(MARKERS)]
        oi = d[oi_key]
        perf = d["perf_flops"] / 1e12
        ax.plot(oi, perf, m,
                color=colors[i], markersize=18, markeredgecolor="black", markeredgewidth=1.0, zorder=5)
        num = label.split()[0]
        ax.annotate(num, (oi, perf),
                    textcoords="offset points", xytext=(8, 8),
                    fontsize=10, fontweight="bold", color="black", zorder=6,
                    path_effects=[pe.withStroke(linewidth=3, foreground="white")])

    # arrows
    for i in range(len(sorted_versions) - 1):
        _, d1 = sorted_versions[i]
        _, d2 = sorted_versions[i + 1]
        ax.annotate("", xy=(d2[oi_key], d2["perf_flops"] / 1e12),
                    xytext=(d1[oi_key], d1["perf_flops"] / 1e12),
                    arrowprops=dict(arrowstyle="-|>", color="gray", lw=1.0, alpha=0.5,
                                    connectionstyle="arc3,rad=0.15"))

    ax.set_xlabel(f"Operational Intensity [FLOP/Byte] ({subtitle})", fontsize=14)
    ax.set_ylabel("Performance [TFLOP/s] (FP64)", fontsize=14)
    ax.set_title(f"{subtitle} Roofline (NVIDIA H100)", fontsize=15, fontweight="bold")
    ax.set_xlim(0.01, 200)
    ax.set_ylim(0.01, 60)
    ax.grid(True, which="both", alpha=0.2)
    ax.tick_params(labelsize=12)

# shared legend on the DRAM panel
legend_labels = [f"{label} ({d['perf_flops']/1e9:.0f} GFLOP/s, regs={d['regs']})"
                 for label, d in sorted_versions]
legend_handles = [plt.Line2D([0], [0], marker=MARKERS[i % len(MARKERS)], color=colors[i],
                              markersize=11, markeredgecolor="black", markeredgewidth=0.8,
                              linestyle="none")
                  for i in range(n)]
leg = ax_dram.legend(legend_handles, legend_labels, fontsize=9, loc="upper left",
                     framealpha=0.9, ncol=1, title="Version (perf, registers/thread)")
leg.get_title().set_fontsize(10)

# --- throughput bar chart ---
bar_labels = [label for label, _ in sorted_versions]
bar_values = [THROUGHPUT_GDOFS.get(label, 0) for label in bar_labels]

bars = ax_tp.bar(range(n), bar_values, color=colors, edgecolor="black", linewidth=0.8, zorder=3)

# value labels on top of bars
for i, (val, bar) in enumerate(zip(bar_values, bars)):
    if val >= 0.1:
        ax_tp.text(bar.get_x() + bar.get_width() / 2, val + 0.15,
                   f"{val:.1f}", ha="center", va="bottom", fontsize=13, fontweight="bold")
    else:
        ax_tp.text(bar.get_x() + bar.get_width() / 2, val + 0.15,
                   f"{val*1000:.0f}M", ha="center", va="bottom", fontsize=12, fontweight="bold")

ax_tp.set_xticks(range(n))
ax_tp.set_xticklabels([l.split()[0] for l in bar_labels], rotation=45, ha="right", fontsize=11)
ax_tp.set_ylabel("Throughput [Gdofs/s]", fontsize=14)
ax_tp.set_title("Operator Throughput (level 8, H100)", fontsize=15, fontweight="bold")
ax_tp.grid(True, axis="y", alpha=0.3)
ax_tp.tick_params(labelsize=12)
ax_tp.set_xlim(-0.6, n - 0.4)

# --- % of peak bar chart (NCU Speed of Light metrics) ---
bar_labels_short = [l.split()[0] for l in bar_labels]
pct_dram = [version_data[l]["sol_dram_pct"] for l in bar_labels]
pct_l2 = [version_data[l]["sol_l2_pct"] for l in bar_labels]
pct_sm = [version_data[l]["sol_sm_pct"] for l in bar_labels]

x = np.arange(n)
bw = 0.25
bars_dram = ax_pct.bar(x - bw, pct_dram, bw, label="DRAM BW", color="#4477AA", edgecolor="black", linewidth=0.6, zorder=3)
bars_l2 = ax_pct.bar(x, pct_l2, bw, label="L2 BW", color="#66CCEE", edgecolor="black", linewidth=0.6, zorder=3)
bars_sm = ax_pct.bar(x + bw, pct_sm, bw, label="SM Compute", color="#EE6677", edgecolor="black", linewidth=0.6, zorder=3)

# value labels
for bars_group in [bars_dram, bars_l2, bars_sm]:
    for bar in bars_group:
        h = bar.get_height()
        if h >= 1:
            ax_pct.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                        f"{h:.0f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

ax_pct.set_xticks(x)
ax_pct.set_xticklabels(bar_labels_short, rotation=45, ha="right", fontsize=11)
ax_pct.set_ylabel("% of Peak (NCU Speed of Light)", fontsize=14)
ax_pct.set_title("Hardware Utilization (NVIDIA H100)", fontsize=15, fontweight="bold")
ax_pct.legend(fontsize=10, loc="upper left")
ax_pct.grid(True, axis="y", alpha=0.3)
ax_pct.tick_params(labelsize=12)
ax_pct.set_xlim(-0.6, n - 0.4)
ax_pct.set_ylim(0, max(max(pct_dram), max(pct_l2), max(pct_sm)) * 1.15)

fig.suptitle("EpsilonDivDiv Optimization History", fontsize=18, fontweight="bold", y=0.99)
plt.tight_layout(rect=[0, 0, 1, 0.97])
import os
out_dir = os.path.expanduser("~/terraneo/doc/epsdivdiv_benchmarks")
os.makedirs(out_dir, exist_ok=True)
plt.savefig(os.path.join(out_dir, "roofline_history.png"), dpi=200)
plt.savefig(os.path.join(out_dir, "roofline_history.pdf"))
print(f"\nSaved to {out_dir}/roofline_history.{{png,pdf}}")
