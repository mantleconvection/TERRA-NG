#!/usr/bin/env python3
"""
Roofline plot for EpsilonDivDivKerngen performance history on AMD MI250X (LUMI-G).
Parses rocprof CSV output and creates a DRAM roofline chart plus a throughput bar chart.
"""
import csv
import re
import sys
import os
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np

# ---------- MI250X GCD specs ----------
# One GCD (Graphics Compute Die) of MI250X:
#   FP64 peak:  26.5 TFLOP/s (theoretical)
#   HBM2e BW:   1.6 TB/s (theoretical per GCD)
PEAK_FP64_TFLOPS = 26.5
PEAK_HBM_TB_S = 1.6
PEAK_FP64 = PEAK_FP64_TFLOPS * 1e12   # FLOP/s
PEAK_HBM = PEAK_HBM_TB_S * 1e12       # B/s
RIDGE_POINT = PEAK_FP64 / PEAK_HBM    # FLOP/Byte

# ---------- parse rocprof csv ----------
rocprof_file = sys.argv[1] if len(sys.argv) > 1 else "rocprof_results.csv"
# Optional: benchmark stdout log file for throughput (Gdofs/s) data
throughput_file = sys.argv[2] if len(sys.argv) > 2 else None

# rocprof CSV has columns: Index, KernelName, gpu-id, queue-id, ...counters..., DispatchNs, BeginNs, EndNs, ...
# Multiple passes produce rows for the same kernel with different counter columns filled.
# We group by KernelName and merge counters.
kernel_data = defaultdict(lambda: defaultdict(float))
kernel_duration_ns = defaultdict(float)

with open(rocprof_file) as f:
    reader = csv.DictReader(f)
    for row in reader:
        kname = row.get("KernelName", row.get("Kernel Name", ""))
        if not kname:
            continue

        # Duration from timestamps
        begin = float(row.get("BeginNs", 0) or 0)
        end = float(row.get("EndNs", 0) or 0)
        dur = end - begin
        if dur <= 0:
            dispatch = float(row.get("DispatchNs", 0) or 0)
            complete = float(row.get("CompleteNs", 0) or 0)
            dur = complete - dispatch

        # Keep the longest invocation (largest kernel, last level)
        if dur > kernel_duration_ns[kname]:
            kernel_duration_ns[kname] = dur

        # Merge all counter columns
        for col, val_str in row.items():
            if col in ("Index", "KernelName", "Kernel Name", "gpu-id", "queue-id",
                        "queue-index", "pid", "tid", "grd", "wgr",
                        "lds", "scr", "vgpr", "sgpr", "fbar",
                        "sig", "obj", "DispatchNs", "BeginNs", "EndNs",
                        "CompleteNs"):
                continue
            try:
                val = float(val_str)
                if val > 0:
                    kernel_data[kname][col] = max(kernel_data[kname][col], val)
            except (ValueError, TypeError):
                pass

# ---------- version labelling ----------
VERSION_NAMES = {
    "EpsilonDivDivSimple":          "v00a simple",
    "EpsilonDivDiv":                "v00b fused",
    "V01Initial":                   "v01 initial",
    "V02SplitDimij":                "v02 split dimij",
    "V02bSingleQuadpoint":          "v02b 1 quadpt",
    "V07SplitPaths":                "v07 split paths",
    "V09SeparateScatter":           "v09 sep. scatter",
}

def version_label(kname):
    for tag, label in VERSION_NAMES.items():
        if tag in kname:
            if tag == "EpsilonDivDiv":
                if "Kerngen" in kname or "Simple" in kname:
                    continue
                return label
            return label
    return None

# ---------- compute roofline quantities ----------
version_data = {}

for kname, counters in kernel_data.items():
    label = version_label(kname)
    if label is None:
        continue

    dur_ns = kernel_duration_ns[kname]
    if dur_ns <= 0:
        continue

    # FP64 FLOPs: each VALU instruction operates on 64 lanes (wavefront width)
    # dadd/dmul = 1 FLOP/lane, dfma = 2 FLOPs/lane
    wavefront_size = 64
    dadd = counters.get("SQ_INSTS_VALU_ADD_F64", 0)
    dmul = counters.get("SQ_INSTS_VALU_MUL_F64", 0)
    dfma = counters.get("SQ_INSTS_VALU_FMA_F64", 0)
    flops = (dadd + dmul + 2.0 * dfma) * wavefront_size

    # HBM bytes: TCC_EA requests are 32B each
    hbm_rd = counters.get("TCC_EA_RDREQ_32B_sum", 0)
    hbm_wr = counters.get("TCC_EA_WRREQ_32B_sum", 0)
    hbm_bytes = (hbm_rd + hbm_wr) * 32.0

    # L2 cache traffic (requests, each 64B cache line on MI250X)
    l2_rd = counters.get("TCC_READ_sum", 0)
    l2_wr = counters.get("TCC_WRITE_sum", 0)
    l2_bytes = (l2_rd + l2_wr) * 64.0

    if hbm_bytes <= 0 and l2_bytes <= 0:
        continue

    oi_hbm = flops / hbm_bytes if hbm_bytes > 0 else 0
    perf = flops / (dur_ns * 1e-9) if dur_ns > 0 else 0
    bw_hbm = hbm_bytes / (dur_ns * 1e-9) if dur_ns > 0 else 0

    if label not in version_data or dur_ns > version_data[label]["duration_ns"]:
        version_data[label] = {
            "duration_ns": dur_ns,
            "flops": flops,
            "hbm_bytes": hbm_bytes,
            "l2_bytes": l2_bytes,
            "oi_hbm": oi_hbm,
            "perf_flops": perf,
            "bw_hbm": bw_hbm,
        }

# Sort by version label
VERSION_ORDER = list(VERSION_NAMES.values())
sorted_versions = sorted(version_data.items(),
                         key=lambda x: VERSION_ORDER.index(x[0]) if x[0] in VERSION_ORDER else 99)

if not sorted_versions:
    print("ERROR: No matching kernels found in", rocprof_file)
    print("Available kernel names:")
    for k in kernel_data:
        print(f"  {k}")
    sys.exit(1)

# ---------- parse throughput from benchmark log ----------
# The benchmark prints CSV-style lines:  level, dofs, duration (s), updated dofs/sec
# We parse the last (highest-level) entry per operator section.
THROUGHPUT_GDOFS = {}

if throughput_file and os.path.exists(throughput_file):
    current_op = None
    with open(throughput_file) as f:
        for line in f:
            line = line.strip()
            # Detect operator header lines from benchmark_description map
            for tag, label in VERSION_NAMES.items():
                if tag in line and ("double" in line.lower() or "simple" in line.lower()
                                    or "split" in line.lower() or "quadpoint" in line.lower()
                                    or "scatter" in line.lower() or "paths" in line.lower()
                                    or "initial" in line.lower() or "fused" in line.lower()
                                    or "naive" in line.lower()):
                    current_op = label
                    break
            # Parse CSV data lines: level, dofs, duration, dofs/sec
            if current_op and "," in line:
                parts = line.split(",")
                if len(parts) >= 4:
                    try:
                        dofs_per_sec = float(parts[-1].strip())
                        gdofs = dofs_per_sec / 1e9
                        # Keep the last (highest level) entry
                        THROUGHPUT_GDOFS[current_op] = gdofs
                    except ValueError:
                        pass

# ---------- print table ----------
print(f"\n{'Version':<22} {'Time(ms)':>10} {'GFLOP/s':>10} {'HBM BW':>12} {'OI(HBM)':>10} {'Gdofs/s':>10}")
print("-" * 82)
for label, d in sorted_versions:
    gdofs = THROUGHPUT_GDOFS.get(label, 0)
    print(f"{label:<22} {d['duration_ns']/1e6:10.2f} {d['perf_flops']/1e9:10.1f} "
          f"{d['bw_hbm']/1e9:10.1f} GB/s {d['oi_hbm']:10.2f} {gdofs:10.3f}")

# ---------- plot ----------
MARKERS = ["s", "D", "o", "^", "v", "p", "h", "P", "*", "X", "d", "H"]
cmap = plt.cm.plasma
n = len(sorted_versions)
colors = [cmap(0.15 + 0.75 * i / max(n - 1, 1)) for i in range(n)]

fig, (ax_roof, ax_tp) = plt.subplots(1, 2, figsize=(18, 8))

oi_range = np.logspace(-2, 2.5, 500)

# --- Roofline panel ---
bw_line = PEAK_HBM * oi_range
compute_line = np.full_like(oi_range, PEAK_FP64)
roofline = np.minimum(bw_line, compute_line)

ax_roof.loglog(oi_range, roofline / 1e12, "k-", linewidth=2.5)
ax_roof.fill_between(oi_range, roofline / 1e12, 200, alpha=0.04, color="black")

# annotations
bw_annot_x = min(0.15, RIDGE_POINT * 0.3)
ax_roof.text(bw_annot_x, PEAK_HBM * bw_annot_x / 1e12 * 0.55,
             f"HBM2e BW = {PEAK_HBM_TB_S} TB/s",
             fontsize=11, rotation=38, color="gray", fontstyle="italic")
ax_roof.text(RIDGE_POINT * 5, PEAK_FP64 / 1e12 * 1.08,
             f"FP64 peak = {PEAK_FP64_TFLOPS} TFLOP/s",
             fontsize=11, color="gray", fontstyle="italic", ha="center")

# plot each version
for i, (label, d) in enumerate(sorted_versions):
    m = MARKERS[i % len(MARKERS)]
    oi = d["oi_hbm"]
    perf = d["perf_flops"] / 1e12
    ax_roof.plot(oi, perf, m,
                 color=colors[i], markersize=18, markeredgecolor="black",
                 markeredgewidth=1.0, zorder=5)
    num = label.split()[0]
    ax_roof.annotate(num, (oi, perf),
                     textcoords="offset points", xytext=(8, 8),
                     fontsize=10, fontweight="bold", color="black", zorder=6,
                     path_effects=[pe.withStroke(linewidth=3, foreground="white")])

# arrows between consecutive versions
for i in range(len(sorted_versions) - 1):
    _, d1 = sorted_versions[i]
    _, d2 = sorted_versions[i + 1]
    ax_roof.annotate("", xy=(d2["oi_hbm"], d2["perf_flops"] / 1e12),
                     xytext=(d1["oi_hbm"], d1["perf_flops"] / 1e12),
                     arrowprops=dict(arrowstyle="-|>", color="gray", lw=1.0, alpha=0.5,
                                     connectionstyle="arc3,rad=0.15"))

ax_roof.set_xlabel("Operational Intensity [FLOP/Byte] (HBM)", fontsize=14)
ax_roof.set_ylabel("Performance [TFLOP/s] (FP64)", fontsize=14)
ax_roof.set_title("HBM Roofline (AMD MI250X GCD, LUMI-G)", fontsize=15, fontweight="bold")
ax_roof.set_xlim(0.01, 200)
ax_roof.set_ylim(0.001, 40)
ax_roof.grid(True, which="both", alpha=0.2)
ax_roof.tick_params(labelsize=12)

# legend
legend_labels = [f"{label} ({d['perf_flops']/1e9:.0f} GFLOP/s)"
                 for label, d in sorted_versions]
legend_handles = [plt.Line2D([0], [0], marker=MARKERS[i % len(MARKERS)], color=colors[i],
                              markersize=11, markeredgecolor="black", markeredgewidth=0.8,
                              linestyle="none")
                  for i in range(n)]
leg = ax_roof.legend(legend_handles, legend_labels, fontsize=9, loc="upper left",
                     framealpha=0.9, ncol=1, title="Version (attained perf)")
leg.get_title().set_fontsize(10)

# --- Throughput bar chart ---
bar_labels = [label for label, _ in sorted_versions]
has_gdofs = any(THROUGHPUT_GDOFS.get(l, 0) > 0 for l in bar_labels)

if has_gdofs:
    bar_values = [THROUGHPUT_GDOFS.get(label, 0) for label in bar_labels]
    ylabel = "Throughput [Gdofs/s]"
    title = "Operator Throughput (level 8, MI250X GCD)"
    fmt_func = lambda v: f"{v:.2f}" if v >= 0.1 else f"{v*1000:.0f}M"
else:
    bar_values = [d["perf_flops"] / 1e9 for _, d in sorted_versions]
    ylabel = "Performance [GFLOP/s] (FP64)"
    title = "Kernel Performance (MI250X GCD)"
    fmt_func = lambda v: f"{v:.0f}"

bars = ax_tp.bar(range(n), bar_values, color=colors, edgecolor="black", linewidth=0.8, zorder=3)

for i, (val, bar) in enumerate(zip(bar_values, bars)):
    if val > 0:
        ax_tp.text(bar.get_x() + bar.get_width() / 2, val * 1.02,
                   fmt_func(val), ha="center", va="bottom", fontsize=12, fontweight="bold")

ax_tp.set_xticks(range(n))
ax_tp.set_xticklabels([l.split()[0] for l in bar_labels], rotation=45, ha="right", fontsize=11)
ax_tp.set_ylabel(ylabel, fontsize=14)
ax_tp.set_title(title, fontsize=15, fontweight="bold")
ax_tp.grid(True, axis="y", alpha=0.3)
ax_tp.tick_params(labelsize=12)
ax_tp.set_xlim(-0.6, n - 0.4)

fig.suptitle("EpsilonDivDiv Optimization History (LUMI-G)", fontsize=18, fontweight="bold", y=0.99)
plt.tight_layout(rect=[0, 0, 1, 0.97])

out_dir = os.path.expanduser("~/terraneo/doc/epsdivdiv_benchmarks")
os.makedirs(out_dir, exist_ok=True)
plt.savefig(os.path.join(out_dir, "roofline_history_mi250x.png"), dpi=200)
plt.savefig(os.path.join(out_dir, "roofline_history_mi250x.pdf"))
print(f"\nSaved to {out_dir}/roofline_history_mi250x.{{png,pdf}}")
