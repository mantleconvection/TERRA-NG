#!/usr/bin/env python3
"""
Roofline plot for EpsilonDivDivKerngen performance history on NVIDIA H100 SXM (Helma).
Parses ncu CSV output and creates a DRAM roofline chart plus a throughput bar chart.

Usage:
    python plot_roofline_h100.py ncu_results.csv [benchmark_stdout.log]
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

# ---------- H100 SXM measured peaks (Helma, measure_peaks_h100.cu) ----------
PEAK_FP64_TFLOPS = 32.1
PEAK_HBM_TB_S = 2.18
PEAK_FP64 = PEAK_FP64_TFLOPS * 1e12   # FLOP/s
PEAK_HBM = PEAK_HBM_TB_S * 1e12       # B/s
RIDGE_POINT = PEAK_FP64 / PEAK_HBM    # FLOP/Byte

# ---------- parse ncu csv ----------
ncu_file = sys.argv[1] if len(sys.argv) > 1 else "ncu_results.csv"
throughput_file = sys.argv[2] if len(sys.argv) > 2 else None

# ncu --csv output: each row has one metric for one kernel invocation
# Columns: "ID","Process ID","Process Name","Host Name","Kernel Name",
#           "Kernel Time","Context","Stream","Section Name",
#           "Metric Name","Metric Unit","Metric Value"
# We group by Kernel Name and collect all metrics.
kernel_metrics = defaultdict(lambda: defaultdict(float))

with open(ncu_file) as f:
    # Only keep CSV lines (start with " for data or contain the header)
    lines = [l for l in f if l.startswith('"')]
    reader = csv.DictReader(lines)
    for row in reader:
        kname = row.get("Kernel Name", "")
        if not kname:
            continue

        metric_name = row.get("Metric Name", "")
        val_str = row.get("Metric Value", "0")
        try:
            # ncu may use comma as thousands separator in some locales
            val = float(val_str.replace(",", "").replace("\"", ""))
        except (ValueError, TypeError):
            continue

        # Keep the max across invocations (largest level)
        kernel_metrics[kname][metric_name] = max(kernel_metrics[kname][metric_name], val)

# ---------- version labelling ----------
VERSION_NAMES = {
    "EpsilonDivDivSimple":          "baseline",
    "V02bSingleQuadpoint":          "cut computation",
    "V09SeparateScatter":           "shmem + teams",
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

for kname, metrics in kernel_metrics.items():
    label = version_label(kname)
    if label is None:
        continue

    # Duration in nanoseconds
    dur_ns = metrics.get("gpu__time_duration.sum", 0)
    if dur_ns <= 0:
        continue

    # FP64 FLOPs from SASS instruction counters (thread-level, already aggregated)
    dadd = metrics.get("smsp__sass_thread_inst_executed_op_dadd_pred_on.sum", 0)
    dmul = metrics.get("smsp__sass_thread_inst_executed_op_dmul_pred_on.sum", 0)
    dfma = metrics.get("smsp__sass_thread_inst_executed_op_dfma_pred_on.sum", 0)
    # If only FMA was collected, approximate: FMA dominates in these kernels
    flops = dadd + dmul + 2.0 * dfma

    # DRAM bytes
    hbm_bytes = metrics.get("dram__bytes.sum", 0)
    if hbm_bytes <= 0 and flops <= 0:
        continue

    oi_hbm = flops / hbm_bytes if hbm_bytes > 0 else 0
    perf = flops / (dur_ns * 1e-9) if dur_ns > 0 else 0
    bw_hbm = hbm_bytes / (dur_ns * 1e-9) if dur_ns > 0 else 0

    if label not in version_data or dur_ns > version_data[label]["duration_ns"]:
        version_data[label] = {
            "duration_ns": dur_ns,
            "flops": flops,
            "hbm_bytes": hbm_bytes,
            "oi_hbm": oi_hbm,
            "perf_flops": perf,
            "bw_hbm": bw_hbm,
        }

# Sort by version label
VERSION_ORDER = list(VERSION_NAMES.values())
sorted_versions = sorted(version_data.items(),
                         key=lambda x: VERSION_ORDER.index(x[0]) if x[0] in VERSION_ORDER else 99)

if not sorted_versions:
    print("ERROR: No matching kernels found in", ncu_file)
    print("Available kernel names:")
    for k in kernel_metrics:
        print(f"  {k}")
    sys.exit(1)

# ---------- parse throughput from benchmark log ----------
THROUGHPUT_GDOFS = {}

# Map benchmark description substrings to version labels
LOG_NAMES = {
    "EpsDivDivSimple":      "baseline",
    "v02b single":          "cut computation",
    "v09 separate":         "shmem + teams",
}

if throughput_file and os.path.exists(throughput_file):
    current_op = None
    with open(throughput_file) as f:
        for raw_line in f:
            # Strip [LOG | rank N | timestamp] prefix
            line = raw_line.strip()
            if "]" in line and line.startswith("["):
                line = line.split("]", 1)[-1].strip()
            # Match operator header or reset current_op for unmatched operators
            matched = False
            for tag, label in LOG_NAMES.items():
                if tag in line:
                    current_op = label
                    matched = True
                    break
            # Reset current_op for unmatched operator headers (not info lines)
            if not matched and not line.startswith("[") and (
                    re.match(r"EpsDivDiv\w*\s*\(", line) or re.match(r"v\d+\w?\s", line)):
                current_op = None
            # Parse CSV data lines: dofs,duration,id,level,timestamp,dofs/sec
            if current_op and "," in line and line[0].isdigit():
                parts = line.split(",")
                if len(parts) >= 4:
                    try:
                        dofs_per_sec = float(parts[-1].strip())
                        gdofs = dofs_per_sec / 1e9
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
cmap = plt.cm.viridis
n = len(sorted_versions)
colors = [cmap(0.15 + 0.75 * i / max(n - 1, 1)) for i in range(n)]

fig, ax_roof = plt.subplots(1, 1, figsize=(10, 8))

oi_range = np.logspace(-2, 2.5, 500)

# --- Roofline panel ---
bw_line = PEAK_HBM * oi_range
compute_line = np.full_like(oi_range, PEAK_FP64)
roofline = np.minimum(bw_line, compute_line)

ax_roof.loglog(oi_range, roofline / 1e12, "k-", linewidth=2.5)


# annotations
bw_annot_x = min(0.15, RIDGE_POINT * 0.3)
ax_roof.text(bw_annot_x, PEAK_HBM * bw_annot_x / 1e12 * 0.55,
             f"HBM BW = {PEAK_HBM_TB_S} TB/s (measured)",
             fontsize=11, rotation=38, color="gray", fontstyle="italic")
ax_roof.text(RIDGE_POINT * 5, PEAK_FP64 / 1e12 * 1.08,
             f"FP64 = {PEAK_FP64_TFLOPS} TFLOP/s (measured)",
             fontsize=11, color="gray", fontstyle="italic", ha="center")

# plot each version
for i, (label, d) in enumerate(sorted_versions):
    m = MARKERS[i % len(MARKERS)]
    oi = d["oi_hbm"]
    perf = d["perf_flops"] / 1e12
    ax_roof.plot(oi, perf, m,
                 color=colors[i], markersize=18, markeredgecolor="black",
                 markeredgewidth=1.0, zorder=5)

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
ax_roof.set_title("HBM Roofline (NVIDIA H100 SXM, Helma)", fontsize=15, fontweight="bold")
ax_roof.set_xlim(0.01, 200)
ax_roof.set_ylim(0.001, 50)
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

# --- Throughput inset (lower-right corner) ---
ax_tp = ax_roof.inset_axes([0.58, 0.05, 0.38, 0.45])

bar_labels = [label for label, _ in sorted_versions]
has_gdofs = any(THROUGHPUT_GDOFS.get(l, 0) > 0 for l in bar_labels)

if has_gdofs:
    bar_values = [max(THROUGHPUT_GDOFS.get(label, 0), 1e-4) for label in bar_labels]
    ylabel = "Gdofs/s"
else:
    bar_values = [d["perf_flops"] / 1e9 for _, d in sorted_versions]
    ylabel = "GFLOP/s"

bars = ax_tp.bar(range(n), bar_values, color=colors, edgecolor="black", linewidth=0.8, zorder=3)

fmt_func = lambda v: f"{v:.2f}" if v >= 0.1 else f"{v*1000:.0f}M"
for i, (val, bar) in enumerate(zip(bar_values, bars)):
    if val > 0:
        ax_tp.text(bar.get_x() + bar.get_width() / 2, val * 1.15,
                   fmt_func(val), ha="center", va="bottom", fontsize=10, fontweight="bold")

ax_tp.set_yscale("log")
ax_tp.set_xticks(range(n))
ax_tp.set_xticklabels([])
ax_tp.set_xlabel("")
ax_tp.set_ylabel(ylabel, fontsize=13)
ax_tp.set_title("Throughput (level 8)", fontsize=13, fontweight="bold")
ax_tp.grid(True, axis="y", alpha=0.3)
ax_tp.tick_params(axis="y", labelsize=12)
ax_tp.tick_params(axis="x", length=0)
ax_tp.set_xlim(-0.6, n - 0.4)
ax_tp.set_ylim(None, max(bar_values) * 2.5)


plt.tight_layout(rect=[0, 0, 1, 0.97])

out_dir = os.path.expanduser("~/terraneo/doc/epsdivdiv_benchmarks")
os.makedirs(out_dir, exist_ok=True)
plt.savefig(os.path.join(out_dir, "roofline_history_h100.png"), dpi=200)
plt.savefig(os.path.join(out_dir, "roofline_history_h100.pdf"))
print(f"\nSaved to {out_dir}/roofline_history_h100.{{png,pdf}}")
