#!/usr/bin/env python3
"""Consolidate the re-measured EpsDivDiv history (H100, level 8, 505M dofs) into
one stats table + a roofline-history figure.

Data sources (all level 8, sdr 0, single H100):
  - throughput (Gdof/s, clean unprofiled timing):  bo_remeasure_hist.o467167 stdout
  - FLOP instruction counts (v01..current):        ncu_history_lean_467249.csv
  - FLOP instruction counts (v00a, v00b):          ncu_data.csv  (original; code unchanged)
  - registers / DRAM bytes / shmem / spill:        ncu_history_lean_467306.csv
"""
import csv, re, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Input paths are env-overridable so the analysis re-runs on another machine
# without editing source. Defaults point at the original run; the producing
# job scripts are remeasure_history.sh (throughput) and
# remeasure_history_ncu_lean.sh (ncu registers/DRAM/spill), kept in this dir.
BP  = os.environ.get("BO_PERF_DIR", "/home/hpc/iwia/iwia054h/terraneo-build/apps/benchmarks/performance")
DOC = os.environ.get("BO_DOC_DIR", os.path.dirname(os.path.abspath(__file__)))
THR  = os.environ.get("BO_THROUGHPUT", f"{BP}/bo_remeasure_hist.o467167")    # throughput stdout
FLO  = os.environ.get("BO_FLOPS_CSV",  f"{BP}/ncu_history_lean_467249.csv")  # FLOPs v01..current
SPL  = os.environ.get("BO_SPILL_CSV",  f"{BP}/ncu_history_lean_467306.csv")  # regs/dram/shmem/spill all 14
ORIG = os.environ.get("BO_ORIG_NCU",   f"{DOC}/ncu_data.csv")                # FLOPs v00a/v00b (code unchanged)
DOFS = 505284102                                                            # level-8 vector dofs

# chronological versions; id = position in all_benchmark_types
VERS = ["v00a","v00b","v01","v02","v02b","v03","v04","v05","v06","v07","v08","v09","v10","current"]

def classify(kname):
    if "EpsilonDivDivSimple" in kname: return "v00a"
    t = re.search(r"KerngenV(\d+)(b?)", kname)
    if t:
        return f"v{int(t.group(1)):02d}{t.group(2)}"
    if "EpsilonDivDivKerngen" in kname: return "current"
    if re.search(r"EpsilonDivDiv<", kname): return "v00b"
    return None

def load_ncu(path):
    """return {version: {metric: value}} keeping the longest-duration launch."""
    lines = open(path).read().splitlines()
    start = next(i for i,l in enumerate(lines) if l.startswith('"ID"'))
    rdr = csv.reader(lines[start:]); hdr = next(rdr); ci = {n:i for i,n in enumerate(hdr)}
    K,M,V,ID = ci["Kernel Name"],ci["Metric Name"],ci["Metric Value"],ci["ID"]
    launches = {}
    for r in rdr:
        if len(r)<=K: continue
        d = launches.setdefault(r[ID], {"k":r[K]}); v=r[V].replace(",","")
        try: d[r[M]]=float(v)
        except: d[r[M]]=v
    out = {}
    for d in launches.values():
        ver = classify(d["k"])
        if ver is None: continue
        dur = d.get("gpu__time_duration.sum",0)
        if ver not in out or dur > out[ver].get("gpu__time_duration.sum",0):
            out[ver] = d
    return out

def flops(d):
    return (d.get("smsp__sass_thread_inst_executed_op_dadd_pred_on.sum",0)
          + d.get("smsp__sass_thread_inst_executed_op_dmul_pred_on.sum",0)
          + 2*d.get("smsp__sass_thread_inst_executed_op_dfma_pred_on.sum",0))

# --- throughput (Gdof/s) by benchmark id from the stdout CSV rows ---
gdofs = {}
for line in open(THR):
    m = re.search(r"(\d{6,}),([\d.eE+-]+),(\d+),8,[^,]*,([\d.eE+]+)", line)
    if m:
        bid = int(m.group(3)); gd = float(m.group(4))/1e9
        # first occurrence = clean STEP-1 throughput; later dup = ncu-inflated, ignore
        if bid < len(VERS) and VERS[bid] not in gdofs: gdofs[VERS[bid]] = gd

flo_new = load_ncu(FLO)       # v01..current FLOPs
flo_old = load_ncu(ORIG)      # v00a/v00b FLOPs (and others, unused)
spill   = load_ncu(SPL)       # regs/dram/shmem/spill all 14

H100_FP64=27100.0; H100_HBM=3350.0  # GFLOP/s, GB/s

rows=[]
for v in VERS:
    s = spill.get(v, {})
    regs = int(s.get("launch__registers_per_thread",0))
    shm  = int(s.get("launch__shared_mem_per_block_allocated",0))
    dram = s.get("dram__bytes.sum",0)
    spb  = s.get("l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum",0)+s.get("l1tex__t_bytes_pipe_lsu_mem_local_op_st.sum",0)
    fl   = flops(flo_new[v]) if v in flo_new else (flops(flo_old[v]) if v in flo_old else 0)
    gd   = gdofs.get(v, 0.0)
    dur  = DOFS/(gd*1e9) if gd else 0.0          # clean matvec time from throughput
    gflops = fl/dur/1e9 if dur else 0.0
    ai   = fl/dram if dram else 0.0
    pct  = 100*gflops/min(ai*H100_HBM, H100_FP64) if ai else 0.0
    rows.append(dict(v=v, gd=gd, gflops=gflops, ai=ai, pct=pct, regs=regs,
                     shm=shm, dram=dram, spill=spb, dur=dur))

# --- table (stdout + markdown + csv) ---
hdr = f"{'ver':7}{'Gdof/s':>8}{'GFLOP/s':>9}{'AI':>7}{'%HBM':>6}{'regs':>5}{'shmemKB':>8}{'DRAM_GB':>9}{'spill_MB':>10}{'dur_ms':>9}"
print(hdr); print("-"*len(hdr))
for r in rows:
    print(f"{r['v']:7}{r['gd']:>8.3f}{r['gflops']:>9.0f}{r['ai']:>7.2f}{r['pct']:>6.0f}"
          f"{r['regs']:>5}{r['shm']/1024:>8.1f}{r['dram']/1e9:>9.2f}{r['spill']/1e6:>10.1f}{r['dur']*1e3:>9.2f}")

with open(f"{DOC}/history_stats_remeasured.csv","w") as f:
    f.write("version,gdofs_per_s,gflops,ai_dram,pct_hbm_roof,regs_per_thread,shmem_bytes,dram_bytes,local_spill_bytes,matvec_s\n")
    for r in rows:
        f.write(f"{r['v']},{r['gd']:.4g},{r['gflops']:.0f},{r['ai']:.3f},{r['pct']:.1f},{r['regs']},{r['shm']},{int(r['dram'])},{int(r['spill'])},{r['dur']:.4e}\n")

with open(f"{DOC}/history_stats_remeasured.md","w") as f:
    f.write("| version | Gdof/s | GFLOP/s | AI (F/B) | % HBM roof | regs | shmem KB | DRAM GB | spill MB | matvec ms |\n")
    f.write("|---|---|---|---|---|---|---|---|---|---|\n")
    for r in rows:
        f.write(f"| {r['v']} | {r['gd']:.3f} | {r['gflops']:.0f} | {r['ai']:.2f} | {r['pct']:.0f} | "
                f"{r['regs']} | {r['shm']/1024:.1f} | {r['dram']/1e9:.2f} | {r['spill']/1e6:.1f} | {r['dur']*1e3:.2f} |\n")

# --- tightened phase-summary table (matches the roofline figure) ---
phase_def=[("baseline","v00a","v00a"),
           ("1. arithmetic reduction","v00a-v02b","v02b"),
           ("2. teams + shared memory","v03-v05","v05"),
           ("3. data layout & tiling","v06-v08","v08"),
           ("4. register / occupancy tuning","v09-v10","v10")]
rb={r['v']:r for r in rows}
print("\n=== Tightened phase history ===")
ph=f"{'phase':32}{'versions':12}{'end':6}{'Gdof/s':>8}{'gain':>7}{'AI':>6}{'GFLOP/s':>9}{'regs':>5}{'DRAM_GB':>9}{'spill_MB':>10}"
print(ph); print("-"*len(ph))
prev=None; phase_rows=[]
for name,vr,end in phase_def:
    r=rb[end]; gain=(r['gd']/prev) if prev else None; prev=r['gd']
    gtxt=f"{gain:.1f}x" if gain else "—"
    print(f"{name:32}{vr:12}{end:6}{r['gd']:>8.3f}{gtxt:>7}{r['ai']:>6.2f}{r['gflops']:>9.0f}{r['regs']:>5}{r['dram']/1e9:>9.2f}{r['spill']/1e6:>10.1f}")
    phase_rows.append((name,vr,end,r,gtxt))
cum=rb['v10']['gd']/rb['v00a']['gd']
print(f"cumulative v00a -> v10: {cum:.0f}x")
with open(f"{DOC}/history_phases.md","w") as f:
    f.write("| Phase | Versions | Endpoint | Gdof/s | Gain | AI (F/B) | GFLOP/s | regs | DRAM GB | spill MB |\n")
    f.write("|---|---|---|---|---|---|---|---|---|---|\n")
    for name,vr,end,r,gtxt in phase_rows:
        f.write(f"| {name} | {vr} | {end} | {r['gd']:.3f} | {gtxt} | {r['ai']:.2f} | {r['gflops']:.0f} | "
                f"{r['regs']} | {r['dram']/1e9:.2f} | {r['spill']/1e6:.1f} |\n")
    f.write(f"\nCumulative v00a → v10: **{cum:.0f}×**\n")
with open(f"{DOC}/history_phases.csv","w") as f:
    f.write("phase,versions,endpoint,gdofs_per_s,gain,ai_dram,gflops,regs,dram_bytes,spill_bytes\n")
    for name,vr,end,r,gtxt in phase_rows:
        f.write(f'"{name}",{vr},{end},{r["gd"]:.4g},{gtxt},{r["ai"]:.3f},{r["gflops"]:.0f},{r["regs"]},{int(r["dram"])},{int(r["spill"])}\n')

# --- roofline-history figure ---
ai_axis = np.logspace(-2, 2, 500)
fig, ax = plt.subplots(figsize=(13,9))
hbm_line,  = ax.loglog(ai_axis, np.minimum(ai_axis*H100_HBM, H100_FP64), 'C0', lw=4,
                       label=f'HBM3 roof ({H100_HBM:.0f} GB/s)')
peak_line, = ax.loglog(ai_axis, np.full_like(ai_axis, H100_FP64), 'gray', lw=2.5, ls='--',
                       label=f'FP64 peak ({H100_FP64/1000:.1f} TFLOP/s)')

# Plot only the phase endpoints: start at v00a, then one point per optimization
# phase (its last version). Each arrow == one phase.
rowsby={r['v']:r for r in rows}
phases=[("v00a","baseline"),
        ("v02b","arithmetic reduction"),
        ("v05","teams + shared memory"),
        ("v08","data layout & tiling"),
        ("v10","register / occupancy tuning")]
pts=[(rowsby[v],lbl) for v,lbl in phases]
# trajectory: one arrow per phase
for (a,_),(b,_) in zip(pts[:-1], pts[1:]):
    ax.annotate("", xy=(b['ai'],b['gflops']), xytext=(a['ai'],a['gflops']),
                arrowprops=dict(arrowstyle="->", color="0.55", lw=1.6, alpha=.8))
markers=['o','s','^','D','*']
colors=cm.plasma(np.linspace(0.1,0.9,len(pts)))
ver_handles=[]
for (r,lbl),mk,col in zip(pts, markers, colors):
    h=ax.scatter([r['ai']],[r['gflops']], marker=mk, s=300, color=col,
                 edgecolors="black", linewidths=1.4, zorder=5,
                 label=f"{r['v']}: {lbl} ({r['gd']:.2g} Gdof/s)")
    ver_handles.append(h)

ax.set_xlabel("Arithmetic Intensity (FLOP / DRAM byte)", fontsize=20)
ax.set_ylabel("Performance (GFLOP/s)", fontsize=20)
ax.set_title("EpsDivDiv history on the H100 roofline", fontsize=20, fontweight="bold")
ax.set_xlim(0.06,70); ax.set_ylim(50,4e4); ax.tick_params(labelsize=14)
ax.grid(True, which="both", ls=":", alpha=.5)
# both legends inside the plot, lower-right: versions stacked above the roofs
roof_leg=ax.legend(handles=[hbm_line,peak_line], loc="lower right",
                   bbox_to_anchor=(1.0,0.0), fontsize=12, framealpha=.95)
ax.add_artist(roof_leg)
ax.legend(handles=ver_handles, loc="lower right", bbox_to_anchor=(0.99,0.14),
          fontsize=10, framealpha=.95, title="Optimization phase (endpoint throughput)",
          title_fontsize=10)
plt.tight_layout()
out=f"{DOC}/roofline_history_remeasured_h100.png"
plt.savefig(out, dpi=140, bbox_inches="tight"); print("\nSaved figure:", out)
print("Saved table: history_stats_remeasured.{md,csv}")
