#!/usr/bin/env python3
"""
Plot convergence comparison: 1-point vs 6-point quadrature for EpsDivDiv Stokes.
Reads convergence_1pt_vs_6pt.csv and produces two log-log plots (velocity and pressure error vs h).
"""

import csv
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def main():
    csv_file = "convergence_1pt_vs_6pt.csv"
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]

    levels, h_vel, h_pre = [], [], []
    l2_vel_1pt, l2_pre_1pt = [], []
    l2_vel_6pt, l2_pre_6pt = [], []

    with open(csv_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            levels.append(int(row["level"]))
            h_vel.append(float(row["h_vel"]))
            h_pre.append(float(row["h_pre"]))
            l2_vel_1pt.append(float(row["l2_vel_1pt"]))
            l2_pre_1pt.append(float(row["l2_pre_1pt"]))
            l2_vel_6pt.append(float(row["l2_vel_6pt"]))
            l2_pre_6pt.append(float(row["l2_pre_6pt"]))

    h_vel = np.array(h_vel)
    h_pre = np.array(h_pre)
    l2_vel_1pt = np.array(l2_vel_1pt)
    l2_pre_1pt = np.array(l2_pre_1pt)
    l2_vel_6pt = np.array(l2_vel_6pt)
    l2_pre_6pt = np.array(l2_pre_6pt)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # --- Velocity error vs h ---
    ax1.loglog(h_vel, l2_vel_1pt, "o-", label="1pt quad", linewidth=2, markersize=8)
    ax1.loglog(h_vel, l2_vel_6pt, "s-", label="6pt quad", linewidth=2, markersize=8)
    # reference slopes
    ref_h = np.array([h_vel[0], h_vel[-1]])
    ax1.loglog(ref_h, l2_vel_6pt[0] * (ref_h / ref_h[0])**2, "k--", alpha=0.4, label="O(h$^2$)")
    ax1.loglog(ref_h, l2_vel_6pt[0] * (ref_h / ref_h[0])**3, "k:", alpha=0.4, label="O(h$^3$)")
    ax1.set_xlabel("h (velocity mesh size)")
    ax1.set_ylabel("L2 error (velocity)")
    ax1.set_title("Velocity convergence")
    ax1.legend()
    ax1.grid(True, which="both", alpha=0.3)
    # annotate convergence orders
    for i in range(1, len(h_vel)):
        rate_1pt = np.log(l2_vel_1pt[i-1] / l2_vel_1pt[i]) / np.log(2)
        rate_6pt = np.log(l2_vel_6pt[i-1] / l2_vel_6pt[i]) / np.log(2)
        ax1.annotate(f"{rate_1pt:.1f}", (h_vel[i], l2_vel_1pt[i]),
                     textcoords="offset points", xytext=(8, 5), fontsize=8, color="C0")
        ax1.annotate(f"{rate_6pt:.1f}", (h_vel[i], l2_vel_6pt[i]),
                     textcoords="offset points", xytext=(8, -12), fontsize=8, color="C1")

    # --- Pressure error vs h ---
    ax2.loglog(h_pre, l2_pre_1pt, "o-", label="1pt quad", linewidth=2, markersize=8)
    ax2.loglog(h_pre, l2_pre_6pt, "s-", label="6pt quad", linewidth=2, markersize=8)
    ref_h_p = np.array([h_pre[0], h_pre[-1]])
    ax2.loglog(ref_h_p, l2_pre_6pt[0] * (ref_h_p / ref_h_p[0])**1, "k--", alpha=0.4, label="O(h$^1$)")
    ax2.loglog(ref_h_p, l2_pre_6pt[0] * (ref_h_p / ref_h_p[0])**2, "k:", alpha=0.4, label="O(h$^2$)")
    ax2.set_xlabel("h (pressure mesh size)")
    ax2.set_ylabel("L2 error (pressure)")
    ax2.set_title("Pressure convergence")
    ax2.legend()
    ax2.grid(True, which="both", alpha=0.3)
    for i in range(1, len(h_pre)):
        rate_1pt = np.log(l2_pre_1pt[i-1] / l2_pre_1pt[i]) / np.log(2)
        rate_6pt = np.log(l2_pre_6pt[i-1] / l2_pre_6pt[i]) / np.log(2)
        ax2.annotate(f"{rate_1pt:.1f}", (h_pre[i], l2_pre_1pt[i]),
                     textcoords="offset points", xytext=(8, 5), fontsize=8, color="C0")
        ax2.annotate(f"{rate_6pt:.1f}", (h_pre[i], l2_pre_6pt[i]),
                     textcoords="offset points", xytext=(8, -12), fontsize=8, color="C1")

    fig.suptitle("EpsDivDiv Stokes: 1pt vs 6pt quadrature convergence", fontsize=13)
    fig.tight_layout()
    fig.savefig("convergence_1pt_vs_6pt.png", dpi=150)
    fig.savefig("convergence_1pt_vs_6pt.pdf")
    print(f"Plots saved to convergence_1pt_vs_6pt.png and convergence_1pt_vs_6pt.pdf")

if __name__ == "__main__":
    main()
