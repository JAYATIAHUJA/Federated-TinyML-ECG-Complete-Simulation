"""
Script 06: Generate All Paper Figures
======================================
Reads results from the previous scripts and generates
publication-quality figures for your paper.

Output:
  figures/fig1_privacy_accuracy.png   -> Figure 1 in paper
  figures/fig2_memory_breakdown.png   -> Figure 2 in paper
  figures/fig3_lmic_dropout.png       -> Figure 3 in paper
  figures/fig4_convergence_curves.png -> Figure 4 (bonus - FL curves)
"""

import os, json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

os.makedirs("./figures", exist_ok=True)

# ── STYLE SETUP ───────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        12,
    "axes.titlesize":   13,
    "axes.labelsize":   12,
    "xtick.labelsize":  11,
    "ytick.labelsize":  11,
    "legend.fontsize":  10,
    "figure.dpi":       150,
    "axes.spines.top":  False,
    "axes.spines.right":False,
})

CLINICAL_THRESHOLD = 0.80   # minimum acceptable F1 for screening

# ── LOAD RESULTS ──────────────────────────────────────────────────────────────
print("Loading results...")

with open("./results/privacy_sweep.json") as f:
    privacy_data = json.load(f)

with open("./results/lmic_dropout.json") as f:
    lmic_data = json.load(f)

with open("./results/resource_profile.json") as f:
    resource_data = json.load(f)

# ── FIGURE 1: Privacy–Accuracy Trade-off Curve ────────────────────────────────
print("Generating Figure 1: Privacy–Accuracy Trade-off...")

eps_order  = ["0.5", "1.0", "2.0", "5.0", "inf"]
eps_labels = ["ε=0.5\n(Strongest)", "ε=1.0\n(Strong)", "ε=2.0\n(Moderate★)",
              "ε=5.0\n(Weak)", "No DP\n(Baseline)"]

f1_vals   = [privacy_data[e]["final_f1"]          for e in eps_order]
sens_vals = [privacy_data[e]["final_sensitivity"] for e in eps_order]
spec_vals = [privacy_data[e]["final_specificity"] for e in eps_order]

fig, ax = plt.subplots(figsize=(8, 5))

x = np.arange(len(eps_order))
ax.plot(x, f1_vals,   "o-", color="#2E86AB", lw=2.5, ms=9, label="F1-Score",    zorder=3)
ax.plot(x, sens_vals, "s--", color="#E76F51", lw=2,   ms=8, label="Sensitivity", zorder=3)
ax.plot(x, spec_vals, "^--", color="#57A773", lw=2,   ms=8, label="Specificity", zorder=3)

ax.axhline(CLINICAL_THRESHOLD, ls=":", color="red", lw=1.8,
           label=f"Clinical minimum (F1 = {CLINICAL_THRESHOLD})", zorder=2)

# Shade the "too-private" region
ax.axvspan(-0.5, 1.5, alpha=0.06, color="red", label="Below clinical threshold risk")
# Mark recommended point
rec_x = 2   # index of ε=2.0
ax.axvline(rec_x, color="#2E86AB", ls="--", alpha=0.4, lw=1.5)
ax.annotate("Recommended\noperating point",
            xy=(rec_x, f1_vals[rec_x]),
            xytext=(rec_x + 0.3, f1_vals[rec_x] - 0.05),
            arrowprops=dict(arrowstyle="->", color="#333333"),
            fontsize=10, color="#333333")

ax.set_xticks(x)
ax.set_xticklabels(eps_labels)
ax.set_xlabel("Differential Privacy Budget (ε)")
ax.set_ylabel("Score")
ax.set_ylim(0.45, 1.02)
ax.set_title("Figure 1: Privacy–Accuracy Trade-off\n(PTB-XL, N=10 Clients, 20 FL Rounds, FedAvg + Opacus)")
ax.legend(loc="lower left", framealpha=0.9)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("./figures/fig1_privacy_accuracy.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: figures/fig1_privacy_accuracy.png")

# ── FIGURE 2: Memory Breakdown Stacked Bar ────────────────────────────────────
print("Generating Figure 2: Memory Breakdown...")

int8_kb    = resource_data["int8_file_kb"]
fp32_kb    = resource_data["fp32_param_kb"]
act_kb     = resource_data["activation_buffer_kb_est"]
dp_kb      = resource_data["dp_buffer_kb_est"]
total_kb   = resource_data["total_sram_est_kb"]
mcu_limit  = resource_data["mcu_limit_kb"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

# Left: FP32 vs INT8 comparison
categories  = ["FP32 Model\n(unquantised)", "INT8 Model\n(quantised)"]
sizes       = [fp32_kb, int8_kb]
colors_bar  = ["#E76F51", "#2E86AB"]
bars = ax1.bar(categories, sizes, color=colors_bar, edgecolor="white",
               width=0.4, zorder=3)
ax1.axhline(mcu_limit, ls="--", color="red", lw=2,
            label=f"256 KB MCU SRAM limit", zorder=4)
for bar, val in zip(bars, sizes):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
             f"{val:.1f} KB", ha="center", fontsize=11, fontweight="bold")
ax1.text(0.5, fp32_kb * 0.5, f"{fp32_kb/int8_kb:.1f}× smaller →",
         ha="center", va="center", fontsize=10, color="white",
         transform=ax1.transData)
ax1.set_ylabel("Size (KB)")
ax1.set_title("Model Size:\nFP32 vs INT8 Quantisation")
ax1.legend(fontsize=10)
ax1.set_ylim(0, mcu_limit + 40)
ax1.grid(axis="y", alpha=0.3, zorder=0)

# Right: Full SRAM breakdown (stacked)
components = ["INT8 Model\nParameters", "Activation\nBuffer (est.)",
              "DP Noise\nBuffer (est.)"]
comp_vals  = [int8_kb, act_kb, dp_kb]
comp_cols  = ["#2E86AB", "#A8DADC", "#E76F51"]
cumulative = 0
for val, color, label in zip(comp_vals, comp_cols, components):
    ax2.bar(0, val, bottom=cumulative, color=color, edgecolor="white",
            width=0.4, label=f"{label} ({val:.0f} KB)", zorder=3)
    ax2.text(0, cumulative + val/2, f"{val:.0f} KB",
             ha="center", va="center", fontsize=10,
             fontweight="bold", color="white")
    cumulative += val

ax2.bar(0, 0, color="none")  # invisible bar for total label
ax2.axhline(mcu_limit, ls="--", color="red", lw=2,
            label=f"256 KB MCU limit (headroom: {mcu_limit-total_kb:.0f} KB)")
ax2.text(0.25, total_kb + 5, f"Total: {total_kb:.0f} KB",
         ha="left", fontsize=11, fontweight="bold", color="#264653")
ax2.set_xlim(-0.5, 1.0)
ax2.set_xticks([])
ax2.set_ylabel("SRAM Usage (KB)")
ax2.set_title("Full SRAM Footprint Breakdown\n(FL + DP + TinyML stack, ε=2.0)")
ax2.legend(loc="upper right", fontsize=9, framealpha=0.9)
ax2.set_ylim(0, mcu_limit + 40)
ax2.grid(axis="y", alpha=0.3, zorder=0)

plt.suptitle("Figure 2: TinyML Resource Profiling — MCU Memory Analysis",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("./figures/fig2_memory_breakdown.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: figures/fig2_memory_breakdown.png")

# ── FIGURE 3: LMIC Dropout Simulation ────────────────────────────────────────
print("Generating Figure 3: LMIC Connectivity Simulation...")

dr_keys    = ["0", "20", "40", "60"]
dr_labels  = ["0%\n(Ideal)", "20%\n(Urban LMIC)", "40%\n(Rural LMIC)", "60%\n(Remote LMIC)"]
dr_f1      = [lmic_data[k]["final_f1"]          for k in dr_keys]
dr_sens    = [lmic_data[k]["final_sensitivity"] for k in dr_keys]
dr_spec    = [lmic_data[k]["final_specificity"] for k in dr_keys]
clients_rnd= [lmic_data[k]["clients_per_round"] for k in dr_keys]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Left: F1 bar chart with scenario colour coding
scenario_colors = ["#2ECC71", "#F39C12", "#E67E22", "#E74C3C"]
bars = ax1.bar(dr_labels, dr_f1, color=scenario_colors, edgecolor="white",
               width=0.5, zorder=3)
ax1.axhline(CLINICAL_THRESHOLD, ls="--", color="red", lw=2,
            label=f"Clinical minimum (F1 = {CLINICAL_THRESHOLD})")

for bar, val, clients in zip(bars, dr_f1, clients_rnd):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.006,
             f"F1={val:.3f}", ha="center", fontsize=10, fontweight="bold")
    ax1.text(bar.get_x() + bar.get_width()/2, 0.48,
             f"{clients}/{len(dr_keys)*2+2} clients", ha="center",
             fontsize=9, color="white", fontweight="bold")

ax1.set_ylabel("F1-Score")
ax1.set_xlabel("Client Dropout Rate (LMIC Network Condition)")
ax1.set_title("Accuracy vs. LMIC Connectivity\n(ε=2.0, N=10 Clients, 20 FL Rounds)")
ax1.set_ylim(0.45, 1.02)
ax1.legend(fontsize=10)
ax1.grid(axis="y", alpha=0.3, zorder=0)

# Shade "below clinical threshold" region
ax1.axhspan(0.45, CLINICAL_THRESHOLD, alpha=0.06, color="red")
ax1.text(3.3, 0.60, "Below clinical\nminimum", fontsize=9,
         color="red", alpha=0.8, ha="right")

# Right: All three metrics grouped bar
x     = np.arange(len(dr_keys))
width = 0.25
ax2.bar(x - width, dr_f1,   width, label="F1-Score",    color="#2E86AB", zorder=3)
ax2.bar(x,         dr_sens,  width, label="Sensitivity", color="#E76F51", zorder=3)
ax2.bar(x + width, dr_spec,  width, label="Specificity", color="#57A773", zorder=3)
ax2.axhline(CLINICAL_THRESHOLD, ls="--", color="red", lw=1.5, alpha=0.8)
ax2.set_xticks(x)
ax2.set_xticklabels(dr_labels)
ax2.set_ylabel("Score")
ax2.set_title("All Metrics vs. Dropout Rate")
ax2.set_ylim(0.45, 1.02)
ax2.legend(fontsize=10)
ax2.grid(axis="y", alpha=0.3, zorder=0)

plt.suptitle("Figure 3: LMIC Intermittent Connectivity Simulation (ε=2.0)",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("./figures/fig3_lmic_dropout.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: figures/fig3_lmic_dropout.png")

# ── FIGURE 4: FL Convergence Curves (bonus) ───────────────────────────────────
print("Generating Figure 4: FL Convergence Curves (bonus figure)...")

eps_plot_order  = ["0.5", "1.0", "2.0", "5.0", "inf"]
eps_plot_labels = ["ε=0.5", "ε=1.0", "ε=2.0 ★", "ε=5.0", "No DP"]
colors_eps = ["#E74C3C", "#E67E22", "#2E86AB", "#9B59B6", "#2ECC71"]
linestyles = ["--", "--", "-", "--", ":"]

fig, ax = plt.subplots(figsize=(9, 5))

for eps, label, color, ls in zip(eps_plot_order, eps_plot_labels,
                                   colors_eps, linestyles):
    rounds_data = privacy_data[eps].get("f1_per_round", [])
    if rounds_data:
        rounds = list(range(1, len(rounds_data) + 1))
        lw = 2.5 if eps == "2.0" else 1.8
        ax.plot(rounds, rounds_data, color=color, ls=ls, lw=lw, label=label,
                marker="." if eps == "2.0" else None, ms=5)

ax.axhline(CLINICAL_THRESHOLD, ls=":", color="red", lw=1.5, alpha=0.8,
           label=f"Clinical minimum (F1={CLINICAL_THRESHOLD})")
ax.set_xlabel("Federated Round")
ax.set_ylabel("F1-Score (aggregated across clients)")
ax.set_title("Figure 4: FL Convergence Curves by Privacy Budget\n(PTB-XL, N=10 Clients, FedAvg + Opacus DP)")
ax.legend(loc="lower right", framealpha=0.9)
ax.grid(alpha=0.3)
ax.set_xlim(1, 20)
ax.set_ylim(0.45, 1.0)

plt.tight_layout()
plt.savefig("./figures/fig4_convergence_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: figures/fig4_convergence_curves.png")

# ── SUMMARY ───────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("ALL FIGURES GENERATED")
print("="*60)
print("\nFiles to insert into your paper Word document:")
print("  figures/fig1_privacy_accuracy.png    -> Section 6.2 (Figure 1)")
print("  figures/fig2_memory_breakdown.png    -> Section 6.3 (Figure 2)")
print("  figures/fig3_lmic_dropout.png        -> Section 6.4 (Figure 3)")
print("  figures/fig4_convergence_curves.png  -> Section 6.2 (optional Figure 4)")
print("\nKey numbers for your paper tables:")
print(f"  Baseline F1 (No DP, 0% dropout):   {privacy_data['inf']['final_f1']:.4f}")
print(f"  F1 at epsilon=2.0:                        {privacy_data['2.0']['final_f1']:.4f}")
print(f"  F1 at epsilon=0.5:                        {privacy_data['0.5']['final_f1']:.4f}")
print(f"  F1 at 40% LMIC dropout:             {lmic_data['40']['final_f1']:.4f}")
print(f"  INT8 model size:                    {resource_data['int8_file_kb']:.1f} KB")
print(f"  Estimated total SRAM:               {resource_data['total_sram_est_kb']:.1f} KB")
print(f"  Fits in 256 KB MCU SRAM:            {resource_data['fits_in_mcu_sram']}")
print(f"  CPU inference time:                 {resource_data['inference_time_ms']:.2f} ms")
