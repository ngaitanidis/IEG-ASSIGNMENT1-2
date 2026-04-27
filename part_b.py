"""
Part b — Interannual Variability (2016–2019)
============================================
Runs the same single-node optimisation for each weather year and compares
optimal capacities and system cost across years.

Run:  python part_b.py
"""

from common import (load_data, build_single_node, CO2_PRICE,
                    CARRIER_COLORS, colors, save_fig)
import numpy as np
import matplotlib.pyplot as plt

YEARS    = [2016, 2017, 2018, 2019]
CARRIERS = ["onshore wind", "offshore wind", "solar",
            "CCGT", "OCGT", "biomass"]
SOLVER   = "gurobi"

# ── Run optimisation for each year ────────────────────────────────────────────
results = {}

for year in YEARS:
    print(f"\n  Optimising {year}...")
    data = load_data(year)
    n    = build_single_node(data, co2_price=CO2_PRICE)
    n.optimize(solver_name=SOLVER, solver_options={"OutputFlag": 0})

    row = {"cost_per_mwh": n.objective / data["demand"].sum()}
    for gen in n.generators.index:
        c = n.generators.loc[gen, "carrier"]
        row[f"cap_{c}"] = n.generators.loc[gen, "p_nom_opt"] / 1000   # GW
        row[f"gen_{c}"] = n.generators_t.p[gen].sum() / 1e6           # TWh
    results[year] = row
    print(f"  Cost {row['cost_per_mwh']:.2f} €/MWh")

# ── Summary table ─────────────────────────────────────────────────────────────
print(f"\n{'='*70}\nCOMPARISON — Part b\n{'='*70}")
print(f"\n{'Capacity [GW]':<22}", end="")
for y in YEARS:
    print(f"  {y:>8}", end="")
print(f"  {'Mean':>8}  {'Std':>8}")
print("-" * 70)
for c in CARRIERS:
    vals = [results[y].get(f"cap_{c}", 0) for y in YEARS]
    print(f"  {c:<20}", end="")
    for v in vals:
        print(f"  {v:>8.2f}", end="")
    print(f"  {np.mean(vals):>8.2f}  {np.std(vals):>8.2f}")
costs = [results[y]["cost_per_mwh"] for y in YEARS]
print(f"\n  {'Cost [€/MWh]':<20}", end="")
for c in costs:
    print(f"  {c:>8.2f}", end="")
print(f"  {np.mean(costs):>8.2f}  {np.std(costs):>8.2f}")

# ── Plots ─────────────────────────────────────────────────────────────────────
active = [c for c in CARRIERS
          if max(results[y].get(f"cap_{c}", 0) for y in YEARS) > 0.01]
x, width = np.arange(len(YEARS)), 0.13

# Grouped capacity bar by year
fig, ax = plt.subplots(figsize=(12, 5))
for i, c in enumerate(active):
    vals   = [results[y][f"cap_{c}"] for y in YEARS]
    offset = (i - len(active) / 2 + 0.5) * width
    ax.bar(x + offset, vals, width, label=c,
           color=CARRIER_COLORS.get(c, "gray"), edgecolor="black", linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(YEARS)
ax.set_ylabel("Capacity [GW]")
ax.set_title("Optimal capacities across weather years")
ax.legend(frameon=False)
fig.savefig("outputs/b_capacities_by_year.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# Mean ± range
means, mins, maxs, labels = [], [], [], []
for c in CARRIERS:
    vals = [results[y].get(f"cap_{c}", 0) for y in YEARS]
    if np.mean(vals) > 0.01:
        means.append(np.mean(vals))
        mins.append(np.mean(vals) - np.min(vals))
        maxs.append(np.max(vals) - np.mean(vals))
        labels.append(c)
xi  = np.arange(len(labels))
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(xi, means, color=colors(labels), edgecolor="black", linewidth=0.5)
ax.errorbar(xi, means, yerr=[mins, maxs], fmt="none",
            ecolor="black", capsize=5, capthick=1.5, linewidth=1.5)
ax.set_xticks(xi)
ax.set_xticklabels(labels)
ax.set_ylabel("Capacity [GW]")
ax.set_title("Average optimal capacity ± interannual variability (2016–2019)")
for i, m in enumerate(means):
    ax.text(i, m + maxs[i] + 0.1, f"{m:.2f}", ha="center", fontsize=10)
fig.savefig("outputs/b_average_variability.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# System cost bar
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(YEARS, costs, color="steelblue", edgecolor="black")
ax.axhline(np.mean(costs), color="red", ls="--", lw=1.5,
           label=f"Mean {np.mean(costs):.1f} €/MWh")
ax.set_ylabel("System cost [€/MWh]")
ax.set_title("System cost across weather years")
ax.legend()
for y, c in zip(YEARS, costs):
    ax.text(y, c + 0.3, f"{c:.1f}", ha="center", fontsize=10)
fig.savefig("outputs/b_system_cost.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# Stacked generation mix
fig, ax = plt.subplots(figsize=(10, 5))
bottom = np.zeros(len(YEARS))
for c in CARRIERS:
    vals = [results[y].get(f"gen_{c}", 0) for y in YEARS]
    if max(vals) > 0.01:
        ax.bar(YEARS, vals, bottom=bottom, label=c,
               color=CARRIER_COLORS.get(c, "gray"),
               edgecolor="black", linewidth=0.3)
        bottom += np.array(vals)
ax.set_ylabel("Annual generation [TWh]")
ax.set_title("Generation mix across weather years")
ax.legend(frameon=False)
fig.savefig("outputs/b_generation_mix.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print("\n✅  Part b done — plots saved to outputs/")