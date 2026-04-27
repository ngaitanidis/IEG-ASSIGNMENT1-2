"""
Part d — Multi-node Transmission Model (DK–DE–SE–NO)
=====================================================
Extends the model to four countries connected by HVAC lines using a
linearised DC power flow approximation. Co-optimises generation across
the whole system.

Run:  python part_d.py
"""

from common import (load_data, build_multi_node, CO2_PRICE,
                    CARRIER_COLORS, colors, save_fig,
                    plot_annual_pie)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

YEAR   = 2019
SOLVER = "gurobi"

# ── Load data & solve ─────────────────────────────────────────────────────────
data      = load_data(YEAR)
demand    = data["demand"]
hours     = data["hours"]

demand_DE    = 4.0 * demand
demand_SE    = 2.0 * demand
demand_NO    = 1.5 * demand
total_demand = demand.sum() + demand_DE.sum() + demand_SE.sum() + demand_NO.sum()
system_demand_gw = pd.Series(
    (demand + demand_DE + demand_SE + demand_NO) / 1000, index=hours)

n = build_multi_node(data, co2_price=CO2_PRICE)
n.optimize(solver_name=SOLVER)

# ── Results ───────────────────────────────────────────────────────────────────
print(f"\n{'='*75}\nOPTIMAL CAPACITIES — Part d\n{'='*75}")
for gen in n.generators.index:
    cap = n.generators.loc[gen, "p_nom_opt"]
    if cap > 0:
        cf = n.generators_t.p[gen].sum() / (cap * 8760)
        print(f"  {gen:<28} {cap/1000:>6.2f} GW   CF {cf:.1%}")

print(f"\n  Total system cost {n.objective:>15,.0f} €/yr")
print(f"  Cost/MWh (system) {n.objective / total_demand:.2f} €/MWh")

print("\n  Line flows (mean absolute)")
for line in n.lines.index:
    print(f"  {line:<10} {n.lines_t.p0[line].abs().mean():.0f} MW")

print("\n  Average electricity prices")
for bus in n.buses.index:
    print(f"  {bus}  {n.buses_t.marginal_price[bus].mean():.2f} €/MWh")

# First-hour nodal imbalances (used in Part e for PTDF verification)
snap0 = n.snapshots[0]
gen0  = n.generators_t.p.loc[snap0].groupby(n.generators.bus).sum()
load0 = n.loads_t.p_set.loc[snap0].groupby(n.loads.bus).sum()
buses = ["DK", "DE", "SE", "NO"]
p0    = (gen0.reindex(buses, fill_value=0)
          - load0.reindex(buses, fill_value=0))
print("\n  First-hour nodal imbalances [MW]:")
print(p0.to_string())
print("\n  First-hour line flows p0 [MW]:")
print(n.lines_t.p0.loc[snap0].to_string())

# ── Plots ─────────────────────────────────────────────────────────────────────
demand_gw        = pd.Series(demand / 1000, index=hours)
dispatch_carrier = n.generators_t.p.T.groupby(n.generators.carrier).sum().T / 1000

for period, start, end, label in [
    ("winter", f"{YEAR}-01-14", f"{YEAR}-01-20", "Winter week (Jan 14–20)"),
    ("summer", f"{YEAR}-07-15", f"{YEAR}-07-21", "Summer week (Jul 15–21)"),
]:
    fig, ax = plt.subplots(figsize=(12, 5))
    w = dispatch_carrier.loc[start:end]
    w.plot.area(ax=ax, linewidth=0, color=colors(w.columns))
    system_demand_gw.loc[start:end].plot(ax=ax, color="black", lw=2,
                                          ls="--", label="Total system demand")
    ax.set_ylabel("Power [GW]")
    ax.set_title(f"System dispatch — {label}")
    ax.legend(loc="upper right", frameon=False)
    ax.set_ylim(0)
    save_fig(fig, f"d_{period}_dispatch.png")

plot_annual_pie(n, f"System annual mix {YEAR}", "d_annual_mix.png")

# Line flow duration curves
fig, ax = plt.subplots(figsize=(10, 5))
for line in n.lines.index:
    s = n.lines_t.p0[line].abs().sort_values(ascending=False).values / 1000
    ax.plot(np.arange(len(s)), s, lw=1.5, label=line)
ax.set_xlabel("Hours (sorted)")
ax.set_ylabel("Absolute flow [GW]")
ax.set_title("Line flow duration curves")
ax.legend(frameon=False)
ax.set_xlim(0, 8760)
save_fig(fig, "d_line_duration_curves.png")

# Optimal capacities by generator
cap_s  = n.generators.p_nom_opt[n.generators.p_nom_opt > 0] / 1000
clrs   = [CARRIER_COLORS.get(n.generators.loc[g, "carrier"], "gray")
          for g in cap_s.index]
fig, ax = plt.subplots(figsize=(12, 5))
cap_s.plot.bar(ax=ax, color=clrs, edgecolor="black")
ax.set_ylabel("Capacity [GW]")
ax.set_title("Optimal installed capacities by generator")
for i, (_, v) in enumerate(cap_s.items()):
    ax.text(i, v + 0.1, f"{v:.2f}", ha="center", fontsize=9)
plt.xticks(rotation=45, ha="right")
save_fig(fig, "d_optimal_capacities.png")

# Average prices by country
fig, ax = plt.subplots(figsize=(7, 4))
avg = n.buses_t.marginal_price.mean()
avg.plot.bar(ax=ax, color="steelblue", edgecolor="black")
ax.set_ylabel("Average price [€/MWh]")
ax.set_title("Average electricity prices by country")
for i, (_, v) in enumerate(avg.items()):
    ax.text(i, v + 0.5, f"{v:.1f}", ha="center", fontsize=10)
plt.xticks(rotation=0)
save_fig(fig, "d_average_prices.png")

print("\n✅  Part d done — plots saved to outputs/")