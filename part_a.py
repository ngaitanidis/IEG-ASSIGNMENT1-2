"""
Part a — Optimal Capacity Expansion (single node, Denmark, 2019)
================================================================
Optimises installed capacities for 6 technologies on a copper-plate
Denmark model. Plots dispatch, annual mix, duration curves, and shadow prices.

Run:  python part_a.py
"""

from common import (load_data, build_single_node, CO2_PRICE,
                    CARRIER_COLORS, colors, save_fig,
                    plot_dispatch, plot_annual_pie,
                    plot_duration_curves, plot_capacity_bar)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

YEAR   = 2019
SOLVER = "gurobi"

# ── Load data & solve ─────────────────────────────────────────────────────────
data = load_data(YEAR)
n    = build_single_node(data, co2_price=CO2_PRICE)
n.optimize(solver_name=SOLVER, solver_options={"OutputFlag": 0})

# ── Results ───────────────────────────────────────────────────────────────────
prices = n.buses_t.marginal_price["DK"]

print(f"\n{'='*65}\nOPTIMAL CAPACITIES — Part a\n{'='*65}")
for gen in n.generators.index:
    cap = n.generators.loc[gen, "p_nom_opt"]
    if cap > 0:
        cf = n.generators_t.p[gen].sum() / (cap * 8760)
        print(f"  {gen:<22} {cap/1000:>6.2f} GW   CF {cf:.1%}")

print(f"\n  System cost  {n.objective:>15,.0f} €/yr")
print(f"  Cost/MWh     {n.objective / data['demand'].sum():>15.2f} €/MWh")
print(f"\n  Shadow prices  mean {prices.mean():.1f}  median {prices.median():.1f}"
      f"  min {prices.min():.1f}  max {prices.max():.1f} €/MWh")
print(f"  Hours at 0 €/MWh : {(prices < 0.01).sum()}")
print(f"  Hours > 70 €/MWh : {(prices > 70).sum()}")

# ── Plots ─────────────────────────────────────────────────────────────────────
demand_gw = pd.Series(data["demand"] / 1000, index=data["hours"])

plot_dispatch(n, demand_gw, f"{YEAR}-01-14", f"{YEAR}-01-20",
              f"Dispatch — winter week (Jan 14–20, {YEAR})",
              "a_winter_dispatch.png")

plot_dispatch(n, demand_gw, f"{YEAR}-07-15", f"{YEAR}-07-21",
              f"Dispatch — summer week (Jul 15–21, {YEAR})",
              "a_summer_dispatch.png")

plot_annual_pie(n, f"Annual electricity mix {YEAR}", "a_annual_mix.png")
plot_duration_curves(n, "Duration curves", "a_duration_curves.png")
plot_capacity_bar(n, "Optimal installed capacities", "a_optimal_capacities.png")

# Capacity factor bar
cf_data = {n.generators.loc[g, "carrier"]: n.generators_t.p[g].sum()
           / (n.generators.loc[g, "p_nom_opt"] * 8760)
           for g in n.generators.index if n.generators.loc[g, "p_nom_opt"] > 0}
cf_s = pd.Series(cf_data)
fig, ax = plt.subplots(figsize=(10, 4))
cf_s.plot.bar(ax=ax, color=colors(cf_s.index), edgecolor="black")
ax.set_ylabel("Capacity factor")
ax.set_title("Achieved capacity factors")
ax.set_ylim(0, 1)
for i, (_, v) in enumerate(cf_s.items()):
    ax.text(i, v + 0.02, f"{v:.1%}", ha="center", fontsize=10)
plt.xticks(rotation=0)
save_fig(fig, "a_capacity_factors.png")

# Shadow price histogram
fig, ax = plt.subplots(figsize=(10, 4))
prices.clip(upper=150).plot.hist(ax=ax, bins=50, edgecolor="black", alpha=0.7)
ax.axvline(prices.mean(), color="red", lw=2, ls="--",
           label=f"Mean {prices.mean():.1f} €/MWh")
ax.set_xlabel("Shadow price [€/MWh]")
ax.set_ylabel("Hours")
ax.set_title("Hourly shadow price distribution")
ax.legend()
save_fig(fig, "a_shadow_prices.png")

print("\n✅  Part a done — plots saved to outputs/")