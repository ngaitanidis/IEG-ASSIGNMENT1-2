"""
===============================================================================
IEG Course Project - Part (b): Interannual Variability
===============================================================================
Runs the same optimization from Part (a) for each weather year (2016-2019)
and compares results. Same model, same costs, different weather data.

HOW TO RUN:
    1. Make sure DK_clean_2016.csv through DK_clean_2019.csv are in the same folder
    2. Run: python project_part_b.py
===============================================================================
"""

import pandas as pd
pd.options.future.infer_string = False

import pypsa
import numpy as np
import matplotlib.pyplot as plt


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  COSTS — identical to Part (a)                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

discount_rate = 0.04

def annuity(lifetime, rate):
    return rate / (1 - (1 + rate) ** (-lifetime))

onshore_wind = {"investment": 1_100_000, "fixed_om": 12_900, "variable_om": 2.7, "lifetime": 30}
offshore_wind = {"investment": 2_000_000, "fixed_om": 25_000, "variable_om": 3.3, "lifetime": 30}
solar_pv = {"investment": 420_000, "fixed_om": 7_500, "variable_om": 0, "lifetime": 30}
ccgt = {"investment": 800_000, "fixed_om": 25_000, "variable_om": 4.0, "lifetime": 25,
        "efficiency": 0.58, "fuel_cost": 25.0, "co2_emissions": 0.2}
ocgt = {"investment": 400_000, "fixed_om": 15_000, "variable_om": 4.0, "lifetime": 25,
        "efficiency": 0.40, "fuel_cost": 25.0, "co2_emissions": 0.2}
biomass = {"investment": 2_500_000, "fixed_om": 50_000, "variable_om": 5.0, "lifetime": 25,
           "efficiency": 0.33, "fuel_cost": 9.0, "co2_emissions": 0.0}

co2_price = 80

capital_cost_onshore = annuity(30, 0.04) * 1_100_000 + 12_900
capital_cost_offshore = annuity(30, 0.04) * 2_000_000 + 25_000
capital_cost_solar = annuity(30, 0.04) * 420_000 + 7_500
capital_cost_ccgt = annuity(25, 0.04) * 800_000 + 25_000
capital_cost_ocgt = annuity(25, 0.04) * 400_000 + 15_000
capital_cost_biomass = annuity(25, 0.04) * 2_500_000 + 50_000

marginal_cost_onshore = 2.7
marginal_cost_offshore = 3.3
marginal_cost_solar = 0.0
marginal_cost_ccgt = 25.0/0.58 + 80*0.2/0.58 + 4.0
marginal_cost_ocgt = 25.0/0.40 + 80*0.2/0.40 + 4.0
marginal_cost_biomass = 9.0/0.33 + 0 + 5.0

# Carrier definitions (for consistent colors)
carrier_colors = {
    "onshore wind": "dodgerblue",
    "offshore wind": "navy",
    "solar": "gold",
    "CCGT": "indianred",
    "OCGT": "salmon",
    "biomass": "forestgreen",
}


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  RUN OPTIMIZATION FOR EACH YEAR                                          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

years = [2016, 2017, 2018, 2019]
results = {}  # year → dict of results

for year in years:
    print(f"\n{'='*65}")
    print(f"  OPTIMIZING YEAR {year}")
    print(f"{'='*65}")

    # Load data
    data = pd.read_csv(f"DK_clean_{year}.csv")
    hours = pd.DatetimeIndex(pd.to_datetime(data["timestamp"].astype(str)))
    demand = data["demand_MW"].astype(float).to_numpy()
    cf_onshore = data["cf_onshore_wind"].astype(float).to_numpy()
    cf_offshore = data["cf_offshore_wind"].astype(float).to_numpy()
    cf_solar = data["cf_solar"].astype(float).to_numpy()

    # Build model (identical to Part a)
    n = pypsa.Network()
    n.set_snapshots(hours)

    n.add("Carrier", "onshore wind", nice_name="Onshore Wind", color="dodgerblue")
    n.add("Carrier", "offshore wind", nice_name="Offshore Wind", color="navy")
    n.add("Carrier", "solar", nice_name="Solar PV", color="gold")
    n.add("Carrier", "CCGT", nice_name="CCGT", color="indianred")
    n.add("Carrier", "OCGT", nice_name="OCGT", color="salmon")
    n.add("Carrier", "biomass", nice_name="Biomass", color="forestgreen")
    n.add("Carrier", "electricity")
    n.add("Bus", "DK", carrier="electricity")
    n.add("Load", "DK demand", bus="DK", p_set=demand, carrier="electricity")

    n.add("Generator", "DK onshore wind", bus="DK", carrier="onshore wind",
          p_nom_extendable=True, capital_cost=capital_cost_onshore,
          marginal_cost=marginal_cost_onshore, p_max_pu=cf_onshore)
    n.add("Generator", "DK offshore wind", bus="DK", carrier="offshore wind",
          p_nom_extendable=True, capital_cost=capital_cost_offshore,
          marginal_cost=marginal_cost_offshore, p_max_pu=cf_offshore)
    n.add("Generator", "DK solar", bus="DK", carrier="solar",
          p_nom_extendable=True, capital_cost=capital_cost_solar,
          marginal_cost=marginal_cost_solar, p_max_pu=cf_solar)
    n.add("Generator", "DK CCGT", bus="DK", carrier="CCGT",
          p_nom_extendable=True, capital_cost=capital_cost_ccgt,
          marginal_cost=marginal_cost_ccgt)
    n.add("Generator", "DK OCGT", bus="DK", carrier="OCGT",
          p_nom_extendable=True, capital_cost=capital_cost_ocgt,
          marginal_cost=marginal_cost_ocgt)
    n.add("Generator", "DK biomass", bus="DK", carrier="biomass",
          p_nom_extendable=True, capital_cost=capital_cost_biomass,
          marginal_cost=marginal_cost_biomass, p_nom_max=1500)

    # Solve
    n.optimize(solver_name="gurobi", solver_options={"OutputFlag": 0})

    # Collect results
    year_results = {"cost_per_mwh": n.objective / demand.sum()}
    for gen in n.generators.index:
        carrier = n.generators.loc[gen, "carrier"]
        cap = n.generators.loc[gen, "p_nom_opt"]
        gen_total = n.generators_t.p[gen].sum() / 1e6  # TWh
        year_results[f"cap_{carrier}"] = cap / 1000  # GW
        year_results[f"gen_{carrier}"] = gen_total

    results[year] = year_results

    # Print summary
    print(f"  System cost: {year_results['cost_per_mwh']:.2f} €/MWh")
    for carrier in carrier_colors:
        cap_key = f"cap_{carrier}"
        if cap_key in year_results:
            print(f"  {carrier:<20} {year_results[cap_key]:.2f} GW")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  COMPARISON TABLE                                                         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print(f"\n{'='*80}")
print(f"  COMPARISON ACROSS WEATHER YEARS")
print(f"{'='*80}")

carriers = ["onshore wind", "offshore wind", "solar", "CCGT", "OCGT", "biomass"]

# Capacities
print(f"\n{'CAPACITY [GW]':<20}", end="")
for y in years:
    print(f"  {y:>8}", end="")
print(f"  {'Mean':>8}  {'Std':>8}")
print("-" * 80)

for carrier in carriers:
    key = f"cap_{carrier}"
    vals = [results[y][key] for y in years]
    print(f"  {carrier:<18}", end="")
    for v in vals:
        print(f"  {v:>8.2f}", end="")
    print(f"  {np.mean(vals):>8.2f}  {np.std(vals):>8.2f}")

# System cost
print(f"\n  {'System cost [€/MWh]':<18}", end="")
costs = [results[y]["cost_per_mwh"] for y in years]
for c in costs:
    print(f"  {c:>8.2f}", end="")
print(f"  {np.mean(costs):>8.2f}  {np.std(costs):>8.2f}")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PLOTS                                                                    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# ── PLOT 1: Capacity by technology across years (grouped bar chart) ──────
fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(len(years))
width = 0.13
active_carriers = []

for carrier in carriers:
    vals = [results[y][f"cap_{carrier}"] for y in years]
    if max(vals) > 0.01:
        active_carriers.append(carrier)

for i, carrier in enumerate(active_carriers):
    vals = [results[y][f"cap_{carrier}"] for y in years]
    offset = (i - len(active_carriers)/2 + 0.5) * width
    ax.bar(x + offset, vals, width, label=carrier, color=carrier_colors[carrier],
           edgecolor="black", linewidth=0.5)

ax.set_xticks(x)
ax.set_xticklabels(years)
ax.set_ylabel("Optimal Capacity [GW]")
ax.set_title("Optimal Capacities Across Weather Years")
ax.legend(frameon=False, loc="upper right")
plt.tight_layout()
plt.savefig("plot_b_capacities_by_year.png", dpi=150)
plt.show()

# ── PLOT 2: Average capacity with error bars (mean ± range) ─────────────
fig, ax = plt.subplots(figsize=(10, 5))

means = []
mins = []
maxs = []
labels = []
colors = []

for carrier in carriers:
    vals = [results[y][f"cap_{carrier}"] for y in years]
    if np.mean(vals) > 0.01:
        means.append(np.mean(vals))
        mins.append(np.mean(vals) - np.min(vals))
        maxs.append(np.max(vals) - np.mean(vals))
        labels.append(carrier)
        colors.append(carrier_colors[carrier])

x = np.arange(len(labels))
ax.bar(x, means, color=colors, edgecolor="black", linewidth=0.5)
ax.errorbar(x, means, yerr=[mins, maxs], fmt="none", ecolor="black",
            capsize=5, capthick=1.5, linewidth=1.5)

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Capacity [GW]")
ax.set_title("Average Optimal Capacity ± Interannual Variability (2016–2019)")
for i, m in enumerate(means):
    ax.text(i, m + maxs[i] + 0.1, f"{m:.2f}", ha="center", fontsize=10)
plt.tight_layout()
plt.savefig("plot_b_average_variability.png", dpi=150)
plt.show()

# ── PLOT 3: System cost across years ────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
costs = [results[y]["cost_per_mwh"] for y in years]
ax.bar(years, costs, color="steelblue", edgecolor="black")
ax.axhline(np.mean(costs), color="red", ls="--", lw=1.5,
           label=f"Mean: {np.mean(costs):.1f} €/MWh")
ax.set_ylabel("System Cost [€/MWh]")
ax.set_title("System Cost Across Weather Years")
ax.legend()
for i, (y, c) in enumerate(zip(years, costs)):
    ax.text(y, c + 0.3, f"{c:.1f}", ha="center", fontsize=10)
plt.tight_layout()
plt.savefig("plot_b_system_cost.png", dpi=150)
plt.show()

# ── PLOT 4: Generation mix across years (stacked bar) ───────────────────
fig, ax = plt.subplots(figsize=(10, 5))
bottom = np.zeros(len(years))

for carrier in carriers:
    vals = [results[y].get(f"gen_{carrier}", 0) for y in years]
    if max(vals) > 0.01:
        ax.bar(years, vals, bottom=bottom, label=carrier,
               color=carrier_colors[carrier], edgecolor="black", linewidth=0.3)
        bottom += np.array(vals)

ax.set_ylabel("Annual Generation [TWh]")
ax.set_title("Generation Mix Across Weather Years")
ax.legend(frameon=False, loc="upper right")
plt.tight_layout()
plt.savefig("plot_b_generation_mix.png", dpi=150)
plt.show()

print("\n✅ All Part (b) plots saved!")
print("   plot_b_capacities_by_year.png")
print("   plot_b_average_variability.png")
print("   plot_b_system_cost.png")
print("   plot_b_generation_mix.png")
