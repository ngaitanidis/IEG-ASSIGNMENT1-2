"""
===============================================================================
IEG Course Project - Part (a): Optimal Capacity Expansion for Denmark
===============================================================================
Course: 46770 Integrated Energy Grids, DTU
===============================================================================

TECHNOLOGIES COMPETING:
    1. Onshore wind
    2. Offshore wind
    3. Solar PV
    4. CCGT (Combined Cycle Gas Turbine)
    5. OCGT (Open Cycle Gas Turbine — peaker)
    6. Biomass (wood pellets, steam turbine)

DATA SOURCES:
    - Demand: OPSD/ENTSO-E actual load for Denmark, 2019
    - Renewable CFs: OPSD (wind), Renewables.ninja (solar)
    - Technology costs: Danish Energy Agency (DEA) Technology Catalogue, 2024

HOW TO RUN:
    1. Make sure DK_clean_2019.csv is in the same folder
    2. Run: python project_part_a.py
===============================================================================
"""

# ── CRITICAL: Disable Arrow strings before importing pypsa ───────────────
import pandas as pd
pd.options.future.infer_string = False

import pypsa
import numpy as np
import matplotlib.pyplot as plt


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 1: COST ASSUMPTIONS                                            ║
# ║  Source: Danish Energy Agency (DEA) Technology Catalogue, 2024           ║
# ║  URL: https://ens.dk/en/our-services/projections-and-models/            ║
# ║       technology-data                                                    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

discount_rate = 0.04

def annuity(lifetime, rate):
    """Convert one-time investment to equal yearly payments."""
    return rate / (1 - (1 + rate) ** (-lifetime))


# ── Technology data ──────────────────────────────────────────────────────

# Onshore Wind — DEA, "Onshore wind turbines" (2024)
onshore_wind = {
    "investment": 1_100_000,  # €/MW
    "fixed_om": 12_900,       # €/MW/year
    "variable_om": 2.7,       # €/MWh
    "lifetime": 30,           # years
}

# Offshore Wind — DEA, "Offshore wind turbines" (2024, incl. grid connection)
offshore_wind = {
    "investment": 2_000_000,  # €/MW
    "fixed_om": 25_000,       # €/MW/year
    "variable_om": 3.3,       # €/MWh
    "lifetime": 30,           # years
}

# Solar PV — DEA, "Large-scale solar PV" (2024)
solar_pv = {
    "investment": 420_000,    # €/MW
    "fixed_om": 7_500,        # €/MW/year
    "variable_om": 0,         # €/MWh
    "lifetime": 30,           # years
}

# CCGT — DEA, "Combined cycle gas turbines" (2024)
# Combined cycle = gas turbine + steam turbine → 58% efficiency
ccgt = {
    "investment": 800_000,    # €/MW
    "fixed_om": 25_000,       # €/MW/year
    "variable_om": 4.0,       # €/MWh
    "lifetime": 25,           # years
    "efficiency": 0.58,       # 58%
    "fuel_cost": 25.0,        # €/MWh_thermal
    "co2_emissions": 0.2,     # tCO2/MWh_thermal
}

# OCGT — DEA, "Simple cycle gas turbines" (2024)
# Cheap to build, inefficient to run — the "peaker" for extreme hours
# Only ~40% efficiency vs CCGT's 58%
ocgt = {
    "investment": 400_000,    # €/MW
    "fixed_om": 15_000,       # €/MW/year
    "variable_om": 4.0,       # €/MWh
    "lifetime": 25,           # years
    "efficiency": 0.40,       # 40%
    "fuel_cost": 25.0,        # €/MWh_thermal (same gas)
    "co2_emissions": 0.2,     # tCO2/MWh_thermal (same gas)
}

# Biomass — DEA, "Wood pellets/chips, steam turbine" (2024)
# Dispatchable renewable — runs when needed, like gas
# CO2 counted as zero in EU ETS (biogenic carbon cycle)
biomass = {
    "investment": 2_500_000,  # €/MW
    "fixed_om": 50_000,       # €/MW/year
    "variable_om": 5.0,       # €/MWh
    "lifetime": 25,           # years
    "efficiency": 0.33,       # 33% (electricity only)
    "fuel_cost": 9.0,         # €/MWh_thermal (wood pellets)
    "co2_emissions": 0.0,     # tCO2/MWh (zero under EU ETS)
}

co2_price = 80  # €/tCO2


# ── Compute PyPSA input costs ────────────────────────────────────────────

capital_cost_onshore = (
    annuity(onshore_wind["lifetime"], discount_rate) * onshore_wind["investment"]
    + onshore_wind["fixed_om"]
)
capital_cost_offshore = (
    annuity(offshore_wind["lifetime"], discount_rate) * offshore_wind["investment"]
    + offshore_wind["fixed_om"]
)
capital_cost_solar = (
    annuity(solar_pv["lifetime"], discount_rate) * solar_pv["investment"]
    + solar_pv["fixed_om"]
)
capital_cost_ccgt = (
    annuity(ccgt["lifetime"], discount_rate) * ccgt["investment"]
    + ccgt["fixed_om"]
)
capital_cost_ocgt = (
    annuity(ocgt["lifetime"], discount_rate) * ocgt["investment"]
    + ocgt["fixed_om"]
)
capital_cost_biomass = (
    annuity(biomass["lifetime"], discount_rate) * biomass["investment"]
    + biomass["fixed_om"]
)

marginal_cost_onshore = onshore_wind["variable_om"]
marginal_cost_offshore = offshore_wind["variable_om"]
marginal_cost_solar = solar_pv["variable_om"]
marginal_cost_ccgt = (
    ccgt["fuel_cost"] / ccgt["efficiency"]
    + co2_price * ccgt["co2_emissions"] / ccgt["efficiency"]
    + ccgt["variable_om"]
)
marginal_cost_ocgt = (
    ocgt["fuel_cost"] / ocgt["efficiency"]
    + co2_price * ocgt["co2_emissions"] / ocgt["efficiency"]
    + ocgt["variable_om"]
)  # ~106.5 €/MWh — very expensive to run!
marginal_cost_biomass = (
    biomass["fuel_cost"] / biomass["efficiency"]
    + co2_price * biomass["co2_emissions"] / biomass["efficiency"]
    + biomass["variable_om"]
)  # ~32.3 €/MWh — cheaper than gas!

# Print summary
print("=" * 65)
print("COST SUMMARY (cite DEA Technology Catalogue 2024 in your report)")
print("=" * 65)
print(f"Discount rate: {discount_rate*100:.0f}%  |  CO2 price: {co2_price} €/tCO2")
print(f"")
print(f"{'Technology':<20} {'Capital [€/MW/yr]':>18} {'Marginal [€/MWh]':>18}")
print(f"{'-'*20} {'-'*18} {'-'*18}")
print(f"{'Onshore wind':<20} {capital_cost_onshore:>18,.0f} {marginal_cost_onshore:>18.1f}")
print(f"{'Offshore wind':<20} {capital_cost_offshore:>18,.0f} {marginal_cost_offshore:>18.1f}")
print(f"{'Solar PV':<20} {capital_cost_solar:>18,.0f} {marginal_cost_solar:>18.1f}")
print(f"{'CCGT':<20} {capital_cost_ccgt:>18,.0f} {marginal_cost_ccgt:>18.1f}")
print(f"{'OCGT':<20} {capital_cost_ocgt:>18,.0f} {marginal_cost_ocgt:>18.1f}")
print(f"{'Biomass':<20} {capital_cost_biomass:>18,.0f} {marginal_cost_biomass:>18.1f}")
print("=" * 65)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 2: LOAD REAL DATA                                               ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

data = pd.read_csv("DK_clean_2019.csv")
hours = pd.DatetimeIndex(pd.to_datetime(data["timestamp"].astype(str)))

demand = data["demand_MW"].astype(float).to_numpy()
cf_onshore = data["cf_onshore_wind"].astype(float).to_numpy()
cf_offshore = data["cf_offshore_wind"].astype(float).to_numpy()
cf_solar = data["cf_solar"].astype(float).to_numpy()

print(f"\nData loaded: {len(hours)} hours ({hours[0].year})")
print(f"  Demand:      mean={demand.mean():.0f} MW, peak={demand.max():.0f} MW")
print(f"  CF onshore:  mean={cf_onshore.mean():.3f}")
print(f"  CF offshore: mean={cf_offshore.mean():.3f}")
print(f"  CF solar:    mean={cf_solar.mean():.3f}")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 3: BUILD THE PyPSA MODEL                                       ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

n = pypsa.Network()
n.set_snapshots(hours)

# Carriers
n.add("Carrier", "onshore wind", nice_name="Onshore Wind", color="dodgerblue")
n.add("Carrier", "offshore wind", nice_name="Offshore Wind", color="navy")
n.add("Carrier", "solar", nice_name="Solar PV", color="gold")
n.add("Carrier", "CCGT", nice_name="CCGT (Gas)", color="indianred")
n.add("Carrier", "OCGT", nice_name="OCGT (Gas Peaker)", color="salmon")
n.add("Carrier", "biomass", nice_name="Biomass", color="forestgreen")
n.add("Carrier", "electricity")

# Bus
n.add("Bus", "DK", carrier="electricity")

# Demand
n.add("Load", "DK demand", bus="DK", p_set=demand, carrier="electricity")

# Generators — all extendable
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
      marginal_cost=marginal_cost_biomass,
      p_nom_max=1500)  # ← MW cap based on realistic DK potential


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 4: SOLVE                                                        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print("\nSolving optimization problem...")
n.optimize(solver_name="gurobi")
print("Done!\n")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 5: RESULTS                                                      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print("=" * 65)
print("OPTIMAL CAPACITIES")
print("=" * 65)
for gen in n.generators.index:
    cap = n.generators.loc[gen, "p_nom_opt"]
    carrier = n.generators.loc[gen, "carrier"]
    print(f"  {carrier:<20} {cap:>10,.0f} MW  ({cap/1000:.2f} GW)")

print(f"\n  Total system cost: {n.objective:>15,.0f} €/year")
print(f"  Cost per MWh:      {n.objective / demand.sum():>15.2f} €/MWh")
print("=" * 65)

print("\nACHIEVED CAPACITY FACTORS & GENERATION")
print("-" * 55)
for gen in n.generators.index:
    p_nom = n.generators.loc[gen, "p_nom_opt"]
    carrier = n.generators.loc[gen, "carrier"]
    if p_nom > 0:
        total_gen = n.generators_t.p[gen].sum()
        cf = total_gen / (p_nom * 8760)
        total_twh = total_gen / 1e6
        print(f"  {carrier:<20} CF={cf:>6.1%}   Generation={total_twh:.2f} TWh")

# ── Shadow prices ────────────────────────────────────────────────────────
prices = n.buses_t.marginal_price["DK"]

print("\nSHADOW PRICES (day-ahead equivalent)")
print("-" * 50)
print(f"  Mean:   {prices.mean():.2f} €/MWh")
print(f"  Median: {prices.median():.2f} €/MWh")
print(f"  Min:    {prices.min():.2f} €/MWh")
print(f"  Max:    {prices.max():.2f} €/MWh")
print(f"  Hours at 0 €/MWh:    {(prices < 0.01).sum()}")
print(f"  Hours at gas price:   {(prices > 70).sum()}")
print(f"\n  System cost/MWh:     {n.objective / demand.sum():.2f} €/MWh")
print(f"  Avg shadow price:    {prices.mean():.2f} €/MWh")
print(f"  Gap = capital costs: {n.objective / demand.sum() - prices.mean():.2f} €/MWh")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 6: PLOTS                                                        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

dispatch = n.generators_t.p.T.groupby(n.generators.carrier).sum().T / 1000  # GW
carrier_colors = n.carriers.color.to_dict()
demand_gw = pd.Series(demand / 1000, index=hours)

# PLOT 1: Winter week
fig, ax = plt.subplots(figsize=(12, 5))
w = dispatch.loc["2019-01-14":"2019-01-20"]
w.plot.area(ax=ax, linewidth=0, color=[carrier_colors[c] for c in w.columns])
demand_gw.loc["2019-01-14":"2019-01-20"].plot(ax=ax, color="black", lw=2, ls="--", label="Demand")
ax.set_ylabel("Power [GW]"); ax.set_title("Dispatch — Winter Week (Jan 14–20, 2019)")
ax.legend(loc="upper right", frameon=False); ax.set_ylim(0)
plt.tight_layout(); plt.savefig("plot_winter_dispatch.png", dpi=150); plt.show()

# PLOT 2: Summer week
fig, ax = plt.subplots(figsize=(12, 5))
s = dispatch.loc["2019-07-15":"2019-07-21"]
s.plot.area(ax=ax, linewidth=0, color=[carrier_colors[c] for c in s.columns])
demand_gw.loc["2019-07-15":"2019-07-21"].plot(ax=ax, color="black", lw=2, ls="--", label="Demand")
ax.set_ylabel("Power [GW]"); ax.set_title("Dispatch — Summer Week (Jul 15–21, 2019)")
ax.legend(loc="upper right", frameon=False); ax.set_ylim(0)
plt.tight_layout(); plt.savefig("plot_summer_dispatch.png", dpi=150); plt.show()

# PLOT 3: Annual mix (pie)
annual_gen = n.generators_t.p.T.groupby(n.generators.carrier).sum().sum(axis=1) / 1e6
annual_gen = annual_gen[annual_gen > 0.001]
fig, ax = plt.subplots(figsize=(7, 7))
annual_gen.plot.pie(ax=ax, autopct="%1.1f%%",
    colors=[carrier_colors[c] for c in annual_gen.index],
    startangle=90, textprops={"fontsize": 12})
ax.set_ylabel(""); ax.set_title(f"Annual Electricity Mix ({annual_gen.sum():.1f} TWh)")
plt.tight_layout(); plt.savefig("plot_annual_mix.png", dpi=150); plt.show()

# PLOT 4: Duration curves
fig, ax = plt.subplots(figsize=(10, 5))
for carrier in dispatch.columns:
    if dispatch[carrier].sum() > 0:
        sorted_vals = dispatch[carrier].sort_values(ascending=False).values
        ax.plot(np.arange(len(sorted_vals)), sorted_vals,
                label=carrier, color=carrier_colors[carrier], lw=1.5)
ax.set_xlabel("Hours (sorted)"); ax.set_ylabel("Power [GW]")
ax.set_title("Duration Curves"); ax.legend(frameon=False); ax.set_xlim(0, 8760)
plt.tight_layout(); plt.savefig("plot_duration_curves.png", dpi=150); plt.show()

# PLOT 5: Capacity factors (bar)
fig, ax = plt.subplots(figsize=(10, 4))
cf_data = {}
for gen in n.generators.index:
    p_nom = n.generators.loc[gen, "p_nom_opt"]
    if p_nom > 0:
        carrier = n.generators.loc[gen, "carrier"]
        cf_data[carrier] = n.generators_t.p[gen].sum() / (p_nom * 8760)
cf_s = pd.Series(cf_data)
cf_s.plot.bar(ax=ax, color=[carrier_colors[c] for c in cf_s.index], edgecolor="black")
ax.set_ylabel("Capacity Factor"); ax.set_title("Achieved Capacity Factors"); ax.set_ylim(0, 1)
for i, (t, v) in enumerate(cf_s.items()):
    ax.text(i, v + 0.02, f"{v:.1%}", ha="center", fontsize=10)
plt.xticks(rotation=0); plt.tight_layout()
plt.savefig("plot_capacity_factors.png", dpi=150); plt.show()

# PLOT 6: Optimal capacities (bar)
fig, ax = plt.subplots(figsize=(10, 4))
cap_d = {}
for gen in n.generators.index:
    carrier = n.generators.loc[gen, "carrier"]
    cap_d[carrier] = n.generators.loc[gen, "p_nom_opt"] / 1000
cap_s = pd.Series(cap_d); cap_s = cap_s[cap_s > 0]
cap_s.plot.bar(ax=ax, color=[carrier_colors[c] for c in cap_s.index], edgecolor="black")
ax.set_ylabel("Capacity [GW]"); ax.set_title("Optimal Installed Capacities")
for i, (t, v) in enumerate(cap_s.items()):
    ax.text(i, v + 0.1, f"{v:.2f} GW", ha="center", fontsize=10)
plt.xticks(rotation=0); plt.tight_layout()
plt.savefig("plot_optimal_capacities.png", dpi=150); plt.show()

# PLOT 7: Shadow price histogram
fig, ax = plt.subplots(figsize=(10, 4))
prices_clipped = prices.clip(upper=150)  # clip for readability
prices_clipped.plot.hist(ax=ax, bins=50, edgecolor="black", alpha=0.7)
ax.axvline(prices.mean(), color="red", lw=2, ls="--", label=f"Mean: {prices.mean():.1f} €/MWh")
ax.set_xlabel("Shadow Price [€/MWh]"); ax.set_ylabel("Hours")
ax.set_title("Distribution of Hourly Shadow Prices (Day-Ahead Equivalent)")
ax.set_xlim(0, 150)
ax.legend(); plt.tight_layout()
plt.savefig("plot_shadow_prices.png", dpi=150); plt.show()

print("\n✅ All plots saved!")
