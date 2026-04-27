"""
===============================================================================
IEG Course Project - Part (d): Transmission Network Expansion
===============================================================================
Multi-node model with Denmark connected to neighboring countries
using HVAC lines and DC power flow approximation.
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
# ╚═══════════════════════════════════════════════════════════════════════════╝

discount_rate = 0.04


def annuity(lifetime: float, rate: float) -> float:
    """Convert one-time investment to equal yearly payments."""
    return rate / (1 - (1 + rate) ** (-lifetime))


# ── Technology data ──────────────────────────────────────────────────────

onshore_wind = {
    "investment": 1_100_000,  # €/MW
    "fixed_om": 12_900,       # €/MW/year
    "variable_om": 2.7,       # €/MWh
    "lifetime": 30,           # years
}

offshore_wind = {
    "investment": 2_000_000,  # €/MW
    "fixed_om": 25_000,       # €/MW/year
    "variable_om": 3.3,       # €/MWh
    "lifetime": 30,           # years
}

solar_pv = {
    "investment": 420_000,    # €/MW
    "fixed_om": 7_500,        # €/MW/year
    "variable_om": 0.0,       # €/MWh
    "lifetime": 30,           # years
}

ccgt = {
    "investment": 800_000,    # €/MW
    "fixed_om": 25_000,       # €/MW/year
    "variable_om": 4.0,       # €/MWh
    "lifetime": 25,           # years
    "efficiency": 0.58,       # -
    "fuel_cost": 25.0,        # €/MWh_th
    "co2_emissions": 0.2,     # tCO2/MWh_th
}

ocgt = {
    "investment": 400_000,    # €/MW
    "fixed_om": 15_000,       # €/MW/year
    "variable_om": 4.0,       # €/MWh
    "lifetime": 25,           # years
    "efficiency": 0.40,       # -
    "fuel_cost": 25.0,        # €/MWh_th
    "co2_emissions": 0.2,     # tCO2/MWh_th
}

biomass = {
    "investment": 2_500_000,  # €/MW
    "fixed_om": 50_000,       # €/MW/year
    "variable_om": 5.0,       # €/MWh
    "lifetime": 25,           # years
    "efficiency": 0.33,       # -
    "fuel_cost": 9.0,         # €/MWh_th
    "co2_emissions": 0.0,     # tCO2/MWh_th
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
)

marginal_cost_biomass = (
    biomass["fuel_cost"] / biomass["efficiency"]
    + co2_price * biomass["co2_emissions"] / biomass["efficiency"]
    + biomass["variable_om"]
)

print("=" * 65)
print("COST SUMMARY")
print("=" * 65)
print(f"Discount rate: {discount_rate*100:.0f}%  |  CO2 price: {co2_price} €/tCO2")
print()
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
print(f"  Demand DK:   mean={demand.mean():.0f} MW, peak={demand.max():.0f} MW")
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
n.add("Carrier", "AC")

# Buses (countries)
n.add("Bus", "DK", carrier="AC")
n.add("Bus", "DE", carrier="AC")
n.add("Bus", "SE", carrier="AC")
n.add("Bus", "NO", carrier="AC")

# Transmission lines (HVAC, linearised AC / DC approximation)
n.add("Line", "DK-DE", bus0="DK", bus1="DE", x=0.1, r=0.01, s_nom=3000, carrier="AC")
n.add("Line", "DK-SE", bus0="DK", bus1="SE", x=0.1, r=0.01, s_nom=2000, carrier="AC")
n.add("Line", "DK-NO", bus0="DK", bus1="NO", x=0.1, r=0.01, s_nom=1500, carrier="AC")
n.add("Line", "SE-NO", bus0="SE", bus1="NO", x=0.1, r=0.01, s_nom=1000, carrier="AC")

# Demand
n.add("Load", "DK demand", bus="DK", p_set=demand)

# Demand in neighbouring countries (scaled from DK profile)
demand_DE = 4.0 * demand
demand_SE = 2.0 * demand
demand_NO = 1.5 * demand

n.add("Load", "DE demand", bus="DE", p_set=demand_DE)
n.add("Load", "SE demand", bus="SE", p_set=demand_SE)
n.add("Load", "NO demand", bus="NO", p_set=demand_NO)

# Total demand of entire system
total_demand = demand.sum() + demand_DE.sum() + demand_SE.sum() + demand_NO.sum()

# Denmark
n.add(
    "Generator", "DK onshore wind", bus="DK", carrier="onshore wind",
    p_nom_extendable=True, capital_cost=capital_cost_onshore,
    marginal_cost=marginal_cost_onshore, p_max_pu=cf_onshore
)
n.add(
    "Generator", "DK offshore wind", bus="DK", carrier="offshore wind",
    p_nom_extendable=True, capital_cost=capital_cost_offshore,
    marginal_cost=marginal_cost_offshore, p_max_pu=cf_offshore
)
n.add(
    "Generator", "DK solar", bus="DK", carrier="solar",
    p_nom_extendable=True, capital_cost=capital_cost_solar,
    marginal_cost=marginal_cost_solar, p_max_pu=cf_solar
)
n.add(
    "Generator", "DK CCGT", bus="DK", carrier="CCGT",
    p_nom_extendable=True, capital_cost=capital_cost_ccgt,
    marginal_cost=marginal_cost_ccgt
)
n.add(
    "Generator", "DK OCGT", bus="DK", carrier="OCGT",
    p_nom_extendable=True, capital_cost=capital_cost_ocgt,
    marginal_cost=marginal_cost_ocgt
)
n.add(
    "Generator", "DK biomass", bus="DK", carrier="biomass",
    p_nom_extendable=True, capital_cost=capital_cost_biomass,
    marginal_cost=marginal_cost_biomass, p_nom_max=1500
)

# Germany (simplified mix)
n.add(
    "Generator", "DE onshore wind", bus="DE", carrier="onshore wind",
    p_nom_extendable=True, capital_cost=capital_cost_onshore,
    marginal_cost=marginal_cost_onshore, p_max_pu=cf_onshore
)
n.add(
    "Generator", "DE solar", bus="DE", carrier="solar",
    p_nom_extendable=True, capital_cost=capital_cost_solar,
    marginal_cost=marginal_cost_solar, p_max_pu=cf_solar
)
n.add(
    "Generator", "DE CCGT", bus="DE", carrier="CCGT",
    p_nom_extendable=True, capital_cost=capital_cost_ccgt,
    marginal_cost=marginal_cost_ccgt
)

# Sweden (simplified mix)
n.add(
    "Generator", "SE onshore wind", bus="SE", carrier="onshore wind",
    p_nom_extendable=True, capital_cost=capital_cost_onshore,
    marginal_cost=marginal_cost_onshore, p_max_pu=cf_onshore
)
n.add(
    "Generator", "SE biomass", bus="SE", carrier="biomass",
    p_nom_extendable=True, capital_cost=capital_cost_biomass,
    marginal_cost=marginal_cost_biomass
)

# Norway (hydro-like proxy with biomass)
n.add(
    "Generator", "NO biomass", bus="NO", carrier="biomass",
    p_nom_extendable=True, capital_cost=capital_cost_biomass,
    marginal_cost=marginal_cost_biomass
)

# Assumptions:
# Neighbouring countries are represented with simplified technology mixes.
# Germany is approximated by wind + solar + gas.
# Sweden is approximated by wind + dispatchable renewable generation.
# Norway is approximated by highly flexible dispatchable renewable generation.


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 4: SOLVE                                                        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print("\nSolving optimization problem...")
n.optimize(solver_name="gurobi")
print("Done!\n")

print("\nLINE FLOWS (average absolute)")
print("-" * 50)
for line in n.lines.index:
    flow = n.lines_t.p0[line]
    print(f"{line:<10} mean flow = {abs(flow).mean():.2f} MW")

print("\nAVERAGE ELECTRICITY PRICES")
print("-" * 50)
for bus in n.buses.index:
    price = n.buses_t.marginal_price[bus]
    print(f"{bus:<5} mean price = {price.mean():.2f} €/MWh")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 5: RESULTS                                                      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print("=" * 75)
print("OPTIMAL CAPACITIES")
print("=" * 75)
for gen in n.generators.index:
    cap = n.generators.loc[gen, "p_nom_opt"]
    print(f"  {gen:<20} {cap:>10,.0f} MW  ({cap/1000:.2f} GW)")

print(f"\n  Total system cost: {n.objective:>15,.0f} €/year")
print(f"  Cost per MWh (whole system): {n.objective / total_demand:>10.2f} €/MWh")
print("=" * 75)

print("\nACHIEVED CAPACITY FACTORS & GENERATION")
print("-" * 75)
for gen in n.generators.index:
    p_nom = n.generators.loc[gen, "p_nom_opt"]
    if p_nom > 0:
        total_gen = n.generators_t.p[gen].sum()
        cf = total_gen / (p_nom * 8760)
        total_twh = total_gen / 1e6
        print(f"  {gen:<20} CF={cf:>6.1%}   Generation={total_twh:.2f} TWh")

# Shadow prices for DK
prices_DK = n.buses_t.marginal_price["DK"]

print("\nSHADOW PRICES FOR DK")
print("-" * 50)
print(f"  Mean:   {prices_DK.mean():.2f} €/MWh")
print(f"  Median: {prices_DK.median():.2f} €/MWh")
print(f"  Min:    {prices_DK.min():.2f} €/MWh")
print(f"  Max:    {prices_DK.max():.2f} €/MWh")
print(f"  Hours at 0 €/MWh:  {(prices_DK < 0.01).sum()}")
print(f"  Hours > 70 €/MWh:  {(prices_DK > 70).sum()}")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 6: PLOTS                                                        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

carrier_colors = n.carriers.color.to_dict()

# Dispatch aggregated by generator name → then by country for selected plots
dispatch_by_gen = n.generators_t.p.copy() / 1000  # GW
dispatch_by_carrier = n.generators_t.p.T.groupby(n.generators.carrier).sum().T / 1000  # GW

# Total system demand for plotting
system_demand_gw = pd.Series(
    (demand + demand_DE + demand_SE + demand_NO) / 1000,
    index=hours
)

# Plot 1: Winter week, total generation by carrier vs total demand
fig, ax = plt.subplots(figsize=(12, 5))
w = dispatch_by_carrier.loc["2019-01-14":"2019-01-20"]
w.plot.area(ax=ax, linewidth=0, color=[carrier_colors[c] for c in w.columns])
system_demand_gw.loc["2019-01-14":"2019-01-20"].plot(
    ax=ax, color="black", lw=2, ls="--", label="Total demand"
)
ax.set_ylabel("Power [GW]")
ax.set_title("System Dispatch — Winter Week (Jan 14–20, 2019)")
ax.legend(loc="upper right", frameon=False)
ax.set_ylim(0)
plt.tight_layout()
plt.savefig("plot_d_winter_dispatch.png", dpi=150)
plt.show()

# Plot 2: Summer week
fig, ax = plt.subplots(figsize=(12, 5))
s = dispatch_by_carrier.loc["2019-07-15":"2019-07-21"]
s.plot.area(ax=ax, linewidth=0, color=[carrier_colors[c] for c in s.columns])
system_demand_gw.loc["2019-07-15":"2019-07-21"].plot(
    ax=ax, color="black", lw=2, ls="--", label="Total demand"
)
ax.set_ylabel("Power [GW]")
ax.set_title("System Dispatch — Summer Week (Jul 15–21, 2019)")
ax.legend(loc="upper right", frameon=False)
ax.set_ylim(0)
plt.tight_layout()
plt.savefig("plot_d_summer_dispatch.png", dpi=150)
plt.show()

# Plot 3: Annual mix
annual_gen = dispatch_by_carrier.sum(axis=0) * 1000 / 1e6  # TWh
annual_gen = annual_gen[annual_gen > 0.001]
fig, ax = plt.subplots(figsize=(7, 7))
annual_gen.plot.pie(
    ax=ax,
    autopct="%1.1f%%",
    colors=[carrier_colors[c] for c in annual_gen.index],
    startangle=90,
    textprops={"fontsize": 12},
)
ax.set_ylabel("")
ax.set_title(f"Annual Electricity Mix ({annual_gen.sum():.1f} TWh)")
plt.tight_layout()
plt.savefig("plot_d_annual_mix.png", dpi=150)
plt.show()

# Plot 4: Line flows duration curves
fig, ax = plt.subplots(figsize=(10, 5))
for line in n.lines.index:
    sorted_vals = n.lines_t.p0[line].abs().sort_values(ascending=False).values / 1000  # GW
    ax.plot(np.arange(len(sorted_vals)), sorted_vals, lw=1.5, label=line)
ax.set_xlabel("Hours (sorted)")
ax.set_ylabel("Absolute flow [GW]")
ax.set_title("Line Flow Duration Curves")
ax.legend(frameon=False)
ax.set_xlim(0, 8760)
plt.tight_layout()
plt.savefig("plot_d_line_duration_curves.png", dpi=150)
plt.show()

# Plot 5: Optimal capacities by generator
fig, ax = plt.subplots(figsize=(12, 5))
cap_s = (n.generators.p_nom_opt / 1000).copy()
cap_s = cap_s[cap_s > 0]
cap_s.plot.bar(
    ax=ax,
    color=[carrier_colors[n.generators.loc[g, "carrier"]] for g in cap_s.index],
    edgecolor="black"
)
ax.set_ylabel("Capacity [GW]")
ax.set_title("Optimal Installed Capacities by Generator")
for i, (name, val) in enumerate(cap_s.items()):
    ax.text(i, val + 0.1, f"{val:.2f}", ha="center", fontsize=9)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("plot_d_optimal_capacities.png", dpi=150)
plt.show()

# Plot 6: Average electricity prices by country
fig, ax = plt.subplots(figsize=(7, 4))
avg_prices = n.buses_t.marginal_price.mean()
avg_prices.plot.bar(ax=ax, color="steelblue", edgecolor="black")
ax.set_ylabel("Average price [€/MWh]")
ax.set_title("Average Electricity Prices by Country")
for i, (name, val) in enumerate(avg_prices.items()):
    ax.text(i, val + 0.5, f"{val:.1f}", ha="center", fontsize=10)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("plot_d_average_prices.png", dpi=150)
plt.show()

print("\n✅ All plots saved!")

snapshot0 = n.snapshots[0]

gen0 = n.generators_t.p.loc[snapshot0].groupby(n.generators.bus).sum()
load0 = n.loads_t.p_set.loc[snapshot0].groupby(n.loads.bus).sum()

buses = ["DK", "DE", "SE", "NO"]
gen0 = gen0.reindex(buses, fill_value=0)
load0 = load0.reindex(buses, fill_value=0)

p0 = gen0 - load0

print("\nFirst-hour nodal imbalances [MW]:")
print(p0)

print("\nFirst-hour PyPSA line flows p0 [MW]:")
print(n.lines_t.p0.loc[snapshot0])