"""
===============================================================================
IEG Course Project - Part (c): Storage Technologies
===============================================================================
Same as Part (a) + battery and hydrogen storage.
Investigates intraday vs seasonal balancing strategies.

HOW TO RUN:
    1. Make sure DK_clean_2019.csv is in the same folder
    2. Run: python project_part_c.py
===============================================================================
"""

import pandas as pd
pd.options.future.infer_string = False

import pypsa
import numpy as np
import matplotlib.pyplot as plt


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  COSTS — same as Part (a)                                                ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

discount_rate = 0.04

def annuity(lifetime, rate):
    return rate / (1 - (1 + rate) ** (-lifetime))

# Generators (identical to Part a)
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
marginal_cost_biomass = 9.0/0.33 + 5.0

# ── NEW: Storage costs (DEA Technology Catalogue, 2024) ──────────────────

# Battery (Lithium-ion, utility scale)
# Investment: power component ~150,000 EUR/MW + energy component ~150,000 EUR/MWh
# Round-trip efficiency: ~92% (96% each way)
# Lifetime: 15 years (calendar), ~5000 cycles
# Good for: intraday shifting (4-6 hour duration)
battery = {
    "power_cost": 150_000,    # EUR/MW (inverter + power electronics)
    "energy_cost": 150_000,   # EUR/MWh (battery cells)
    "fixed_om": 5_000,        # EUR/MW/year
    "efficiency": 0.92,       # round-trip (charge × discharge)
    "lifetime": 15,           # years
    "max_hours": 6,           # maximum energy-to-power ratio [hours]
}

# Hydrogen storage (electrolyser + fuel cell + underground/tank storage)
# This is modelled as a StorageUnit with low round-trip efficiency
# but very cheap energy storage (underground caverns or steel tanks)
# Good for: seasonal shifting (weeks to months)
hydrogen = {
    "power_cost": 800_000,    # EUR/MW (electrolyser + fuel cell)
    "energy_cost": 2_000,     # EUR/MWh (underground storage is very cheap per MWh)
    "fixed_om": 20_000,       # EUR/MW/year
    "efficiency": 0.35,       # round-trip (electrolyser ~65% × fuel cell ~55%)
    "lifetime": 25,           # years
    "max_hours": 168,         # max 1 week of storage at full power
}

# Calculate annualised costs for PyPSA
# StorageUnit in PyPSA takes:
#   capital_cost = cost per MW of power capacity [EUR/MW/year]
#   But energy capacity is linked via max_hours parameter

capital_cost_battery_power = (
    annuity(battery["lifetime"], discount_rate) * battery["power_cost"]
    + battery["fixed_om"]
)
capital_cost_battery_energy = (
    annuity(battery["lifetime"], discount_rate) * battery["energy_cost"]
)

capital_cost_h2_power = (
    annuity(hydrogen["lifetime"], discount_rate) * hydrogen["power_cost"]
    + hydrogen["fixed_om"]
)
capital_cost_h2_energy = (
    annuity(hydrogen["lifetime"], discount_rate) * hydrogen["energy_cost"]
)

print("=" * 65)
print("STORAGE COST SUMMARY")
print("=" * 65)
print(f"{'Battery':<20} Power: {capital_cost_battery_power:>10,.0f} EUR/MW/yr")
print(f"{'':20} Energy: {capital_cost_battery_energy:>10,.0f} EUR/MWh/yr")
print(f"{'':20} Efficiency: {battery['efficiency']*100:.0f}%")
print(f"{'Hydrogen':<20} Power: {capital_cost_h2_power:>10,.0f} EUR/MW/yr")
print(f"{'':20} Energy: {capital_cost_h2_energy:>10,.0f} EUR/MWh/yr")
print(f"{'':20} Efficiency: {hydrogen['efficiency']*100:.0f}%")
print("=" * 65)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  LOAD DATA                                                                ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

data = pd.read_csv("DK_clean_2019.csv")
hours = pd.DatetimeIndex(pd.to_datetime(data["timestamp"].astype(str)))
demand = data["demand_MW"].astype(float).to_numpy()
cf_onshore = data["cf_onshore_wind"].astype(float).to_numpy()
cf_offshore = data["cf_offshore_wind"].astype(float).to_numpy()
cf_solar = data["cf_solar"].astype(float).to_numpy()


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  BUILD MODEL                                                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

n = pypsa.Network()
n.set_snapshots(hours)

# Carriers
n.add("Carrier", "onshore wind", nice_name="Onshore Wind", color="dodgerblue")
n.add("Carrier", "offshore wind", nice_name="Offshore Wind", color="navy")
n.add("Carrier", "solar", nice_name="Solar PV", color="gold")
n.add("Carrier", "CCGT", nice_name="CCGT", color="indianred")
n.add("Carrier", "OCGT", nice_name="OCGT", color="salmon")
n.add("Carrier", "biomass", nice_name="Biomass", color="forestgreen")
n.add("Carrier", "battery", nice_name="Battery", color="mediumpurple")
n.add("Carrier", "hydrogen", nice_name="Hydrogen Storage", color="cyan")
n.add("Carrier", "electricity")

# Bus
n.add("Bus", "DK", carrier="electricity")

# Demand
n.add("Load", "DK demand", bus="DK", p_set=demand, carrier="electricity")

# ── Generators (same as Part a) ─────────────────────────────────────────
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

# ── NEW: Storage units ──────────────────────────────────────────────────

# Battery — using Store + Link for separate power/energy optimisation
# But StorageUnit is simpler and works fine here
n.add(
    "StorageUnit", "DK battery",
    bus="DK",
    carrier="battery",
    p_nom_extendable=True,
    capital_cost=capital_cost_battery_power,     # EUR/MW/year for power
    max_hours=battery["max_hours"],              # max energy/power ratio
    efficiency_store=battery["efficiency"]**0.5, # charge efficiency (sqrt of round-trip)
    efficiency_dispatch=battery["efficiency"]**0.5,  # discharge efficiency
    cyclic_state_of_charge=True,                 # SOC at end = SOC at start
    marginal_cost=0,                             # no fuel cost
)

# Hydrogen storage
n.add(
    "StorageUnit", "DK hydrogen",
    bus="DK",
    carrier="hydrogen",
    p_nom_extendable=True,
    capital_cost=capital_cost_h2_power,          # EUR/MW/year for power
    max_hours=hydrogen["max_hours"],             # up to 168 hours (1 week)
    efficiency_store=hydrogen["efficiency"]**0.5,
    efficiency_dispatch=hydrogen["efficiency"]**0.5,
    cyclic_state_of_charge=True,
    marginal_cost=0,
)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  SOLVE                                                                    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print("\nSolving optimization problem (with storage)...")
n.optimize(solver_name="gurobi", solver_options={"OutputFlag": 0})
print("Done!\n")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  RESULTS                                                                  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print("=" * 65)
print("OPTIMAL CAPACITIES (with storage)")
print("=" * 65)

# Generators
for gen in n.generators.index:
    cap = n.generators.loc[gen, "p_nom_opt"]
    carrier = n.generators.loc[gen, "carrier"]
    print(f"  {carrier:<20} {cap:>10,.0f} MW  ({cap/1000:.2f} GW)")

# Storage
print(f"\n  STORAGE:")
for su in n.storage_units.index:
    cap_mw = n.storage_units.loc[su, "p_nom_opt"]
    max_h = n.storage_units.loc[su, "max_hours"]
    cap_mwh = cap_mw * max_h
    carrier = n.storage_units.loc[su, "carrier"]
    print(f"  {carrier:<20} {cap_mw:>10,.0f} MW / {cap_mwh:>10,.0f} MWh ({max_h:.0f}h)")

print(f"\n  Total system cost: {n.objective:>15,.0f} EUR/year")
print(f"  Cost per MWh:      {n.objective / demand.sum():>15.2f} EUR/MWh")
print("=" * 65)

# Generator performance
print("\nGENERATOR PERFORMANCE")
print("-" * 55)
for gen in n.generators.index:
    p_nom = n.generators.loc[gen, "p_nom_opt"]
    carrier = n.generators.loc[gen, "carrier"]
    if p_nom > 0:
        total_gen = n.generators_t.p[gen].sum()
        cf = total_gen / (p_nom * 8760)
        total_twh = total_gen / 1e6
        print(f"  {carrier:<20} CF={cf:>6.1%}   Generation={total_twh:.2f} TWh")

# Storage performance
print("\nSTORAGE PERFORMANCE")
print("-" * 55)
for su in n.storage_units.index:
    carrier = n.storage_units.loc[su, "carrier"]
    p_nom = n.storage_units.loc[su, "p_nom_opt"]
    if p_nom > 0:
        charged = n.storage_units_t.p[su].clip(upper=0).sum() * -1 / 1e6  # TWh charged
        discharged = n.storage_units_t.p[su].clip(lower=0).sum() / 1e6     # TWh discharged
        cycles = discharged * 1e6 / (p_nom * n.storage_units.loc[su, "max_hours"]) if p_nom > 0 else 0
        print(f"  {carrier:<20} Charged={charged:.2f} TWh  Discharged={discharged:.2f} TWh  Cycles={cycles:.0f}/yr")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  COMPARISON WITH PART (a) — print for easy reference                     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print("\n" + "=" * 65)
print("COMPARE: Part (a) without storage vs Part (c) with storage")
print("=" * 65)
print("  Part (a): 58.56 EUR/MWh  |  Part (c): {:.2f} EUR/MWh".format(
    n.objective / demand.sum()))
print("  Reduction: {:.2f} EUR/MWh ({:.1f}%)".format(
    58.56 - n.objective / demand.sum(),
    (58.56 - n.objective / demand.sum()) / 58.56 * 100))


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PLOTS                                                                    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

carrier_colors = n.carriers.color.to_dict()
demand_gw = pd.Series(demand / 1000, index=hours)

# ── Dispatch data (generators + storage) ────────────────────────────────
gen_dispatch = n.generators_t.p.T.groupby(n.generators.carrier).sum().T / 1000

# Storage dispatch (positive = discharging, negative = charging)
storage_dispatch = n.storage_units_t.p.T.groupby(n.storage_units.carrier).sum().T / 1000

# For area plots: split storage into discharge (positive) and charge (negative)
dispatch_positive = gen_dispatch.copy()
for col in storage_dispatch.columns:
    dispatch_positive[col] = storage_dispatch[col].clip(lower=0)

charge = pd.DataFrame()
for col in storage_dispatch.columns:
    charge[col] = storage_dispatch[col].clip(upper=0)

# PLOT 1: Winter week with storage
fig, ax = plt.subplots(figsize=(12, 5))
w = dispatch_positive.loc["2019-01-14":"2019-01-20"]
w.plot.area(ax=ax, linewidth=0, color=[carrier_colors.get(c, "gray") for c in w.columns])
wc = charge.loc["2019-01-14":"2019-01-20"]
for col in wc.columns:
    ax.fill_between(wc.index, wc[col], 0, alpha=0.3, color=carrier_colors.get(col, "gray"),
                    label=f"{col} charge")
demand_gw.loc["2019-01-14":"2019-01-20"].plot(ax=ax, color="black", lw=2, ls="--", label="Demand")
ax.set_ylabel("Power [GW]"); ax.set_title("Dispatch with Storage — Winter Week")
ax.legend(loc="upper right", frameon=False, fontsize=8); ax.set_ylim(bottom=ax.get_ylim()[0]*1.1)
plt.tight_layout(); plt.savefig("plot_c_winter_dispatch.png", dpi=150); plt.show()

# PLOT 2: Summer week with storage
fig, ax = plt.subplots(figsize=(12, 5))
s = dispatch_positive.loc["2019-07-15":"2019-07-21"]
s.plot.area(ax=ax, linewidth=0, color=[carrier_colors.get(c, "gray") for c in s.columns])
sc = charge.loc["2019-07-15":"2019-07-21"]
for col in sc.columns:
    ax.fill_between(sc.index, sc[col], 0, alpha=0.3, color=carrier_colors.get(col, "gray"),
                    label=f"{col} charge")
demand_gw.loc["2019-07-15":"2019-07-21"].plot(ax=ax, color="black", lw=2, ls="--", label="Demand")
ax.set_ylabel("Power [GW]"); ax.set_title("Dispatch with Storage — Summer Week")
ax.legend(loc="upper right", frameon=False, fontsize=8); ax.set_ylim(bottom=ax.get_ylim()[0]*1.1)
plt.tight_layout(); plt.savefig("plot_c_summer_dispatch.png", dpi=150); plt.show()

# PLOT 3: State of charge — battery (one week winter + summer)
fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=False)

for su in n.storage_units.index:
    carrier = n.storage_units.loc[su, "carrier"]
    soc = n.storage_units_t.state_of_charge[su] / 1000  # GWh
    color = carrier_colors.get(carrier, "gray")

    # Winter
    soc.loc["2019-01-14":"2019-01-20"].plot(ax=axes[0], label=carrier, color=color, lw=1.5)
    # Summer
    soc.loc["2019-07-15":"2019-07-21"].plot(ax=axes[1], label=carrier, color=color, lw=1.5)

axes[0].set_ylabel("SOC [GWh]"); axes[0].set_title("State of Charge — Winter Week")
axes[0].legend(frameon=False)
axes[1].set_ylabel("SOC [GWh]"); axes[1].set_title("State of Charge — Summer Week")
axes[1].legend(frameon=False)
plt.tight_layout(); plt.savefig("plot_c_state_of_charge.png", dpi=150); plt.show()

# PLOT 4: Full year SOC (to see seasonal patterns)
fig, ax = plt.subplots(figsize=(12, 4))
for su in n.storage_units.index:
    carrier = n.storage_units.loc[su, "carrier"]
    soc = n.storage_units_t.state_of_charge[su] / 1000
    color = carrier_colors.get(carrier, "gray")
    soc.plot(ax=ax, label=carrier, color=color, lw=1)
ax.set_ylabel("State of Charge [GWh]")
ax.set_title("Annual State of Charge Profile")
ax.legend(frameon=False)
plt.tight_layout(); plt.savefig("plot_c_annual_soc.png", dpi=150); plt.show()

# PLOT 5: Annual mix comparison (pie)
annual_gen = n.generators_t.p.T.groupby(n.generators.carrier).sum().sum(axis=1) / 1e6
annual_gen = annual_gen[annual_gen > 0.001]
fig, ax = plt.subplots(figsize=(7, 7))
annual_gen.plot.pie(ax=ax, autopct="%1.1f%%",
    colors=[carrier_colors.get(c, "gray") for c in annual_gen.index],
    startangle=90, textprops={"fontsize": 12})
ax.set_ylabel(""); ax.set_title(f"Annual Electricity Mix with Storage ({annual_gen.sum():.1f} TWh)")
plt.tight_layout(); plt.savefig("plot_c_annual_mix.png", dpi=150); plt.show()

print("\n✅ All Part (c) plots saved!")
