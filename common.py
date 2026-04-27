"""
common.py
=========
Shared code for the IEG Course Project (DTU 46770).
Import what you need in each part script — nothing here runs on import.

Sections
--------
1. Imports & pandas fix
2. Financial assumptions
3. Technology cost data          (DEA Technology Catalogue, 2024)
4. Derived capital & marginal costs
5. Carrier colours
6. Data loading
7. Network builders
8. Shared plot helpers
"""

# ── CRITICAL: must come before importing pypsa ────────────────────────────────
import pandas as pd
pd.options.future.infer_string = False

import pypsa
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Plots are saved here — created automatically if it doesn't exist
OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

DATA_DIR = Path(__file__).parent / "data"


# =============================================================================
# 1. FINANCIAL ASSUMPTIONS
# =============================================================================

DISCOUNT_RATE = 0.04   # 4 % real discount rate
CO2_PRICE     = 80.0   # €/tCO2  (EU ETS reference, Parts a–d)


def annuity(lifetime, rate=DISCOUNT_RATE):
    """One-time investment [€/MW] → equal annual payments [€/MW/yr]."""
    return rate / (1 - (1 + rate) ** (-lifetime))


# =============================================================================
# 2. TECHNOLOGY COST DATA  (DEA Technology Catalogue, 2024)
# =============================================================================

TECH = {
    # ── generators ────────────────────────────────────────────────────────────
    "onshore_wind": {
        "investment":  1_100_000,  # €/MW
        "fixed_om":       12_900,  # €/MW/yr
        "variable_om":       2.7,  # €/MWh
        "lifetime":           30,
    },
    "offshore_wind": {
        "investment":  2_000_000,
        "fixed_om":       25_000,
        "variable_om":       3.3,
        "lifetime":           30,
    },
    "solar_pv": {
        "investment":    420_000,
        "fixed_om":        7_500,
        "variable_om":       0.0,
        "lifetime":           30,
    },
    "ccgt": {
        "investment":    800_000,
        "fixed_om":       25_000,
        "variable_om":       4.0,
        "lifetime":           25,
        "efficiency":       0.58,
        "fuel_cost":        25.0,  # €/MWh_th
        "co2_intensity":     0.2,  # tCO2/MWh_th
    },
    "ocgt": {
        "investment":    400_000,
        "fixed_om":       15_000,
        "variable_om":       4.0,
        "lifetime":           25,
        "efficiency":       0.40,
        "fuel_cost":        25.0,
        "co2_intensity":     0.2,
    },
    "biomass": {
        "investment":  2_500_000,
        "fixed_om":       50_000,
        "variable_om":       5.0,
        "lifetime":           25,
        "efficiency":       0.33,
        "fuel_cost":         9.0,  # €/MWh_th  (wood pellets)
        "co2_intensity":     0.0,  # biogenic carbon = 0 under EU ETS
    },
    # ── storage (Part c) ──────────────────────────────────────────────────────
    "battery": {
        "power_cost":    150_000,  # €/MW
        "energy_cost":   150_000,  # €/MWh
        "fixed_om":        5_000,  # €/MW/yr
        "efficiency_rt":    0.92,  # round-trip
        "lifetime":           15,
        "max_hours":           6,
    },
    "hydrogen": {
        "power_cost":    800_000,  # €/MW  (electrolyser + fuel cell)
        "energy_cost":     2_000,  # €/MWh (underground cavern)
        "fixed_om":       20_000,  # €/MW/yr
        "efficiency_rt":    0.35,  # round-trip (65 % × 55 %)
        "lifetime":           25,
        "max_hours":         168,  # up to 1 week
    },
}


# =============================================================================
# 3. DERIVED CAPITAL & MARGINAL COSTS
# =============================================================================

def capital_cost(key):
    """Annualised capital cost [€/MW/yr]."""
    t   = TECH[key]
    inv = t.get("power_cost", t["investment"])
    return annuity(t["lifetime"]) * inv + t["fixed_om"]


def marginal_cost(key, co2_price=CO2_PRICE):
    """
    Marginal generation cost [€/MWh_el].
    Renewables/storage → variable_om only.
    Thermal → fuel/eff + co2×intensity/eff + variable_om.
    """
    t  = TECH[key]
    mc = t["variable_om"]
    if "fuel_cost" in t:
        mc += t["fuel_cost"]   / t["efficiency"]
        mc += co2_price * t["co2_intensity"] / t["efficiency"]
    return mc


# Pre-computed at the default CO2_PRICE — import these directly in part scripts
CC = {k: capital_cost(k) for k in TECH}   # capital costs
MC = {k: marginal_cost(k) for k in        # marginal costs
      ["onshore_wind", "offshore_wind", "solar_pv", "ccgt", "ocgt", "biomass"]}


# =============================================================================
# 4. CARRIER COLOURS
# =============================================================================

CARRIER_COLORS = {
    "onshore wind":  "dodgerblue",
    "offshore wind": "navy",
    "solar":         "gold",
    "CCGT":          "indianred",
    "OCGT":          "salmon",
    "biomass":       "forestgreen",
    "battery":       "mediumpurple",
    "hydrogen":      "cyan",
}


# =============================================================================
# 5. DATA LOADING
# =============================================================================

def load_data(year):
    """
    Load hourly data from data/DK_clean_{year}.csv.

    Returns dict with keys:
        hours, demand, cf_onshore, cf_offshore, cf_solar
    """
    path = DATA_DIR / f"DK_clean_{year}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing: {path}\n"
            f"Put DK_clean_{year}.csv inside the data/ folder."
        )
    df = pd.read_csv(path)
    return {
        "hours":       pd.DatetimeIndex(pd.to_datetime(df["timestamp"].astype(str))),
        "demand":      df["demand_MW"].astype(float).to_numpy(),
        "cf_onshore":  df["cf_onshore_wind"].astype(float).to_numpy(),
        "cf_offshore": df["cf_offshore_wind"].astype(float).to_numpy(),
        "cf_solar":    df["cf_solar"].astype(float).to_numpy(),
    }


# =============================================================================
# 6. NETWORK BUILDERS
# =============================================================================

# Maps internal key → PyPSA carrier string
_CARRIER = {
    "onshore_wind":  "onshore wind",
    "offshore_wind": "offshore wind",
    "solar_pv":      "solar",
    "ccgt":          "CCGT",
    "ocgt":          "OCGT",
    "biomass":       "biomass",
}

# CF arrays to attach per technology
_CF_KEY = {
    "onshore_wind":  "cf_onshore",
    "offshore_wind": "cf_offshore",
    "solar_pv":      "cf_solar",
}


def _add_carriers(n, with_storage=False):
    n.add("Carrier", "onshore wind",  nice_name="Onshore Wind",     color="dodgerblue")
    n.add("Carrier", "offshore wind", nice_name="Offshore Wind",     color="navy")
    n.add("Carrier", "solar",         nice_name="Solar PV",          color="gold")
    n.add("Carrier", "CCGT",          nice_name="CCGT (Gas)",        color="indianred")
    n.add("Carrier", "OCGT",          nice_name="OCGT (Gas Peaker)", color="salmon")
    n.add("Carrier", "biomass",       nice_name="Biomass",           color="forestgreen")
    if with_storage:
        n.add("Carrier", "battery",   nice_name="Battery",           color="mediumpurple")
        n.add("Carrier", "hydrogen",  nice_name="Hydrogen Storage",  color="cyan")


def _add_generators(n, bus, data, technologies, co2_price=CO2_PRICE,
                    biomass_cap_mw=None):
    """Add extendable generators for one bus."""
    for key in technologies:
        kwargs = dict(
            bus=bus,
            carrier=_CARRIER[key],
            p_nom_extendable=True,
            capital_cost=capital_cost(key),
            marginal_cost=marginal_cost(key, co2_price),
        )
        if key in _CF_KEY:
            kwargs["p_max_pu"] = data[_CF_KEY[key]]
        if key == "biomass" and biomass_cap_mw is not None:
            kwargs["p_nom_max"] = biomass_cap_mw
        n.add("Generator", f"{bus} {_CARRIER[key]}", **kwargs)


def _add_storage(n, bus):
    """Add battery and hydrogen StorageUnits to a bus."""
    bat = TECH["battery"]
    n.add("StorageUnit", f"{bus} battery",
          bus=bus, carrier="battery",
          p_nom_extendable=True,
          capital_cost=CC["battery"],
          max_hours=bat["max_hours"],
          efficiency_store=bat["efficiency_rt"] ** 0.5,
          efficiency_dispatch=bat["efficiency_rt"] ** 0.5,
          cyclic_state_of_charge=True,
          marginal_cost=0.0)
    h2 = TECH["hydrogen"]
    n.add("StorageUnit", f"{bus} hydrogen",
          bus=bus, carrier="hydrogen",
          p_nom_extendable=True,
          capital_cost=CC["hydrogen"],
          max_hours=h2["max_hours"],
          efficiency_store=h2["efficiency_rt"] ** 0.5,
          efficiency_dispatch=h2["efficiency_rt"] ** 0.5,
          cyclic_state_of_charge=True,
          marginal_cost=0.0)


def build_single_node(data, co2_price=CO2_PRICE, with_storage=False,
                      co2_limit=None):
    """
    Single-bus Denmark copper-plate model.

    Parameters
    ----------
    data         : dict from load_data(year)
    co2_price    : €/tCO2 baked into marginal costs (Parts a–c use 80).
                   Pass 0 when using co2_limit to avoid double-counting.
    with_storage : add battery + hydrogen StorageUnits  (Part c)
    co2_limit    : annual CO2 cap [tonnes] — adds a GlobalConstraint  (Part f)

    Returns unsolved pypsa.Network.
    """
    n = pypsa.Network()
    n.set_snapshots(data["hours"])

    _add_carriers(n, with_storage=with_storage)
    n.add("Carrier", "electricity")
    n.add("Bus",  "DK", carrier="electricity")
    n.add("Load", "DK demand", bus="DK", p_set=data["demand"], carrier="electricity")

    _add_generators(n, "DK", data,
                    ["onshore_wind", "offshore_wind", "solar_pv",
                     "ccgt", "ocgt", "biomass"],
                    co2_price=co2_price, biomass_cap_mw=1500)

    if with_storage:
        _add_storage(n, "DK")

    if co2_limit is not None:
        for key, name in [("ccgt", "DK CCGT"), ("ocgt", "DK OCGT")]:
            t = TECH[key]
            n.generators.loc[name, "co2_emissions"] = (
                t["co2_intensity"] / t["efficiency"]
            )
        n.add("GlobalConstraint", "co2_cap",
              type="primary_energy",
              carrier_attribute="co2_emissions",
              sense="<=",
              constant=co2_limit)

    return n


def build_multi_node(data, co2_price=CO2_PRICE):
    """
    Four-bus model: DK–DE–SE–NO connected via HVAC lines (DC approximation).

    Neighbouring demands are scaled from the DK profile:
        DE × 4.0 | SE × 2.0 | NO × 1.5

    Line capacities (existing NTC values) [MW]:
        DK-DE 3000 | DK-SE 2000 | DK-NO 1500 | SE-NO 1000

    Returns unsolved pypsa.Network.
    """
    demand    = data["demand"]
    demand_DE = 4.0 * demand
    demand_SE = 2.0 * demand
    demand_NO = 1.5 * demand

    n = pypsa.Network()
    n.set_snapshots(data["hours"])
    _add_carriers(n)
    n.add("Carrier", "AC")

    for bus in ["DK", "DE", "SE", "NO"]:
        n.add("Bus", bus, carrier="AC")

    for name, b0, b1, s_nom in [
        ("DK-DE", "DK", "DE", 3000),
        ("DK-SE", "DK", "SE", 2000),
        ("DK-NO", "DK", "NO", 1500),
        ("SE-NO", "SE", "NO", 1000),
    ]:
        n.add("Line", name, bus0=b0, bus1=b1,
              x=0.1, r=0.01, s_nom=s_nom, carrier="AC")

    n.add("Load", "DK demand", bus="DK", p_set=demand)
    n.add("Load", "DE demand", bus="DE", p_set=demand_DE)
    n.add("Load", "SE demand", bus="SE", p_set=demand_SE)
    n.add("Load", "NO demand", bus="NO", p_set=demand_NO)

    # DK — full mix, biomass capped at 1500 MW
    _add_generators(n, "DK", data,
                    ["onshore_wind", "offshore_wind", "solar_pv",
                     "ccgt", "ocgt", "biomass"],
                    co2_price=co2_price, biomass_cap_mw=1500)
    # DE — wind + solar + gas
    _add_generators(n, "DE", data,
                    ["onshore_wind", "solar_pv", "ccgt"],
                    co2_price=co2_price)
    # SE — wind + biomass
    _add_generators(n, "SE", data,
                    ["onshore_wind", "biomass"],
                    co2_price=co2_price)
    # NO — biomass only (flexible hydro proxy)
    _add_generators(n, "NO", data,
                    ["biomass"],
                    co2_price=co2_price)

    return n


# =============================================================================
# 7. SHARED PLOT HELPERS
# =============================================================================

def colors(carriers):
    """Return a list of colours matching a list of carrier names."""
    return [CARRIER_COLORS.get(c, "gray") for c in carriers]


def save_fig(fig, filename):
    """Save figure to outputs/ and close it."""
    fig.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_dispatch(n, demand_gw, start, end, title, filename,
                  system_demand_gw=None):
    """Stacked area dispatch plot for a date range."""
    dispatch = n.generators_t.p.T.groupby(n.generators.carrier).sum().T / 1000
    fig, ax  = plt.subplots(figsize=(12, 5))
    w = dispatch.loc[start:end]
    w.plot.area(ax=ax, linewidth=0, color=colors(w.columns))
    demand_gw.loc[start:end].plot(ax=ax, color="black", lw=2,
                                   ls="--", label="Demand")
    if system_demand_gw is not None:
        system_demand_gw.loc[start:end].plot(ax=ax, color="dimgray",
                                              lw=1.5, ls=":", label="System demand")
    ax.set_ylabel("Power [GW]")
    ax.set_title(title)
    ax.legend(loc="upper right", frameon=False)
    ax.set_ylim(0)
    save_fig(fig, filename)


def plot_annual_pie(n, title, filename):
    """Pie chart of annual generation mix."""
    annual = (n.generators_t.p.T.groupby(n.generators.carrier)
              .sum().sum(axis=1) / 1e6)
    annual = annual[annual > 0.001]
    fig, ax = plt.subplots(figsize=(7, 7))
    annual.plot.pie(ax=ax, autopct="%1.1f%%", colors=colors(annual.index),
                    startangle=90, textprops={"fontsize": 12})
    ax.set_ylabel("")
    ax.set_title(f"{title} ({annual.sum():.1f} TWh)")
    save_fig(fig, filename)


def plot_duration_curves(n, title, filename):
    """Duration curves for all active carriers."""
    dispatch = n.generators_t.p.T.groupby(n.generators.carrier).sum().T / 1000
    fig, ax  = plt.subplots(figsize=(10, 5))
    for c in dispatch.columns:
        if dispatch[c].sum() > 0:
            s = dispatch[c].sort_values(ascending=False).values
            ax.plot(np.arange(len(s)), s, label=c,
                    color=CARRIER_COLORS.get(c, "gray"), lw=1.5)
    ax.set_xlabel("Hours (sorted)")
    ax.set_ylabel("Power [GW]")
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.set_xlim(0, 8760)
    save_fig(fig, filename)


def plot_capacity_bar(n, title, filename):
    """Bar chart of optimal installed capacities."""
    cap    = n.generators.p_nom_opt / 1000
    cap    = cap[cap > 0.001]
    clrs   = [CARRIER_COLORS.get(n.generators.loc[g, "carrier"], "gray")
              for g in cap.index]
    fig, ax = plt.subplots(figsize=(10, 4))
    cap.plot.bar(ax=ax, color=clrs, edgecolor="black")
    ax.set_ylabel("Capacity [GW]")
    ax.set_title(title)
    for i, (_, v) in enumerate(cap.items()):
        ax.text(i, v + 0.05, f"{v:.2f}", ha="center", fontsize=9)
    plt.xticks(rotation=45, ha="right")
    save_fig(fig, filename)