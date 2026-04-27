"""
Part c — Storage Technologies (battery + hydrogen)
===================================================
Adds battery (intraday) and hydrogen (seasonal) storage to the Part a model.
Investigates how storage changes the optimal mix and at what timescales it
operates.

Run:  python part_c.py
"""

from common import (load_data, build_single_node, CO2_PRICE,
                    CARRIER_COLORS, colors, save_fig,
                    plot_annual_pie)
import pandas as pd
import matplotlib.pyplot as plt

YEAR   = 2019
SOLVER = "gurobi"

# ── Load data & solve ─────────────────────────────────────────────────────────
data = load_data(YEAR)
n    = build_single_node(data, co2_price=CO2_PRICE, with_storage=True)
n.optimize(solver_name=SOLVER, solver_options={"OutputFlag": 0})

# ── Results ───────────────────────────────────────────────────────────────────
print(f"\n{'='*65}\nOPTIMAL CAPACITIES — Part c\n{'='*65}")
for gen in n.generators.index:
    cap = n.generators.loc[gen, "p_nom_opt"]
    if cap > 0:
        cf = n.generators_t.p[gen].sum() / (cap * 8760)
        print(f"  {gen:<22} {cap/1000:>6.2f} GW   CF {cf:.1%}")

print(f"\n  Storage:")
for su in n.storage_units.index:
    p          = n.storage_units.loc[su, "p_nom_opt"]
    h          = n.storage_units.loc[su, "max_hours"]
    carrier    = n.storage_units.loc[su, "carrier"]
    discharged = n.storage_units_t.p[su].clip(lower=0).sum() / 1e6
    charged    = n.storage_units_t.p[su].clip(upper=0).sum() * -1 / 1e6
    print(f"  {carrier:<12} {p:>7,.0f} MW / {p*h:>8,.0f} MWh  "
          f"| charged {charged:.2f} TWh  discharged {discharged:.2f} TWh")

print(f"\n  System cost  {n.objective:>15,.0f} €/yr")
print(f"  Cost/MWh     {n.objective / data['demand'].sum():>15.2f} €/MWh")

# ── Dispatch plots (generators + storage) ─────────────────────────────────────
demand_gw     = pd.Series(data["demand"] / 1000, index=data["hours"])
gen_dispatch  = n.generators_t.p.T.groupby(n.generators.carrier).sum().T / 1000
stor_dispatch = n.storage_units_t.p.T.groupby(n.storage_units.carrier).sum().T / 1000

for period, start, end, label in [
    ("winter", f"{YEAR}-01-14", f"{YEAR}-01-20", "Winter week (Jan 14–20)"),
    ("summer", f"{YEAR}-07-15", f"{YEAR}-07-21", "Summer week (Jul 15–21)"),
]:
    pos = gen_dispatch.copy()
    for col in stor_dispatch.columns:
        pos[col] = stor_dispatch[col].clip(lower=0)
    fig, ax = plt.subplots(figsize=(12, 5))
    w = pos.loc[start:end]
    w.plot.area(ax=ax, linewidth=0, color=colors(w.columns))
    for col in stor_dispatch.columns:
        charge = stor_dispatch[col].clip(upper=0)
        ax.fill_between(charge.loc[start:end].index,
                        charge.loc[start:end], 0, alpha=0.3,
                        color=CARRIER_COLORS.get(col, "gray"),
                        label=f"{col} charging")
    demand_gw.loc[start:end].plot(ax=ax, color="black", lw=2,
                                   ls="--", label="Demand")
    ax.set_ylabel("Power [GW]")
    ax.set_title(f"Dispatch with storage — {label}")
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    save_fig(fig, f"c_{period}_dispatch.png")

# ── State of charge — weekly subplots ─────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=False)
for su in n.storage_units.index:
    c   = n.storage_units.loc[su, "carrier"]
    soc = n.storage_units_t.state_of_charge[su] / 1000
    col = CARRIER_COLORS.get(c, "gray")
    soc.loc[f"{YEAR}-01-14":f"{YEAR}-01-20"].plot(ax=axes[0], label=c, color=col, lw=1.5)
    soc.loc[f"{YEAR}-07-15":f"{YEAR}-07-21"].plot(ax=axes[1], label=c, color=col, lw=1.5)
for ax, title in zip(axes, ["SOC — winter week", "SOC — summer week"]):
    ax.set_ylabel("SOC [GWh]")
    ax.set_title(title)
    ax.legend(frameon=False)
plt.tight_layout()
save_fig(fig, "c_state_of_charge.png")

# ── Full-year SOC ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 4))
for su in n.storage_units.index:
    c   = n.storage_units.loc[su, "carrier"]
    soc = n.storage_units_t.state_of_charge[su] / 1000
    soc.plot(ax=ax, label=c, color=CARRIER_COLORS.get(c, "gray"), lw=1)
ax.set_ylabel("State of charge [GWh]")
ax.set_title("Annual state of charge profile")
ax.legend(frameon=False)
save_fig(fig, "c_annual_soc.png")

plot_annual_pie(n, f"Annual mix with storage {YEAR}", "c_annual_mix.png")

print("\n✅  Part c done — plots saved to outputs/")