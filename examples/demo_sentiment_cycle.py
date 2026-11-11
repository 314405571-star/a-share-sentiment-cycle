"""Synthetic, offline demo of the sentiment-cycle engine.

A 48-stock, three-sector, 120-day market is generated with a scripted cycle:

    quiet -> main line launches -> climax -> ebb -> dead

Prices are generated bar by bar, then classified by ``src.indicators`` exactly
as live baostock data would be, then scored by ``EmotionEngine``. No network,
no pandas -- just NumPy and Matplotlib.

Run:  python examples/demo_sentiment_cycle.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.indicators import Bar, count_limit_ups, to_signal  # noqa: E402
from src.sentiment import Config, EmotionEngine, State  # noqa: E402

INDUSTRIES = ["Semis", "Robotics", "Liquor"]
PER_SECTOR = 16
N_DAYS = 120
MAIN = "Semis"          # the sector that leads this cycle


# --------------------------------------------------------------------------
# Scripted regime: (limit-up pressure in main sector, elsewhere, break prob)
# --------------------------------------------------------------------------
PHASES = {
    "quiet":               dict(p_main=0.03, p_other=0.010, p_break=0.00, p_ld=0.01, hs=0.0),
    "ignition":            dict(p_main=0.14, p_other=0.015, p_break=0.03, p_ld=0.01, hs=0.0),
    "main rally":          dict(p_main=0.30, p_other=0.020, p_break=0.10, p_ld=0.005, hs=0.0),
    "distribution":        dict(p_main=0.34, p_other=0.020, p_break=0.30, p_ld=0.01, hs=0.0),
    "ebb":                 dict(p_main=0.08, p_other=0.010, p_break=0.40, p_ld=0.15, hs=-0.032),
    "dead":                dict(p_main=0.02, p_other=0.010, p_break=0.02, p_ld=0.01, hs=0.0),
}


def phase(day: int) -> dict:
    if day < 20:
        name = "quiet"
    elif day < 30:
        name = "ignition"
    elif day < 40:
        name = "main rally"
    elif day < 49:
        name = "distribution"
    elif day < 61:
        name = "ebb"
    else:
        name = "dead"
    spec = dict(PHASES[name])
    spec["name"] = name
    if name == "ebb" and day not in (50, 53):
        spec["hs"] = -0.004
    return spec


# Forced board sequences: stock -> set of days where it is sealed / breaks
FORCED_SEALED = {
    "Semis_00": range(22, 34),
    "Semis_01": range(22, 34),
    "Semis_02": range(22, 34),
    "Semis_03": range(29, 36),
    "Semis_04": range(29, 36),
}
FORCED_BREAK = {
    "Semis_00": {36, 37},
    "Semis_01": {36, 37},
    "Semis_02": {36, 37},
    "Semis_03": {37, 38},
    "Semis_04": {37, 38},
}
# two hot names go limit-up -> limit-down on the same day: L3 confirm
FORCED_SKY_FLOOR = {"Semis_00": {40}, "Semis_02": {40}}


def make_market(seed: int = 7):
    rng = np.random.default_rng(seed)
    codes = [f"{ind}_{i:02d}" for ind in INDUSTRIES for i in range(PER_SECTOR)]
    industry = {c: c.split("_")[0] for c in codes}
    close = {c: 10.0 + rng.uniform(0, 20) for c in codes}
    history = {c: [] for c in codes}
    days = []
    hs300 = 3500.0

    for day in range(N_DAYS):
        spec = phase(day)
        p_break, p_ld, name = spec["p_break"], spec["p_ld"], spec["name"]
        hs_ret = float(rng.normal(0, 0.006)) + spec["hs"]
        hs300 *= 1 + hs_ret

        signals = []
        for code in codes:
            preclose = close[code]
            ind = industry[code]
            p_lu = spec["p_main"] if ind == MAIN else spec["p_other"]
            forced = True

            if code in FORCED_SEALED and day in FORCED_SEALED[code]:
                kind = "seal"
            elif code in FORCED_BREAK and day in FORCED_BREAK[code]:
                kind = "break"
            elif code in FORCED_SKY_FLOOR and day in FORCED_SKY_FLOOR[code]:
                kind = "sky"
            else:
                forced = False
                if rng.random() < p_break and history[code]:
                    # only names already hot can produce a meaningful break print
                    hot = count_limit_ups(history[code][-20:]) >= 1
                    kind = "break" if hot and rng.random() < 0.5 else "normal"
                elif rng.random() < p_ld:
                    kind = "limit_down"
                elif rng.random() < p_lu:
                    kind = "seal"
                else:
                    kind = "normal"

            if kind == "seal":
                k = 1.10
                c = round(preclose * k, 2)
                o = round(preclose * (1 + rng.uniform(0.01, 0.06)), 2)
                h = c
                low = round(preclose * (1 + rng.uniform(-0.01, 0.02)), 2)
                turn = float(rng.uniform(2, 16))
                if not forced and rng.random() < 0.04:   # sealed one-word print
                    o = h = c
                    turn = float(rng.uniform(0.1, 0.8))
            elif kind == "break":
                h = round(preclose * 1.099, 2)
                c = round(preclose * (1 + rng.uniform(-0.02, 0.03)), 2)
                o = round(preclose * (1 + rng.uniform(0.02, 0.06)), 2)
                low = round(min(c, preclose) * (1 - rng.uniform(0, 0.02)), 2)
                turn = float(rng.uniform(8, 30))
            elif kind == "sky":
                h = round(preclose * 1.099, 2)
                c = round(preclose * 0.90, 2)
                o = h
                low = c
                turn = float(rng.uniform(15, 35))
            elif kind == "limit_down":
                c = round(preclose * 0.90, 2)
                o = round(preclose * (1 + rng.uniform(-0.03, 0.0)), 2)
                h = round(preclose * (1 + rng.uniform(0.0, 0.01)), 2)
                low = c
                turn = float(rng.uniform(1, 10))
            else:
                c = round(preclose * (1 + float(rng.normal(0, 0.018))), 2)
                o = round(preclose * (1 + float(rng.normal(0, 0.008))), 2)
                h = round(max(o, c) * (1 + abs(float(rng.normal(0, 0.006)))), 2)
                low = round(min(o, c) * (1 - abs(float(rng.normal(0, 0.006)))), 2)
                turn = float(rng.uniform(0.5, 8))

            bar = Bar(code=code, preclose=preclose, open=o, high=h, low=low,
                      close=c, turnover=turn, industry=ind)
            signals.append(to_signal(bar, history[code]))
            history[code].append(bar)
            history[code] = history[code][-60:]
            close[code] = c

        days.append((f"D{day:03d}", signals, hs_ret, name, hs300))

    return days


# --------------------------------------------------------------------------
# Run + report
# --------------------------------------------------------------------------
def main():
    days = make_market()
    engine = EmotionEngine(Config(), State())
    snaps = [engine.update(d, sig, ret) for d, sig, ret, _, _ in days]
    phases = [p for *_, p, _ in days]

    print(f"{'day':>6} {'phase':>7} {'L1':>5} {'L2':>5} {'L3':>8} {'sector':>9} "
          f"{'LU':>4} {'LD':>4} {'HF':>3} {'ebb':>4} {'pos':>4}")
    print("-" * 72)
    for snap, ph in zip(snaps, phases):
        print(f"{snap.date:>6} {ph:>7} {snap.l1:5.1f} {snap.l2:5.1f} {snap.l3:>8} "
              f"{str(snap.main_sector):>9} {snap.limit_up_count:4d} "
              f"{snap.limit_down_count:4d} {snap.high_face_count:3d} "
              f"{str(snap.ebb_confirmed):>5} {snap.position_tier:>4}")

    peaks = {
        "L2 peak": max(snaps, key=lambda s: s.l2),
        "breadth peak": max(snaps, key=lambda s: s.limit_up_count),
        "L1 crash": min(snaps, key=lambda s: s.l1),
        "first ebb": next((s for s in snaps if s.ebb_confirmed), None),
    }
    print("\nKey moments")
    for label, s in peaks.items():
        if s:
            print(f"  {label:14s} {s.date}  L1={s.l1}  L2={s.l2}  L3={s.l3}  "
                  f"LU={s.limit_up_count}  ebb={s.ebb_confirmed}  pos={s.position_tier}")

    _plot(snaps, phases)


def _plot(snaps, phases):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(len(snaps))
    l1 = np.array([s.l1 for s in snaps])
    l2 = np.array([s.l2 for s in snaps])
    lu = np.array([s.limit_up_count for s in snaps])
    ld = np.array([s.limit_down_count for s in snaps])
    hf = np.array([s.high_face_count for s in snaps])
    ebb = np.array([s.ebb_confirmed for s in snaps])
    l3_code = np.array([{"none": 0, "warning": 1, "confirm": 2}[s.l3] for s in snaps])

    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True,
                             gridspec_kw={"height_ratios": [2.0, 2.4, 1.6, 1.2]})

    ax = axes[0]
    ax.axhspan(0, 4, color="#d9534f", alpha=0.08)
    ax.axhspan(4, 7, color="#f0ad4e", alpha=0.08)
    ax.axhspan(7, 10, color="#5cb85c", alpha=0.08)
    ax.plot(x, l1, color="#2c3e50", lw=2)
    ax.set_ylabel("L1 global")
    ax.set_ylim(0, 10.4)
    ax.set_yticks([0, 4, 7, 10])
    ax.set_title("A-share sentiment cycle -- synthetic 120-day demo", fontsize=13)
    ax.grid(alpha=0.25)

    ax = axes[1]
    comp = np.array([[s.l2_parts.tier, s.l2_parts.fry, s.l2_parts.promote,
                      s.l2_parts.face] for s in snaps])
    colors = ["#2980b9", "#27ae60", "#8e44ad", "#e67e22"]
    labels = ["tier", "break rate", "promotion", "high-face"]
    ax.stackplot(x, comp.T, labels=labels, colors=colors, alpha=0.75)
    ax.plot(x, l2, color="black", lw=1.2)
    for xi in np.where(ebb)[0]:
        ax.axvspan(xi - 0.5, xi + 0.5, color="#c0392b", alpha=0.12)
    ax.set_ylabel("L2 main line (0-13)")
    ax.set_ylim(0, 13.5)
    ax.legend(loc="upper left", ncol=4, fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.25)

    ax = axes[2]
    ax.bar(x, lu, color="#c0392b", alpha=0.8, label="limit-up")
    ax.bar(x, -ld, color="#16a085", alpha=0.8, label="limit-down")
    ax.plot(x, hf, color="#8e44ad", lw=1.5, label="high-face")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_ylabel("breadth")
    ax.legend(loc="upper left", fontsize=8, ncol=3)
    ax.grid(alpha=0.25)

    ax = axes[3]
    ax.step(x, l3_code, where="mid", color="#8e44ad", lw=1.5)
    for xi, s in zip(x, snaps):
        if s.ebb_confirmed:
            ax.axvspan(xi - 0.5, xi + 0.5, color="#c0392b", alpha=0.15)
    ax.set_ylabel("L3 signal")
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["none", "warning", "confirm"])
    ax.set_xlabel("trading day")

    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)

    # phase shading
    boundaries = []
    for i in range(1, len(phases)):
        if phases[i] != phases[i - 1]:
            boundaries.append(i)
    for b in boundaries:
        for ax in axes:
            ax.axvline(b, color="grey", ls=":", lw=1)
    label_y = 0.97
    start = 0
    for i in range(1, len(phases) + 1):
        if i == len(phases) or phases[i] != phases[start]:
            axes[0].text((start + i) / 2, label_y, phases[start],
                         ha="center", va="top", fontsize=8, color="grey",
                         transform=axes[0].get_xaxis_transform())
            start = i

    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "figures", "sentiment_cycle_demo.png")
    fig.savefig(out, dpi=150)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
