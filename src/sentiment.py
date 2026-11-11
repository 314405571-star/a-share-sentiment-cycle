"""The sentiment-cycle state machine: L1 / L2 / L3, ebb confirmation, sizing.

The model is an explicit rule system, not a statistical factor. It answers one
question every trading day:

    "Is the speculative risk appetite of the market expanding, holding,
     or contracting -- and if contracting, is it confirmed?"

Three layers:

* **L1 (global)**  0-10. Index stress, limit-down/limit-up ratio, high-face count.
* **L2 (main line)** 0-13. Tier completeness, break rate vs. its own history,
  promotion rate vs. its own history, high-face penalty.
* **L3 (leader)** ``none`` / ``warning`` / ``confirm``. Leader failed to seal,
  or sky-floor events in the hot pool.

L2 relies on *deviations from a rolling baseline* (fry rate, promotion rate)
rather than absolute levels, so a permanently high-turnover market does not
keep the score pinned. The baseline is the mean of the stored daily sequence
(last 20 entries, floor 0.3), appended once per trading day.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence

from .indicators import StockSignal

L1_MAX = 10.0
L2_MAX = 13.0


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


@dataclass
class Config:
    # --- L1 ---
    systemic_drop: float = -0.025      # HS300 daily return that trips risk-off
    systemic_penalty: float = 4.0
    drop_ratio: float = 1.2            # limit_down / limit_up that flips the ratio
    drop_persistence_penalty: float = 3.0
    high_face_soft: int = 8
    high_face_hard: int = 15

    # --- main sector ---
    sector_min_count: int = 3          # a sector needs this many boards to lead
    sector_hold_ratio: float = 0.5     # incumbent keeps leadership at >= 50% of top

    # --- L3 ---
    leader_break_warn: int = 1
    leader_break_confirm: int = 2
    sky_floor_warn: int = 1
    sky_floor_confirm: int = 2

    # --- ebb confirmation ---
    ebb_l2: float = 6.0
    ebb_prev_l2: float = 7.0

    # --- baselines ---
    seq_len: int = 60
    baseline_window: int = 20
    default_fry_rate: float = 0.3
    default_promote_rate: float = 0.3

    # --- position tiers on the real 0..13 L2 scale ---
    position_tiers: tuple = (
        (11.0, 0.25, 0.90, "T5"),
        (8.0, 0.20, 0.70, "T4"),
        (5.0, 0.10, 0.30, "T3"),
        (2.0, 0.05, 0.10, "T2"),
    )


# --------------------------------------------------------------------------
# State
# --------------------------------------------------------------------------


@dataclass
class State:
    """Everything the next day needs. Serialised to JSON between runs."""

    prev_main_sector: Optional[str] = None
    prev_sector_hl: int = 0
    prev_hl_codes: List[str] = field(default_factory=list)
    prev_l2: Optional[float] = None
    prev_drop_trigger: bool = False
    leader_code: Optional[str] = None
    leader_break_days: int = 0
    fry_rate_seq: List[float] = field(default_factory=list)
    promote_rate_seq: List[float] = field(default_factory=list)

    # -- persistence ------------------------------------------------------
    def save(self, path: str) -> None:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str) -> "State":
        if not os.path.exists(path):
            return cls()
        with open(path, encoding="utf-8") as f:
            return cls(**json.load(f))


# --------------------------------------------------------------------------
# Result objects
# --------------------------------------------------------------------------


@dataclass
class L2Parts:
    tier: int = 0
    fry: int = 0
    promote: int = 0
    face: int = 0

    def total(self) -> int:
        return min(int(L2_MAX), self.tier + self.fry + self.promote + self.face)


@dataclass
class Snapshot:
    date: str
    l1: float
    mood: str
    systemic_risk: bool
    drop_trigger: bool
    limit_up_count: int
    limit_down_count: int
    high_face_count: int
    main_sector: Optional[str]
    main_sector_hl: int
    sector_sticky: bool
    l2: float
    l2_parts: L2Parts
    l3: str
    leader: Optional[str]
    leader_break_days: int
    sky_floor_count: int
    fry_rate: float
    promote_rate: float
    ebb_confirmed: bool
    thresholds: Dict[str, object]
    single_position: float
    total_position: float
    position_tier: str

    def as_dict(self) -> dict:
        d = asdict(self)
        return d

    def report(self) -> str:
        tide = "🔴 confirmed -- flat, no new entries" if self.ebb_confirmed else "🟢 not triggered"
        return (
            "==============================================\n"
            f"  sentiment cycle report ({self.date})\n"
            "==============================================\n"
            f"  L1 global      {self.l1:4.1f}/10   regime: {self.mood}\n"
            f"  L2 main line   {self.l2:4.1f}/13   "
            f"(tier {self.l2_parts.tier} fry {self.l2_parts.fry} "
            f"promote {self.l2_parts.promote} face {self.l2_parts.face})\n"
            f"  L3 leader      {self.l3} (leader {self.leader}, "
            f"{self.leader_break_days} day(s) unsealed)\n"
            f"  two-day ebb    {tide}\n"
            "----------------------------------------------\n"
            f"  breadth        limit-up {self.limit_up_count}  "
            f"limit-down {self.limit_down_count}  high-face {self.high_face_count}\n"
            f"  main sector    {self.main_sector} ({self.main_sector_hl} boards"
            f"{', sticky' if self.sector_sticky else ''})\n"
            f"  fry / promote  {self.fry_rate:.2f} / {self.promote_rate:.2f}\n"
            "----------------------------------------------\n"
            f"  stop loss {self.thresholds['stop_loss']}   "
            f"auction premium {self.thresholds['auction_premium']}\n"
            f"  position: single {self.single_position:.0%}  "
            f"total {self.total_position:.0%}  ({self.position_tier})\n"
            "=============================================="
        )


# --------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------


class EmotionEngine:
    def __init__(self, config: Optional[Config] = None, state: Optional[State] = None):
        self.cfg = config or Config()
        self.state = state or State()
        self.history: List[Snapshot] = []

    # -- public -----------------------------------------------------------
    def update(self, date: str, signals: Sequence[StockSignal],
               hs300_return: Optional[float]) -> Snapshot:
        """Consume one trading day of already-classified stock signals."""
        cfg = self.cfg

        hot = [s for s in signals if s.limit_up or s.break_up]
        sealed_codes = {s.code for s in signals if s.limit_up}
        limit_down = sum(1 for s in signals if s.limit_down)
        high_face = sum(1 for s in signals if s.high_face)
        sky_floor = sum(1 for s in signals if s.sky_floor)

        sector, sector_hl, sticky = self._detect_main_sector(hot)
        l1, systemic, drop_trigger = self._l1(hs300_return, len(sealed_codes),
                                              limit_down, high_face)

        # Today's rates are measured against yesterday's baseline, then stored.
        fry_rate = self._today_fry_rate(hot)
        promote_rate = self._today_promote_rate(sealed_codes)
        avg_fry = self._baseline(self.state.fry_rate_seq, cfg.default_fry_rate)
        avg_promote = self._baseline(self.state.promote_rate_seq, cfg.default_promote_rate)

        parts = self._l2_parts(hot, sector, high_face, fry_rate, avg_fry,
                               promote_rate, avg_promote)
        l2 = parts.total()

        l3_signal, leader, break_days = self._l3(sector, hot, sealed_codes, sky_floor)
        ebb = (self.state.prev_l2 is not None
               and l2 <= cfg.ebb_l2
               and self.state.prev_l2 <= cfg.ebb_prev_l2
               and l3_signal in ("warning", "confirm"))

        mood, thresholds = self.dynamic_params(l1)
        single_pos, total_pos, tier = self.position_advice(l2, systemic)

        snap = Snapshot(
            date=date, l1=l1, mood=mood, systemic_risk=systemic,
            drop_trigger=drop_trigger, limit_up_count=len(sealed_codes),
            limit_down_count=limit_down, high_face_count=high_face,
            main_sector=sector, main_sector_hl=sector_hl, sector_sticky=sticky,
            l2=l2, l2_parts=parts, l3=l3_signal, leader=leader,
            leader_break_days=break_days, sky_floor_count=sky_floor,
            fry_rate=fry_rate, promote_rate=promote_rate, ebb_confirmed=ebb,
            thresholds=thresholds, single_position=single_pos,
            total_position=total_pos, position_tier=tier,
        )

        self._advance_state(snap, sealed_codes, drop_trigger, fry_rate, promote_rate)
        self.history.append(snap)
        return snap

    # -- scoring ----------------------------------------------------------
    def _l1(self, hs300_return, up: int, down: int, high_face: int):
        cfg = self.cfg
        score = L1_MAX
        systemic = False
        if hs300_return is not None and hs300_return <= cfg.systemic_drop:
            score -= cfg.systemic_penalty
            systemic = True

        drop_trigger = up > 0 and (down / up) > cfg.drop_ratio
        if drop_trigger and self.state.prev_drop_trigger:
            score -= cfg.drop_persistence_penalty

        if high_face >= cfg.high_face_hard:
            score -= 3
        elif high_face >= cfg.high_face_soft:
            score -= 1

        return max(0.0, min(L1_MAX, score)), systemic, drop_trigger

    def _detect_main_sector(self, hot: Sequence[StockSignal]):
        """Largest sector by board count, with hysteresis on the incumbent.

        A sector with fewer than ``sector_min_count`` boards is not a main
        line -- it is noise. The original code always returned the argmax, so
        a single stray limit-up in a dead market was treated as a main line and
        scored L2 ~7. Gating here is what makes "no main line" representable.
        """
        counts = Counter(s.industry for s in hot if s.industry)
        if not counts:
            return None, 0, False
        top_name, top_n = max(counts.items(), key=lambda kv: kv[1])

        prev = self.state.prev_main_sector
        if prev:
            prev_n = counts.get(prev, 0)
            floor = max(self.cfg.sector_min_count, top_n * self.cfg.sector_hold_ratio)
            if prev_n >= floor:
                return prev, prev_n, True

        if top_n < self.cfg.sector_min_count:
            return None, top_n, False
        return top_name, top_n, False

    def _today_fry_rate(self, hot: Sequence[StockSignal]) -> float:
        """Share of high-board (>=3) names in the hot pool that failed to seal."""
        valid = [s for s in hot if s.board_num >= 3]
        if not valid:
            return 0.0
        return sum(1 for s in valid if s.is_break) / len(valid)

    def _today_promote_rate(self, sealed_codes: set) -> float:
        """Share of yesterday's sealed names that sealed again today."""
        prev = self.state.prev_hl_codes
        if not prev:
            return 0.0
        return len([c for c in prev if c in sealed_codes]) / len(prev)

    def _baseline(self, seq: Sequence[float], default: float) -> float:
        window = list(seq)[-self.cfg.baseline_window:]
        return sum(window) / len(window) if len(window) >= self.cfg.baseline_window else default

    def _l2_parts(self, hot, sector, high_face, fry_rate, avg_fry,
                  promote_rate, avg_promote) -> L2Parts:
        parts = L2Parts()
        if not sector:
            return parts
        main_hl = [s for s in hot if s.industry == sector]

        # 1. tier completeness: how many 4-board+ names exist in the main line
        four_plus = len([s for s in main_hl if s.board_num >= 4])
        parts.tier = 4 if four_plus >= 3 else (2 if four_plus >= 1 else 0)

        # 2. break rate vs its own 20-day baseline
        # Small-sample guard: with 0-1 high-board names a 0% break rate carries
        # no information, yet the original rule scored it 4/4 and pinned L2 up
        # in a dead market.
        if len([s for s in main_hl if s.board_num >= 3]) < 2:
            parts.fry = 2
        else:
            diff = fry_rate - avg_fry
            parts.fry = 4 if diff < 0 else (2 if diff <= 0.15 else 0)

        # 3. promotion rate vs its own 20-day baseline
        dev = promote_rate - avg_promote
        parts.promote = 2 if dev >= 0 else (1 if dev >= -0.20 else 0)

        # 4. high-face penalty, with a sky-floor escalation
        if high_face == 0:
            parts.face = 3
        elif high_face <= 5:
            parts.face = 1
        else:
            has_sky = any(s.sky_floor for s in hot)
            parts.face = -1 if has_sky else 0
        return parts

    def _l3(self, sector, hot, sealed_codes, sky_floor_count):
        """Returns (signal, leader_code, break_days).

        A change of leader resets the counter -- yesterday's leader failing is
        no longer informative once the market has anointed someone else.
        """
        cfg = self.cfg
        if not sector:
            return "none", None, 0

        leader = self.pick_leader(sector, hot)
        prev_leader = self.state.leader_code

        # No sealed candidate today: the previous leader is still the reference
        # point the market is watching (it is what failed).
        if leader is None:
            leader = prev_leader
        if leader is None:
            return "none", None, 0

        break_days = self.state.leader_break_days
        if leader != prev_leader and prev_leader is not None:
            break_days = 0

        sealed = leader in sealed_codes
        new_break_days = 0 if sealed else break_days + 1

        if new_break_days >= cfg.leader_break_confirm or sky_floor_count >= cfg.sky_floor_confirm:
            signal = "confirm"
        elif new_break_days >= cfg.leader_break_warn or sky_floor_count >= cfg.sky_floor_warn:
            signal = "warning"
        else:
            signal = "none"
        return signal, leader, new_break_days

    @staticmethod
    def pick_leader(sector, hot) -> Optional[str]:
        """Highest board count, ties broken by turnover."""
        if not sector:
            return None
        cands = [s for s in hot if s.industry == sector and not s.is_break]
        if not cands:
            return None
        cands.sort(key=lambda s: (-s.board_num, -s.turnover))
        return cands[0].code

    # -- advice -----------------------------------------------------------
    @staticmethod
    def dynamic_params(l1: float):
        if l1 <= 4:
            return "panic", {"stop_loss": "-3%", "auction_premium": "+1%",
                             "limit_down_capacity": 5, "min_entry_score": 38}
        if l1 <= 7:
            return "neutral", {"stop_loss": "-5%", "auction_premium": "+1.5%",
                               "limit_down_capacity": 10, "min_entry_score": 45}
        return "euphoric", {"stop_loss": "-7%", "auction_premium": "+2%",
                            "limit_down_capacity": 15, "min_entry_score": 55}

    def position_advice(self, l2: float, systemic: bool):
        """Position caps on the true 0..13 L2 scale.

        The original notebook cut at L2 >= 16 for the top tier, but L2 is
        capped at 13 by construction, so the 25%/90% tier was unreachable and
        the top of the scale collapsed onto 20%/70%. Fixed here by cutting on
        11 / 8 / 5 / 2, which maps onto the four reachable tiers.
        """
        if systemic:
            return 0.0, 0.0, "T0 (flat)"
        for cut, single, total, name in self.cfg.position_tiers:
            if l2 >= cut:
                return single, total, name
        return 0.0, 0.0, "T1 (flat)"

    # -- state transition -------------------------------------------------
    def _advance_state(self, snap, sealed_codes, drop_trigger, fry_rate, promote_rate):
        s = self.state
        s.prev_main_sector = snap.main_sector
        s.prev_sector_hl = snap.main_sector_hl
        s.prev_hl_codes = sorted(sealed_codes)
        s.prev_l2 = snap.l2
        s.prev_drop_trigger = drop_trigger
        s.leader_code = snap.leader
        s.leader_break_days = snap.leader_break_days
        s.fry_rate_seq = (s.fry_rate_seq + [fry_rate])[-self.cfg.seq_len:]
        s.promote_rate_seq = (s.promote_rate_seq + [promote_rate])[-self.cfg.seq_len:]


# Convenience wrapper for the common "one call per day" use case
def run_days(days, config: Optional[Config] = None,
             state: Optional[State] = None) -> List[Snapshot]:
    """`days` is an iterable of (date, signals, hs300_return)."""
    engine = EmotionEngine(config, state)
    return [engine.update(d, sig, ret) for d, sig, ret in days]


__all__ = ["Config", "State", "Snapshot", "L2Parts", "EmotionEngine", "run_days",
           "L1_MAX", "L2_MAX"]
