"""Unit tests for the scoring state machine.  Run: python tests/test_sentiment.py"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.indicators import Flags, StockSignal  # noqa: E402
from src.sentiment import Config, EmotionEngine, State  # noqa: E402


def sig(code, industry="Semis", lu=False, ld=False, brk=False, hf=False,
        sky=False, board=0, turnover=5.0):
    return StockSignal(code=code, name=code, industry=industry, turnover=turnover,
                       close=11.0, preclose=10.0,
                       flags=Flags(limit_up=lu, limit_down=ld, break_up=brk,
                                   high_face=hf, sky_floor=sky, board_num=board))


def test_l1_systemic_risk_forces_flat():
    eng = EmotionEngine()
    snap = eng.update("d1", [], hs300_return=-0.03)
    assert snap.l1 == 6.0 and snap.systemic_risk
    assert snap.position_tier.startswith("T0")
    assert snap.single_position == 0.0 and snap.total_position == 0.0


def test_l1_drop_ratio_penalty_needs_two_days():
    eng = EmotionEngine()
    day1 = [sig("A", lu=True), sig("B", ld=True), sig("C", ld=True)]
    assert eng.update("d1", day1, 0.0).l1 == 10.0      # trigger, first day
    snap2 = eng.update("d2", day1, 0.0)
    assert snap2.drop_trigger and snap2.l1 == 7.0      # persistence penalty


def test_l1_high_face_penalty():
    eng = EmotionEngine()
    soft = [sig(f"H{i}", hf=True) for i in range(Config().high_face_soft)]
    hard = [sig(f"H{i}", hf=True) for i in range(Config().high_face_hard)]
    assert EmotionEngine().update("d", soft, 0.0).l1 == 9.0
    assert EmotionEngine().update("d", hard, 0.0).l1 == 7.0


def test_sector_gate_needs_enough_boards():
    """One or two stray limit-ups are not a main line."""
    eng = EmotionEngine()
    assert eng.update("d1", [sig("A", lu=True), sig("B", lu=True)], 0.0).main_sector is None
    assert eng.update("d1", [sig("A", lu=True)], 0.0).l2 == 0.0


def test_sector_hysteresis_keeps_incumbent():
    state = State(prev_main_sector="Semis")
    eng = EmotionEngine(state=state)
    day = [sig(f"S{i}", "Semis", lu=True) for i in range(3)]
    day += [sig(f"R{i}", "Robotics", lu=True) for i in range(6)]
    snap = eng.update("d", day, 0.0)
    assert snap.main_sector == "Semis" and snap.sector_sticky


def test_l2_is_capped_and_tier_reachable():
    eng = EmotionEngine()
    from src.sentiment import L2Parts
    assert L2Parts(4, 4, 2, 3).total() == 13          # max is 13, not 16
    assert L2Parts(4, 4, 2, 3).total() == int(13)
    # the top position tier is reachable on the real scale
    assert eng.position_advice(11.0, False)[:3] == (0.25, 0.90, "T5")
    assert eng.position_advice(8.0, False)[:3] == (0.20, 0.70, "T4")
    assert eng.position_advice(3.0, False)[:3] == (0.05, 0.10, "T2")
    assert eng.position_advice(1.0, False)[2].startswith("T1")


def test_no_leader_is_not_a_warning():
    """Regression: a dead market scored L3='warning' forever.

    With no leader, ``break_days`` used to increment every day because
    'leader in sealed_codes' was False. It must stay quiet instead.
    """
    eng = EmotionEngine()
    for i in range(5):
        snap = eng.update(f"d{i}", [], 0.0)
        assert snap.l3 == "none" and snap.leader_break_days == 0
        assert not snap.ebb_confirmed


def test_leader_break_escalates_to_confirm():
    state = State(prev_main_sector="Semis", prev_sector_hl=3,
                  leader_code="S0", leader_break_days=0)
    eng = EmotionEngine(state=state)
    broken = [sig(f"S{i}", "Semis", brk=True, board=4, turnover=9) for i in range(3)]

    day1 = eng.update("d1", broken, 0.0)
    assert day1.l3 == "warning" and day1.leader_break_days == 1
    day2 = eng.update("d2", broken, 0.0)
    assert day2.l3 == "confirm" and day2.leader_break_days == 2


def test_two_day_ebb_confirmation():
    state = State(prev_main_sector="Semis", prev_sector_hl=3, prev_l2=5.0,
                  leader_code="S0", leader_break_days=0, prev_hl_codes=[])
    eng = EmotionEngine(state=state)
    broken = [sig(f"S{i}", "Semis", brk=True, board=3) for i in range(3)]
    snap = eng.update("d1", broken, 0.0)
    assert snap.l2 <= 6.0
    assert snap.ebb_confirmed


def test_fry_rate_counts_broken_high_boards():
    """Regression: break-up bars had board_num 0, so the fry rate was always 0."""
    eng = EmotionEngine()
    hot = [sig(f"S{i}", "Semis", brk=True, board=5) for i in range(2)]
    hot += [sig(f"T{i}", "Semis", lu=True, board=5) for i in range(2)]
    snap = eng.update("d", hot, 0.0)
    assert abs(snap.fry_rate - 0.5) < 1e-9


def test_state_json_roundtrip():
    state = State(prev_main_sector="Semis", prev_l2=9.0,
                  fry_rate_seq=[0.1, 0.2], prev_hl_codes=["S1"])
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "state.json")
        state.save(path)
        loaded = State.load(path)
    assert loaded.prev_main_sector == "Semis"
    assert loaded.prev_l2 == 9.0
    assert loaded.fry_rate_seq == [0.1, 0.2]


def test_dynamic_params_by_regime():
    assert EmotionEngine.dynamic_params(3.0)[0] == "panic"
    assert EmotionEngine.dynamic_params(6.0)[0] == "neutral"
    assert EmotionEngine.dynamic_params(9.0)[0] == "euphoric"
    assert EmotionEngine.dynamic_params(3.0)[1]["stop_loss"] == "-3%"


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {name}: {exc}")
    print("OK" if not failures else f"{failures} failure(s)")
    sys.exit(1 if failures else 0)
