"""Unit tests for the bar-level primitives.  Run: python tests/test_indicators.py"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.indicators import (  # noqa: E402
    Bar,
    attempted_board_number,
    classify,
    count_consecutive_boards,
    is_break_up,
    is_high_face,
    is_limit_down,
    is_limit_up,
    is_sky_floor,
)


def bar(pre, close, high=None, low=None, open_=None, turn=5.0, code="A0"):
    return Bar(code=code, preclose=pre, open=pre if open_ is None else open_,
               high=close if high is None else high,
               low=close if low is None else low, close=close, turnover=turn,
               industry="Semis")


def test_limit_up_boundary():
    assert is_limit_up(bar(10.0, 10.95))          # +9.5%
    assert is_limit_up(bar(10.0, 11.00))          # +10%
    assert not is_limit_up(bar(10.0, 10.90))      # +9.0% is not the limit
    assert not is_limit_up(bar(0.0, 11.0))        # degenerate preclose


def test_one_word_board_filter():
    # open == high == close with almost no turnover: suspended / illiquid print
    assert not is_limit_up(bar(10.0, 11.0, open_=11.0, turn=0.3))
    # ...but a genuine one-word board with real turnover still counts
    assert is_limit_up(bar(10.0, 11.0, open_=11.0, turn=4.0))


def test_limit_down():
    assert is_limit_down(bar(10.0, 9.05))
    assert is_limit_down(bar(10.0, 9.00))
    assert not is_limit_down(bar(10.0, 9.20))


def test_break_up_requires_touching_the_limit():
    assert is_break_up(bar(10.0, 10.30, high=10.99))
    assert not is_break_up(bar(10.0, 10.30, high=10.50))   # never touched
    assert not is_break_up(bar(10.0, 11.00, high=11.00))   # sealed, not broken


def test_consecutive_boards():
    b = [bar(10.0, 11.0), bar(11.0, 12.1), bar(12.1, 13.31)]
    assert count_consecutive_boards(b) == 3
    assert count_consecutive_boards([]) == 0
    assert count_consecutive_boards(b + [bar(13.31, 13.0)]) == 0


def test_break_up_board_number_is_not_zero():
    """Regression: the notebook counted 0 for every break-up bar.

    Three sealed boards yesterday, a failed board today -> today's attempt is
    the 4th board, not 0.
    """
    hist = [bar(10.0, 11.0), bar(11.0, 12.1), bar(12.1, 13.31)]
    today = bar(13.31, 13.60, high=14.64, open_=14.0)   # high +9.99%, closed +2.2%
    assert is_break_up(today)
    assert count_consecutive_boards(hist + [today]) == 0   # the old, wrong number
    assert attempted_board_number(hist, today) == 4        # the fixed one


def test_high_face_needs_a_burst():
    hot = bar(10.0, 9.40, high=10.90)      # -6% close, +9% high, 15% pullback
    assert is_high_face(hot, burst_20d=6)
    assert not is_high_face(hot, burst_20d=2)   # not a hot stock


def test_sky_floor():
    sf = bar(10.0, 9.00, high=10.99, open_=10.99)
    assert is_sky_floor(sf, burst_20d=8)
    assert not is_sky_floor(sf, burst_20d=1)     # same print, cold stock


def test_classify_end_to_end():
    hist = [bar(10.0, 11.0)] * 3
    sig = classify(bar(13.31, 14.64, high=14.64, turn=8.0), hist)
    assert sig.limit_up and sig.board_num == 4
    assert not sig.break_up and not sig.limit_down
    assert not sig.high_face


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
