"""Daily-bar primitives that encode A-share limit-up semantics.

Everything here is a pure function of one bar plus its trailing history, so the
whole layer is unit-testable without a network connection.

Chinese terms used throughout the project:

* 涨停   limit-up   -- closed at the daily price limit
* 炸板   break-up   -- touched the limit intraday but closed below it
* 连板   boards     -- number of consecutive limit-up closes, today included
* 高位大面 high face -- a hot stock (>=5 limit-ups in 20 sessions) that gave back
                      >=8% from its intraday high
* 天地板 sky floor  -- opened/traded at the limit-up, closed at the limit-down
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

# --------------------------------------------------------------------------
# Thresholds (kept as module constants because they are market conventions,
# not tuning knobs; tune scoring in sentiment.Config instead)
# --------------------------------------------------------------------------

LIMIT_UP_MIN = 0.095          # +9.5% covers the 10% board after tick rounding
LIMIT_DOWN_MAX = -0.095
ONE_WORD_MAX_TURNOVER = 1.0   # a sealed one-word board with <1% turnover
BREAK_WICK_MIN = 0.095        # high must actually touch the limit
HIGH_FACE_PULLBACK = 0.08     # (high - close) / preclose
HIGH_FACE_BURST_MIN = 5       # limit-ups in the trailing window
HIGH_FACE_BURST_WINDOW = 20
SKY_FLOOR_UP = 0.09
SKY_FLOOR_DOWN = -0.09

# 10.95 / 10 - 1 == 0.094999... in binary floating point; an exact-limit close
# would otherwise fail the comparison. Real limit prices sit on this boundary
# all the time, so compare with a tolerance.
EPS = 1e-9


@dataclass
class Bar:
    """One daily bar. `preclose` is the previous close, already adjusted."""

    code: str
    preclose: float
    open: float
    high: float
    low: float
    close: float
    turnover: float = 0.0
    industry: Optional[str] = None
    name: str = ""
    date: str = ""          # YYYY-MM-DD, used by the scanner to reject stale bars

    def change(self) -> float:
        if self.preclose <= 0:
            return 0.0
        return self.close / self.preclose - 1.0

    def high_change(self) -> float:
        if self.preclose <= 0:
            return 0.0
        return self.high / self.preclose - 1.0

    def pullback(self) -> float:
        if self.preclose <= 0:
            return 0.0
        return (self.high - self.close) / self.preclose


@dataclass(frozen=True)
class Flags:
    limit_up: bool = False
    limit_down: bool = False
    break_up: bool = False
    high_face: bool = False
    sky_floor: bool = False
    board_num: int = 0

    @property
    def is_break(self) -> bool:
        return self.break_up


@dataclass
class StockSignal:
    """A bar after classification: what the scoring layer actually consumes."""

    code: str
    name: str
    industry: Optional[str]
    turnover: float
    close: float
    preclose: float
    flags: Flags = field(default_factory=Flags)

    @property
    def limit_up(self) -> bool:
        return self.flags.limit_up

    @property
    def break_up(self) -> bool:
        return self.flags.break_up

    @property
    def limit_down(self) -> bool:
        return self.flags.limit_down

    @property
    def high_face(self) -> bool:
        return self.flags.high_face

    @property
    def sky_floor(self) -> bool:
        return self.flags.sky_floor

    @property
    def board_num(self) -> int:
        return self.flags.board_num

    @property
    def is_break(self) -> bool:
        return self.flags.break_up


# --------------------------------------------------------------------------
# Primitives
# --------------------------------------------------------------------------


def is_limit_up(bar: Bar) -> bool:
    """Closed at the limit.

    The turnover guard removes the "one-word board" artefact: a print with
    open == high == close and almost no turnover is usually a suspended or
    illiquid name, not a real sealed board.
    """
    if bar.preclose <= 0:
        return False
    if bar.change() < LIMIT_UP_MIN - EPS:
        return False
    sealed = bar.open == bar.high == bar.close
    if sealed and bar.turnover < ONE_WORD_MAX_TURNOVER:
        return False
    return True


def is_limit_down(bar: Bar) -> bool:
    if bar.preclose <= 0:
        return False
    return bar.change() <= LIMIT_DOWN_MAX + EPS


def is_break_up(bar: Bar) -> bool:
    """Touched the limit but failed to close there."""
    if bar.preclose <= 0:
        return False
    return bar.high_change() >= BREAK_WICK_MIN - EPS and bar.close < bar.high


def count_consecutive_boards(bars: Sequence[Bar]) -> int:
    """Trailing run of limit-up closes, counting today if it is one.

    `bars` must be in ascending date order and end with the bar of interest.
    """
    n = 0
    for b in reversed(list(bars)):
        if is_limit_up(b):
            n += 1
        else:
            break
    return n


def count_limit_ups(bars: Sequence[Bar]) -> int:
    return sum(1 for b in bars if is_limit_up(b))


def attempted_board_number(history: Sequence[Bar], bar: Bar) -> int:
    """The board number of *today's attempt*, whether sealed or not.

    For a sealed limit-up this equals the usual 连板 count.

    For a break-up this is ``consecutive boards before today + 1`` -- which is
    what 炸板率 needs. The original notebook instead called
    ``count_consecutive_boards(history + [bar])``, which returns 0 for every
    break-up bar because today is not a limit-up. That made the set of
    "high-board stocks that broke" permanently empty, so the fry rate was
    always 0 and the fry component of L2 was always 4. See docs/算法总结.md.
    """
    if not (is_limit_up(bar) or is_break_up(bar)):
        return 0
    return count_consecutive_boards(history) + 1


def is_high_face(bar: Bar, burst_20d: int) -> bool:
    """Hot stock that gave back >= 8% from the intraday high."""
    if bar.preclose <= 0:
        return False
    return burst_20d >= HIGH_FACE_BURST_MIN and bar.pullback() >= HIGH_FACE_PULLBACK - EPS


def is_sky_floor(bar: Bar, burst_20d: int) -> bool:
    """天板 to 地板 within one session, on a stock that was already hot."""
    if not is_high_face(bar, burst_20d):
        return False
    return bar.change() <= SKY_FLOOR_DOWN and bar.high_change() >= SKY_FLOOR_UP


def classify(bar: Bar, history: Sequence[Bar] = ()) -> Flags:
    """Classify today's bar given the bars *before* it (ascending order).

    `history` is used for two things: the 20-day burst count behind the
    high-face rule, and the board streak behind today's attempted board.
    """
    history = list(history)
    burst = count_limit_ups(history[-HIGH_FACE_BURST_WINDOW:])
    lu = is_limit_up(bar)
    bu = (not lu) and is_break_up(bar)
    return Flags(
        limit_up=lu,
        limit_down=is_limit_down(bar),
        break_up=bu,
        high_face=is_high_face(bar, burst),
        sky_floor=is_sky_floor(bar, burst),
        board_num=attempted_board_number(history, bar),
    )


def to_signal(bar: Bar, history: Sequence[Bar] = ()) -> StockSignal:
    return StockSignal(
        code=bar.code,
        name=bar.name,
        industry=bar.industry,
        turnover=bar.turnover,
        close=bar.close,
        preclose=bar.preclose,
        flags=classify(bar, history),
    )
