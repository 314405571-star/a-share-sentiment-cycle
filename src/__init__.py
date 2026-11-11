"""A-share speculative sentiment cycle: limit-up breadth -> sector -> scoring.

Pure scoring core (no network). See ``scanner`` for the baostock adapter.
"""

from .indicators import (  # noqa: F401
    Bar,
    Flags,
    StockSignal,
    classify,
    count_consecutive_boards,
    is_break_up,
    is_high_face,
    is_limit_down,
    is_limit_up,
    is_sky_floor,
    to_signal,
)
from .sentiment import (  # noqa: F401
    Config,
    EmotionEngine,
    L2Parts,
    Snapshot,
    State,
    run_days,
)

__all__ = [
    "Bar", "Flags", "StockSignal", "classify", "count_consecutive_boards",
    "is_break_up", "is_high_face", "is_limit_down", "is_limit_up",
    "is_sky_floor", "to_signal", "Config", "EmotionEngine", "L2Parts",
    "Snapshot", "State", "run_days",
]
