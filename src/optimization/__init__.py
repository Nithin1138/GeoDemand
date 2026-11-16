"""
GeoDemand AI — Optimization Engine Package
"""

try:
    from src.optimization.time_windows import resolve_candidate_time_windows, TimeWindow
    from src.optimization.shift_optimizer import ShiftOptimizer
except ImportError:
    from optimization.time_windows import resolve_candidate_time_windows, TimeWindow
    from optimization.shift_optimizer import ShiftOptimizer

__all__ = ["ShiftOptimizer", "resolve_candidate_time_windows", "TimeWindow"]
