"""
py_rollmatch — Fast rolling entry matching for staggered adoption.

Python reimplementation of the R rollmatch package using polars + numpy
for scalable matching on large panel datasets (100K+ units).
"""

from .core import rollmatch, alpha_sweep, RollmatchResult
from .reduce import reduce_data
from .score import score_data
from .balance import compute_balance, smd_table
from .diagnostics import balance_test, equivalence_test

__version__ = "0.1.0"
__all__ = [
    "rollmatch",
    "alpha_sweep",
    "RollmatchResult",
    "reduce_data",
    "score_data",
    "compute_balance",
    "smd_table",
    "balance_test",
    "equivalence_test",
]
