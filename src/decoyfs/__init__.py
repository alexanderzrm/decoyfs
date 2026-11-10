"""DecoyFS: decoy-referenced feature selection.

Quick start::

    from decoyfs import DecoySelector
    sel = DecoySelector().fit(X_train, y_train)
    X_small = sel.transform(X_test)
    sel.report_            # audit trail: votes and decoy comparison per feature
"""
from __future__ import annotations

from .compare import compare_selectors
from .decoy import DecoySelector
from .policies import list_policies, register_policy, unregister_policy
from .rankers import list_rankers, register_ranker, unregister_ranker
from .selectors import BorutaSelector, CVKSelector, FixedKSelector, NABFSSelector
from .validation import DecoyInputError

__version__ = "0.1.0"

__all__ = [
    "DecoySelector", "FixedKSelector", "CVKSelector", "BorutaSelector", "NABFSSelector", "compare_selectors",
    "register_ranker", "unregister_ranker", "list_rankers", "register_policy", "unregister_policy", "list_policies",
    "DecoyInputError", "__version__",
]
