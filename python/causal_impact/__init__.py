"""CausalImpact: Causal inference using Bayesian structural time series."""

from causal_impact._core import __version__
from causal_impact.main import CausalImpact
from causal_impact.options import ModelOptions
from causal_impact.selection import select_controls

__all__ = ["CausalImpact", "ModelOptions", "select_controls", "__version__"]
