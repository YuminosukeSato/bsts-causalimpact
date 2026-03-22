"""Typed configuration for MCMC sampling parameters."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ModelOptions:
    """MCMC parameters for the Gibbs sampler.

    All defaults match R CausalImpact / bsts.
    """

    niter: int = 1000
    nwarmup: int = 500
    nchains: int = 1
    seed: int = 0
    standardize_data: bool = True
    prior_level_sd: float = 0.01
    expected_model_size: int = 1

    def __post_init__(self) -> None:
        if self.niter < 1:
            msg = f"niter must be >= 1, got {self.niter}"
            raise ValueError(msg)
        if self.nwarmup < 0:
            msg = f"nwarmup must be >= 0, got {self.nwarmup}"
            raise ValueError(msg)
        if self.nchains < 1:
            msg = f"nchains must be >= 1, got {self.nchains}"
            raise ValueError(msg)
        if self.prior_level_sd <= 0:
            msg = f"prior_level_sd must be > 0, got {self.prior_level_sd}"
            raise ValueError(msg)
        if self.expected_model_size <= 0:
            msg = f"expected_model_size must be > 0, got {self.expected_model_size}"
            raise ValueError(msg)

    def to_dict(self) -> dict:
        """Convert to dict for backward compatibility with dict-based model_args."""
        return asdict(self)
