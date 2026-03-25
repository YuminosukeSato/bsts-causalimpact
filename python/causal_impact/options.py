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
    expected_model_size: int = 2
    dynamic_regression: bool = False
    state_model: str = "local_level"
    prior_type: str = "spike_slab"
    nseasons: int | None = None
    season_duration: int | None = None

    def __post_init__(self) -> None:
        if self.prior_type not in {"spike_slab", "horseshoe"}:
            msg = (
                "prior_type must be 'spike_slab' or 'horseshoe', "
                f"got {self.prior_type!r}"
            )
            raise ValueError(msg)
        if self.prior_type == "horseshoe" and self.dynamic_regression:
            msg = (
                "horseshoe prior is not supported with dynamic_regression=True. "
                "Use prior_type='spike_slab' for time-varying coefficients."
            )
            raise ValueError(msg)
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
        if not isinstance(self.dynamic_regression, bool):
            msg = (
                "dynamic_regression must be a bool, "
                f"got {type(self.dynamic_regression).__name__}"
            )
            raise ValueError(msg)
        if self.state_model not in {"local_level", "local_linear_trend"}:
            msg = (
                "state_model must be one of "
                "{'local_level', 'local_linear_trend'}, "
                f"got {self.state_model}"
            )
            raise ValueError(msg)
        if self.nseasons is not None:
            if not isinstance(self.nseasons, int):
                msg = f"nseasons must be an integer, got {self.nseasons}"
                raise ValueError(msg)
            if self.nseasons < 1:
                msg = f"nseasons must be >= 1, got {self.nseasons}"
                raise ValueError(msg)
            if self.season_duration is None:
                object.__setattr__(self, "season_duration", 1)
        elif self.season_duration is not None:
            msg = "nseasons must be provided when season_duration is set"
            raise ValueError(msg)

        if self.season_duration is not None:
            if not isinstance(self.season_duration, int):
                msg = f"season_duration must be an integer, got {self.season_duration}"
                raise ValueError(msg)
            if self.season_duration < 1:
                msg = f"season_duration must be >= 1, got {self.season_duration}"
                raise ValueError(msg)

    def to_dict(self) -> dict:
        """Convert to dict for backward compatibility with dict-based model_args."""
        return asdict(self)
