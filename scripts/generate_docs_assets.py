#!/usr/bin/env python
# ruff: noqa: E402, I001
"""Generate plot images for the documentation site.

Usage:
    python scripts/generate_docs_assets.py [--output-dir DIR]

Produces docs/images/causal_impact_plot.png by running CausalImpact
with a fixed seed on synthetic data.
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from causal_impact import CausalImpact  # noqa: E402


def generate(output_dir: Path) -> Path:
    """Run CausalImpact on synthetic data and save the plot.

    Parameters
    ----------
    output_dir : Path
        Directory where causal_impact_plot.png will be written.

    Returns
    -------
    Path
        Full path to the generated PNG file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)

    n_pre = 100
    n_post = 30
    n = n_pre + n_post

    x = rng.normal(0, 1, size=n).cumsum() + 100
    y = 1.2 * x + rng.normal(0, 1, size=n)
    # Add an intervention effect in the post period
    y[n_pre:] += 5.0

    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    data = pd.DataFrame({"y": y, "x": x}, index=dates)

    pre_period = ["2020-01-01", dates[n_pre - 1].strftime("%Y-%m-%d")]
    post_period = [dates[n_pre].strftime("%Y-%m-%d"), dates[-1].strftime("%Y-%m-%d")]

    ci = CausalImpact(
        data, pre_period, post_period, model_args={"seed": 42, "niter": 1000}
    )

    fig = ci.plot()
    fig.set_size_inches(10, 8)

    out_path = output_dir / "causal_impact_plot.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Generated: {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "docs" / "images",
        help="Directory to write generated images (default: docs/images/)",
    )
    args = parser.parse_args()

    try:
        generate(args.output_dir)
    except Exception:
        sys.exit(1)


if __name__ == "__main__":
    main()
