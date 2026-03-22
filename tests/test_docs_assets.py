"""Tests for docs asset generation script."""

import os
import subprocess
import sys
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent
    / "scripts"
    / "generate_docs_assets.py"
)


def run_script(*, output_dir: str | None = None, env_override: dict | None = None):
    """Run generate_docs_assets.py as a subprocess and return CompletedProcess."""
    cmd = [sys.executable, str(SCRIPT_PATH)]
    if output_dir is not None:
        cmd.extend(["--output-dir", output_dir])
    env = os.environ.copy()
    if env_override:
        env.update(env_override)
    return subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=120)


class TestGenerateDocsAssets:
    """Normal-case tests for the docs asset generation script."""

    def test_generate_creates_output_dir(self, tmp_path):
        """Output directory is created when it does not exist."""
        out = tmp_path / "new_dir"
        assert not out.exists()
        result = run_script(output_dir=str(out))
        assert result.returncode == 0, result.stderr
        assert out.is_dir()

    def test_generate_creates_png_file(self, tmp_path):
        """Running the script produces a causal_impact_plot.png file."""
        result = run_script(output_dir=str(tmp_path))
        assert result.returncode == 0, result.stderr
        assert (tmp_path / "causal_impact_plot.png").exists()

    def test_generated_png_is_nonempty(self, tmp_path):
        """Generated PNG file has size > 0 bytes."""
        run_script(output_dir=str(tmp_path))
        png = tmp_path / "causal_impact_plot.png"
        assert png.stat().st_size > 0

    def test_generated_png_is_valid_image(self, tmp_path):
        """Generated file starts with the PNG magic bytes (\\x89PNG)."""
        run_script(output_dir=str(tmp_path))
        png = tmp_path / "causal_impact_plot.png"
        with open(png, "rb") as f:
            header = f.read(8)
        # PNG signature: 137 80 78 71 13 10 26 10
        assert header[:4] == b"\x89PNG"

    def test_generate_is_deterministic(self, tmp_path):
        """Two runs with the same seed produce files of the same size."""
        dir1 = tmp_path / "run1"
        dir2 = tmp_path / "run2"
        run_script(output_dir=str(dir1))
        run_script(output_dir=str(dir2))
        size1 = (dir1 / "causal_impact_plot.png").stat().st_size
        size2 = (dir2 / "causal_impact_plot.png").stat().st_size
        assert size1 == size2


class TestBoundaryConditions:
    """Boundary-value tests for the docs asset generation script."""

    def test_output_dir_already_exists(self, tmp_path):
        """Script succeeds when the output directory already exists."""
        out = tmp_path / "existing"
        out.mkdir()
        result = run_script(output_dir=str(out))
        assert result.returncode == 0, result.stderr
        assert (out / "causal_impact_plot.png").exists()

    def test_headless_no_display_error(self, tmp_path):
        """Script succeeds without DISPLAY env var (headless, Agg backend)."""
        env = {"DISPLAY": "", "MPLBACKEND": "Agg"}
        result = run_script(output_dir=str(tmp_path), env_override=env)
        assert result.returncode == 0, result.stderr

    def test_overwrite_existing_file(self, tmp_path):
        """Existing PNG is overwritten on re-run."""
        png = tmp_path / "causal_impact_plot.png"
        png.write_bytes(b"old content")
        result = run_script(output_dir=str(tmp_path))
        assert result.returncode == 0, result.stderr
        with open(png, "rb") as f:
            header = f.read(4)
        # After overwrite, the file must be a valid PNG, not "old content"
        assert header == b"\x89PNG"


class TestErrorCases:
    """Error-case tests for the docs asset generation script."""

    def test_invalid_output_dir_raises(self):
        """Passing a non-writable path causes a non-zero exit code."""
        result = run_script(output_dir="/proc/nonexistent/path")
        assert result.returncode != 0
