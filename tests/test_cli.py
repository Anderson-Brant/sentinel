"""Smoke tests for the CLI. Doesn't exercise yfinance (that's an integration test)."""

from __future__ import annotations

from typer.testing import CliRunner

from sentinel import __version__
from sentinel.cli import app

runner = CliRunner()


def test_cli_version():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_cli_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "ingest" in result.stdout
    assert "features" in result.stdout
    assert "demo" in result.stdout


def test_cli_reddit_stub_exits_nonzero():
    result = runner.invoke(app, ["ingest", "reddit", "SPY"])
    assert result.exit_code != 0
