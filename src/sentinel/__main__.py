"""Allow `python -m sentinel ...` as an alternative to the `sentinel` script."""

from sentinel.cli import app

if __name__ == "__main__":
    app()
