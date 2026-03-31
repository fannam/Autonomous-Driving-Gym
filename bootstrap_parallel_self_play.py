#!/usr/bin/env python3
"""Compatibility wrapper for the moved bootstrap utility."""

from tools.bootstrap_parallel_self_play import main


if __name__ == "__main__":
    raise SystemExit(main())
