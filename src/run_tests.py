from __future__ import annotations

import unittest


def main() -> None:
    suite = unittest.defaultTestLoader.discover(
        start_dir="src/tests",
        pattern="test_*.py",
    )
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
