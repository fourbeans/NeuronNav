"""Run reference tests, preserving the raw test log and machine-readable result."""
import argparse
import io
import json
from pathlib import Path
import time
import unittest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    suite = unittest.defaultTestLoader.loadTestsFromName(
        "dendritic_nav.tests.test_core_dend"
    )
    stream = io.StringIO()
    start = time.time()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    (out / "test.log").write_text(stream.getvalue())
    summary = dict(
        tests=result.testsRun,
        errors=len(result.errors),
        failures=len(result.failures),
        success=result.wasSuccessful(),
        seconds=time.time() - start,
    )
    (out / "result.json").write_text(json.dumps(summary, indent=2))
    print(stream.getvalue())
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
