#!/usr/bin/env bash
# Run pytest test suite quickly. Usage: ./scripts/run_tests.sh [options]
# Options are passed to pytest.
set -e
PYTEST_ARGS="$@"
python -m pytest -q ${PYTEST_ARGS}
