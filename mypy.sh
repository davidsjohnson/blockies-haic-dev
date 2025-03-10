#! /usr/bin/env bash
#
# this script runs flake8 on all clean files.

set -e
echo "Executing mypy..."



python -m mypy \
    blockies/ \
    test/ \
    setup.py  \
    "$@"
