#!/bin/zsh
set -e    # exit on error
python scripts/convert_notebook_tests.py
# pytest kalepy
tox -p
