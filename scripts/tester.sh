#!/bin/zsh
set -e    # exit on error
python convert_notebook_tests.py
# pytest kalepy
tox -p
