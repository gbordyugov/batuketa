#!/bin/sh

# Fail the whole script if any single line fails
set -e

poetry run black --diff --check .
poetry run isort --check .
poetry run pylint -E batuketa tests scripts
