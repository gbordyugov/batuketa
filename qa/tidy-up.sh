#!/bin/sh

poetry run black .
poetry run isort .
poetry run pylint -E batuketa tests scripts
