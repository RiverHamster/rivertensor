#!/bin/bash

set -e

cmake -S . -B build
cmake --build build
PYTHONPATH=build pytest test/
