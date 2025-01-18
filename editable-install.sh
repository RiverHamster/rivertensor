#!/usr/bin/bash
# This script will trigger a CMake rebuild on package import

pip install --no-build-isolation --config-settings=editable.rebuild=true -Cbuild-dir=build -ve.