#!/bin/bash
# Build the kvpress overlay venv on Babel: the shared kvcache env's packages plus kvpress and its import-time deps.
set -e
P=$HOME/miniconda3/envs/kvcache/bin/python
[ -x $HOME/envs/kvpress_overlay/bin/python ] || $P -m venv --system-site-packages $HOME/envs/kvpress_overlay
$HOME/envs/kvpress_overlay/bin/pip install -q --no-deps kvpress==0.5.5 cachetools "fire==0.6.0" termcolor
# pytest for tests/test_window_rules.py; installed with its own small deps, nothing in the stack changes.
$HOME/envs/kvpress_overlay/bin/pip install -q pytest
