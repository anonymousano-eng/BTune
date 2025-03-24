#!/bin/bash

set -e

SRC_DIR="./src"
ALGO_DIR="./third_party/OPPerTune/oppertune-algorithms/src/oppertune/algorithms"
OPPERTUNE_ALG_DIR="./third_party/OPPerTune/oppertune-algorithms"
OPPERTUNE_CORE_DIR="./third_party/OPPerTune/oppertune-cores"

echo "copy moe_ppo_tuner..."
rm -rf "${ALGO_DIR}/moe_ppo_tuner"
cp -r "${SRC_DIR}/moe_ppo_tuner" "${ALGO_DIR}/"

echo "replace all.py..."
cp "${SRC_DIR}/all.py" "${ALGO_DIR}/all.py"

echo "pip install -e oppertune-algorithms..."
pip install -e "${OPPERTUNE_ALG_DIR}"

echo "pip install -e oppertune-cores..."
pip install -e "${OPPERTUNE_CORE_DIR}"

echo "done！"
