#!/usr/bin/env bash
set -eu

dataroot=$1
exp=$2
gpu="${3:-0}"

for epoch in {5..400..5}
do
    python test.py --dataroot "$dataroot" --name "$exp" \
                --epoch "$epoch" \
                --direction "BtoA" \
                --gpu_ids "$gpu"
done
