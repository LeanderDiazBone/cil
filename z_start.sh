#!/usr/bin/env bash
for p in 0,1 2,3 4,5 6,7; do
  (
    for i in {1..3}; do
      CUDA_VISIBLE_DEVICES=$p python run_overnight.py
    done
  ) &
done
wait
