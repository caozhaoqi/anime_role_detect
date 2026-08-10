#!/bin/bash
# Launcher for v7 EfficientNet-B3 training, submitted as a launchd job so the
# process is truly detached (PPID=1, survives agent/session disconnect).
# caffeinate -isd keeps the Mac awake (idle+system+display) for the full run.
cd /Users/caozhaoqi/PycharmProjects/anime_role_detect || exit 1
exec /usr/bin/caffeinate -isd .venv/bin/python scripts/model_training/train_efficientnet_b3.py \
  --epochs 40 --batch-size 16 --image-size 256 \
  --split-dir data/splits/seed42 \
  --model-dir models/efficientnet_b3_v7 \
  --pretrained-weights models/pretrained/efficientnet_b3_imagenet.pth \
  --device mps --resume
