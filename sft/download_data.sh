#!/bin/bash
set -euo pipefail

DATA_DIR="${1:-sft_data}"
mkdir -p "$DATA_DIR"

echo ">>> Downloading EB-Alfred trajectory dataset ..."
hf download \
    EmbodiedBench/EB-Alfred_trajectory_dataset \
    --repo-type dataset \
    --local-dir "$DATA_DIR/EB-Alfred_trajectory_dataset"

echo ">>> Extracting images ..."
cd "$DATA_DIR/EB-Alfred_trajectory_dataset"
if [ -f images.zip ] && [ ! -d images ]; then
    unzip -q images.zip -d images
fi
cd - > /dev/null

echo ">>> Done.  Dataset at: $DATA_DIR/EB-Alfred_trajectory_dataset"
