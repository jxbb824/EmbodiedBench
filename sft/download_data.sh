#!/bin/bash
set -euo pipefail

DATA_DIR="${1:-sft_data}"
mkdir -p "$DATA_DIR"

echo ">>> Downloading EB-Alfred trajectory dataset ..."
hf download \
    EmbodiedBench/EB-Alfred_trajectory_dataset \
    --repo-type dataset \
    --local-dir "$DATA_DIR/EB-Alfred_trajectory_dataset"

echo ">>> Downloading EB-Nav trajectory dataset ..."
hf download \
    EmbodiedBench/EB-Nav_trajectory_dataset \
    --repo-type dataset \
    --local-dir "$DATA_DIR/EB-Nav_trajectory_dataset"

echo ">>> Extracting images ..."
for ds in EB-Alfred_trajectory_dataset EB-Nav_trajectory_dataset; do
    cd "$DATA_DIR/$ds"
    if [ -f images.zip ] && [ ! -d images ]; then
        echo "  Extracting $ds/images.zip ..."
        unzip -q images.zip -d images
    fi
    cd - > /dev/null
done

echo ">>> Done."
echo "  Alfred: $DATA_DIR/EB-Alfred_trajectory_dataset"
echo "  Nav:    $DATA_DIR/EB-Nav_trajectory_dataset"
