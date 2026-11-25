#!/usr/bin/env bash
# =============================================================================
# scripts/package_figshare_dataset.sh
# Creates the official compressed Figshare distribution archive
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_ARCHIVE="${REPO_ROOT}/vc_gat_figshare_dataset.tar.gz"

echo "=== PACKAGING FIGSHARE DATASET ==="
echo "Source:  $REPO_ROOT"
echo "Target:  $TARGET_ARCHIVE"

cd "$REPO_ROOT"

# Exclude temporary cache and OS metadata files
tar -czf "$TARGET_ARCHIVE" \
    --exclude="*.DS_Store" \
    --exclude="__pycache__" \
    --exclude="*.pyc" \
    README_DATASET.md \
    data/ \
    scene_conditioned_gold/ \
    legacy_archive/data_label_generation_pipeline/ \
    experiments/features/

echo "✓ Archive created successfully at: $TARGET_ARCHIVE"
ls -lh "$TARGET_ARCHIVE"
shasum -a 256 "$TARGET_ARCHIVE"
