#!/bin/bash
# ===========================================================================
# run_local.sh — Launch the app from local disk to avoid FUSE permission errors
#
# Problem: FUSE-mounted home directories on university clusters intermittently
#          deny read access (EPERM / Errno 1) to both the main process and
#          spawned child processes.  This affects .py source files AND compiled
#          .so extensions inside the virtualenv.
#
# Solution: Copy the project + virtualenv to node-local storage (/tmp) where
#           the filesystem is a real ext4/xfs mount, then run from there.
#           A symlink keeps experiment outputs on the persistent FUSE volume.
#
# Usage:
#   chmod +x run_local.sh
#   ./run_local.sh                    # uses default config
#   ./run_local.sh quick_test.yaml    # uses configs/quick_test.yaml
# ===========================================================================

set -euo pipefail

# --- Configuration --------------------------------------------------------
FUSE_PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"   # where this script lives
FUSE_VENV_DIR="$HOME/al_env"                         # your virtualenv on FUSE
LOCAL_BASE="/tmp/${USER}_al"                          # local scratch area
LOCAL_PROJECT="$LOCAL_BASE/active-learning"
LOCAL_VENV="$LOCAL_BASE/al_env"
# --------------------------------------------------------------------------

echo "=== Active Learning Local Launcher ==="
echo "FUSE project : $FUSE_PROJECT_DIR"
echo "Local copy   : $LOCAL_PROJECT"
echo ""

# 1. Sync project source to local disk (fast incremental copy)
echo "[1/4] Syncing project to local disk..."
mkdir -p "$LOCAL_PROJECT"
rsync -a --delete \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude 'experiments/' \
    --exclude 'state.json' \
    "$FUSE_PROJECT_DIR/" "$LOCAL_PROJECT/"

# 2. Sync virtualenv to local disk
echo "[2/4] Syncing virtualenv to local disk..."
mkdir -p "$LOCAL_VENV"
rsync -a --delete \
    "$FUSE_VENV_DIR/" "$LOCAL_VENV/"

# 3. Symlink experiment output back to FUSE (so results persist)
echo "[3/4] Linking experiment output to persistent storage..."
FUSE_EXPERIMENTS="$FUSE_PROJECT_DIR/experiments"
LOCAL_EXPERIMENTS="$LOCAL_PROJECT/experiments"
mkdir -p "$FUSE_EXPERIMENTS"
# Remove local experiments dir/link if it exists, then symlink
rm -rf "$LOCAL_EXPERIMENTS"
ln -s "$FUSE_EXPERIMENTS" "$LOCAL_EXPERIMENTS"

# Also symlink data directory to avoid copying large image files
LOCAL_DATA="$LOCAL_PROJECT/data"
rm -rf "$LOCAL_DATA"
ln -s "$FUSE_PROJECT_DIR/data" "$LOCAL_DATA"

# 4. Launch from local disk
echo "[4/4] Launching Streamlit from local disk..."
echo ""
export PATH="$LOCAL_VENV/bin:$PATH"
export VIRTUAL_ENV="$LOCAL_VENV"

cd "$LOCAL_PROJECT"
exec "$LOCAL_VENV/bin/streamlit" run app.py "$@"
