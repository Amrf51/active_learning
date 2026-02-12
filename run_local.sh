#!/bin/bash
# ===========================================================================
# run_local.sh — Launch the app from local disk to avoid FUSE permission errors
#
# Copies project + virtualenv to /tmp (node-local storage), symlinks data/
# and experiments/ back to FUSE so results persist.
#
# Usage:
#   chmod +x run_local.sh
#   ./run_local.sh
# ===========================================================================

set -euo pipefail

# --- Configuration --------------------------------------------------------
FUSE_PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
FUSE_VENV_DIR="$HOME/al_env"
LOCAL_BASE="/tmp/${USER}_al"
LOCAL_PROJECT="$LOCAL_BASE/active-learning"
LOCAL_VENV="$LOCAL_BASE/al_env"
# --------------------------------------------------------------------------

echo "=== Active Learning Local Launcher ==="
echo "FUSE project : $FUSE_PROJECT_DIR"
echo "Local copy   : $LOCAL_PROJECT"
echo ""

# 1. Copy project source to local disk
echo "[1/4] Copying project to local disk..."
rm -rf "$LOCAL_PROJECT"
mkdir -p "$LOCAL_PROJECT"
cd "$FUSE_PROJECT_DIR"
# Copy everything except heavy/generated dirs
for item in *; do
    case "$item" in
        .git|__pycache__|experiments|data) continue ;;
        *) cp -r "$item" "$LOCAL_PROJECT/" ;;
    esac
done
# Copy hidden files (.gitignore, .kiro, etc.)
cp -r .gitignore "$LOCAL_PROJECT/" 2>/dev/null || true
cp -r .kiro "$LOCAL_PROJECT/" 2>/dev/null || true

# 2. Copy virtualenv to local disk
echo "[2/4] Copying virtualenv to local disk..."
if [ ! -d "$LOCAL_VENV" ]; then
    cp -r "$FUSE_VENV_DIR" "$LOCAL_VENV"
    echo "    (full copy — first run takes a moment)"
else
    echo "    (reusing existing local venv)"
fi

# 3. Symlink data + experiments back to FUSE
echo "[3/4] Linking data and experiments to persistent storage..."
mkdir -p "$FUSE_PROJECT_DIR/experiments"
ln -sfn "$FUSE_PROJECT_DIR/experiments" "$LOCAL_PROJECT/experiments"
ln -sfn "$FUSE_PROJECT_DIR/data" "$LOCAL_PROJECT/data"

# 4. Launch from local disk
echo "[4/4] Launching Streamlit from local disk..."
echo ""
export PATH="$LOCAL_VENV/bin:$PATH"
export VIRTUAL_ENV="$LOCAL_VENV"

cd "$LOCAL_PROJECT"
exec "$LOCAL_VENV/bin/streamlit" run app.py "$@"
