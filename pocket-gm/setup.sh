#!/usr/bin/env bash
# Pocket GM — one-shot install script.
# Run: bash setup.sh
set -euo pipefail

GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
RESET='\033[0m'

echo -e "\n${BOLD}${CYAN}Pocket GM — Setup${RESET}\n"

# ── Python version check ────────────────────────────────────────────────────
if ! command -v python3 &>/dev/null; then
    echo -e "${YELLOW}Python 3 not found. Install Python 3.11+ and re-run.${RESET}"
    exit 1
fi

PY_VER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PY_MAJOR=$(python3 -c 'import sys; print(sys.version_info.major)')
PY_MINOR=$(python3 -c 'import sys; print(sys.version_info.minor)')

if [[ "$PY_MAJOR" -lt 3 || ( "$PY_MAJOR" -eq 3 && "$PY_MINOR" -lt 11 ) ]]; then
    echo -e "${YELLOW}Python 3.11+ required (found $PY_VER). Please upgrade.${RESET}"
    exit 1
fi

echo -e "  ${GREEN}✓${RESET} Python $PY_VER"

# ── Virtual env ─────────────────────────────────────────────────────────────
if [[ ! -d ".venv" ]]; then
    echo -e "  Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate
# shellcheck disable=SC1091
source .venv/bin/activate
echo -e "  ${GREEN}✓${RESET} Virtual environment active"

# ── Install ──────────────────────────────────────────────────────────────────
echo -e "  Installing Pocket GM and dependencies..."
pip install --quiet --upgrade pip
pip install --quiet -e ".[serve,web]"
echo -e "  ${GREEN}✓${RESET} Installed"

# ── Done ────────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${GREEN}Done!${RESET} Run the setup wizard:"
echo ""
echo -e "  ${CYAN}source .venv/bin/activate${RESET}"
echo -e "  ${CYAN}pocket-gm init${RESET}"
echo ""
