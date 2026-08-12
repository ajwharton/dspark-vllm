#!/bin/bash
# Idempotent PIP applier. Dry-run by default; pass 'apply' to apply.
# Usage: ./apply.sh <dry-run|apply> <stable|main>
set -u
MODE="${1:-dry-run}"
BASE="${2:-main}"
HERE="$(cd "$(dirname "$0")" && pwd)"
# Dependency order mirrors PORT-PLAN milestones M2->M5.
PIPS=(200 201 300 400 500)
fail=0
for n in "${PIPS[@]}"; do
    f="$HERE/pip-${n}.patch"
    [ -f "$f" ] || { echo "PIP-$n: no patch file yet (spec only)"; continue; }
    if command -v git >/dev/null; then
        if git apply --check --verbose "$f" >/dev/null 2>&1; then
            echo "PIP-$n: applies cleanly"
            [ "$MODE" = "apply" ] && { git apply "$f" && echo "  applied"; } || true
        else
            echo "PIP-$n: does NOT apply cleanly on $BASE -> reroll (port conflict)"
            fail=1
        fi
    else
        echo "PIP-$n: git not installed; cannot verify"
    fi
done
echo "---"
[ "$fail" -eq 0 ] && echo "package OK on $BASE (dry-run)" || echo "package has conflicts on $BASE"
exit $fail
