#!/usr/bin/env bash
# Phase-Zero Modal suite runner. Fires every live test in order, tees real output
# to a timestamped log, prints a PASS/FAIL summary. This is what to run the INSTANT
# Modal auth lands. Each test fails LOUD; the runner never hides a failure.
#
# Usage:
#   bash phase-zero/modal/run_all.sh                  # T4 GPU default, cheap
#   MODAL_VERIFY_GPU=H100 bash phase-zero/modal/run_all.sh
#   MODAL_VERIFY_GPU=cpu  bash phase-zero/modal/run_all.sh   # mechanism-only, no GPU spend
set -uo pipefail
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="$R/.venv/bin/python"
MODAL="$R/.venv/bin/modal"; [ -x "$MODAL" ] || MODAL="$HOME/.local/bin/modal"
HERE="$R/phase-zero/modal"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="$R/phase-zero/logs/modal-run-${STAMP}.log"
GPU="${MODAL_VERIFY_GPU:-T4}"
mkdir -p "$R/phase-zero/logs"

echo "Modal Phase-Zero suite | gpu=$GPU | $STAMP | log=$LOG" | tee "$LOG"

# 0) Auth gate — abort early if not authed.
bash "$HERE/00_auth_check.sh" 2>&1 | tee -a "$LOG"
if [ "${PIPESTATUS[0]}" -ne 0 ]; then
  echo "ABORT: not authed. Stage stops here until ~/.modal.toml exists." | tee -a "$LOG"
  exit 1
fi

declare -a NAMES=(
  "01_sandbox_gpu (Sandbox.create gpu + exec nvidia-smi)"
  "02_modal_run_job (modal run @app.function gpu + retries)"
  "03_snapshot_fork (snapshot_filesystem -> fork N branches)"
  "04_volume_roundtrip (Volume commit/reload across containers)"
  "05_pool_autoscale (parallel sandbox pool speedup)"
  "06_function_autoscaler (.map -> multi-container, max_containers)"
)
declare -a CMDS=(
  "$PY $HERE/01_sandbox_gpu.py"
  "$MODAL run $HERE/02_modal_run_job.py"
  "$PY $HERE/03_snapshot_fork.py"
  "$MODAL run $HERE/04_volume_roundtrip.py"
  "$PY $HERE/05_pool_autoscale.py"
  "$MODAL run $HERE/06_function_autoscaler.py"
)
declare -a RESULTS=()

for idx in "${!CMDS[@]}"; do
  echo "" | tee -a "$LOG"
  echo "########## [$((idx+1))/6] ${NAMES[$idx]} ##########" | tee -a "$LOG"
  echo "+ ${CMDS[$idx]}" | tee -a "$LOG"
  if eval "${CMDS[$idx]}" 2>&1 | tee -a "$LOG"; then
    RESULTS[$idx]="PASS"
  else
    RESULTS[$idx]="FAIL"
  fi
done

echo "" | tee -a "$LOG"
echo "==================== SUMMARY ($STAMP, gpu=$GPU) ====================" | tee -a "$LOG"
fails=0
for idx in "${!NAMES[@]}"; do
  printf "  %-4s %s\n" "${RESULTS[$idx]}" "${NAMES[$idx]}" | tee -a "$LOG"
  [ "${RESULTS[$idx]}" = "FAIL" ] && fails=$((fails+1))
done
echo "  log: $LOG" | tee -a "$LOG"
if [ "$fails" -eq 0 ]; then
  echo "  ALL MODAL PHASE-ZERO TESTS PASSED — Modal proven LIVE." | tee -a "$LOG"
  exit 0
else
  echo "  $fails test(s) FAILED — Modal NOT fully verified. Read the log above." | tee -a "$LOG"
  exit 1
fi
