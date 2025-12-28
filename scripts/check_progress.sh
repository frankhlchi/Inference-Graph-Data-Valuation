#!/bin/bash
# Check progress of SVGL reproduction experiments

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_OUTPUT_BASE="${REPO_DIR}/outputs/reproduction"

OUTPUT_DIR="${1:-}"
LOG_FILE="${2:-}"

if [ -z "${OUTPUT_DIR}" ]; then
  OUTPUT_DIR="$(ls -td "${DEFAULT_OUTPUT_BASE}"/*/ 2>/dev/null | head -1 || true)"
fi

echo "=== SVGL Reproduction Progress ==="
echo "Repo dir: $REPO_DIR"
echo "Output dir: ${OUTPUT_DIR:-<none>}"
if [ -n "${LOG_FILE}" ]; then
  echo "Log file: $LOG_FILE"
fi
echo ""

# Check if process is running
if pgrep -f "scripts/run_parallel_experiments.py" > /dev/null; then
    echo "Status: RUNNING"
else
    echo "Status: NOT RUNNING"
fi
echo ""

# Count completed experiments
if [ -n "${OUTPUT_DIR}" ] && [ -d "${OUTPUT_DIR}" ]; then
    echo "=== Dataset Progress ==="
    for dataset in Cora Citeseer Pubmed CS Physics Computers Photo WikiCS chameleon squirrel Roman-empire Amazon-ratings; do
        if [ -d "$OUTPUT_DIR/$dataset" ]; then
            completed=$(ls -d $OUTPUT_DIR/$dataset/seed_*/val_samples 2>/dev/null | wc -l)
            samples=$(ls $OUTPUT_DIR/$dataset/seed_*/val_samples/*.pkl 2>/dev/null | wc -l)
            echo "$dataset: $completed seeds, $samples total samples"
        else
            echo "$dataset: not started"
        fi
    done
fi

echo ""
echo "=== Latest Log Entries ==="
if [ -n "${LOG_FILE}" ] && [ -f "${LOG_FILE}" ]; then
    tail -20 "$LOG_FILE"
else
    echo "No log file provided/found (pass it as the 2nd arg)."
fi

echo ""
echo "=== Results JSON ==="
if [ -f "$OUTPUT_DIR/all_results.json" ]; then
    echo "Results file exists. Completed experiments:"
    grep -o '\"status\": \"[^\"]*\"' "$OUTPUT_DIR/all_results.json" | sort | uniq -c
    echo ""
    echo "Latest results:"
    # Show last successful result
    python3 -c "
import json
with open('$OUTPUT_DIR/all_results.json') as f:
    results = json.load(f)
for r in results[-3:]:
    if r['status'] == 'success':
        print(f\"  {r['dataset']} seed={r['seed']}: val_acc={r['val_acc']:.4f}, val_corr={r.get('val_correlation', 'N/A'):.4f}, test_corr={r.get('test_correlation', 'N/A'):.4f}\")
    else:
        print(f\"  {r['dataset']} seed={r['seed']}: FAILED - {r.get('error', 'unknown')}\")
" 2>/dev/null
else
    echo "No results file yet"
fi
