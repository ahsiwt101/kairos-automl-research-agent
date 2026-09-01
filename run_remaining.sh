#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")"
until grep -qE "saved runs/exp15|Traceback" runs/exp15.log 2>/dev/null; do sleep 30; done
echo "##### exp12: grading selection rules on the objective-axis pool #####"
./.venv/bin/python -u experiments/exp12_selection_analysis.py runs/exp15_selection_v2.json 2>&1 | grep -v Warning
echo; echo "##### exp16: is refitting on train+valid worth it? #####"
./.venv/bin/python -u experiments/exp16_refit.py 2>&1 | grep -v Warning
echo; echo "##### exp14: control-arm ablation (auditor on vs off) #####"
./.venv/bin/python -u experiments/exp14_control_arm.py 2>&1 | grep -v Warning
echo; echo "##### ALL DONE #####"
