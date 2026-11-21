#!/bin/bash
# wait until a given run finishes (or the driver dies), max 12 min
key="${1:-END   part2_4shot_seed0}"
timeout 700 bash -c "until grep -q '$key' /tmp/qzheng610_p2/logs/driver.log; do sleep 10; done"
echo "--- driver.log:"; cat /tmp/qzheng610_p2/logs/driver.log
echo "--- procs:"; ps -u qzheng610 -o pid,etime,cmd | grep run_anomalydino | grep -v grep | cut -c1-120
