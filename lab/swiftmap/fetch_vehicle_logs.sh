#!/bin/sh
# fetch_vehicle_logs.sh [user@host] [runs dir]
# Copies gs25456's drone, encoder and framestore logs (and the preprocess and
# reachback ones from before the rename) into <runs>/vehicle/ (asks for the
# password once), for stats.py --vehicle to slice per trial.
# The daemon truncates its logs each restart, so these are the copies that
# tail -F keeps appending to in ~/swiftmap_vehicle_logs/ on gs25456.
set -e
HOST=${1:-xianglic@100.80.166.79}
RUNS=${2:-$(dirname "$0")/runs/swiftmap}
mkdir -p "$RUNS/vehicle"
ssh "$HOST" 'cd ~/swiftmap_vehicle_logs && tar -cf - $(ls drone.log encoder.log framestore.log preprocess.log reachback.log 2>/dev/null)' \
    | tar -xf - -C "$RUNS/vehicle"
ls -la "$RUNS/vehicle"
