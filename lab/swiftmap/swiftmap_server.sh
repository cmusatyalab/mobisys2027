#!/bin/sh
# swiftmap_server.sh start|render|finish|done|stop
#
# The SwiftMap server's part in each trial, run from the experiment's hooks:
# start a fresh server before the flow (setup), tell whether a batch's NFN
# round is over (done), render the open batch as the grace period starts
# (grace), map the NFN round with what was sent back and wait for the worker
# to be idle (finish, at grace_end), then stop. The server's log is
# $EXPERIMENT_TRIAL_DIR/swiftmap_server.log, so it is kept with the trial.
set -u

SWIFTMAP_DIR=${SWIFTMAP_DIR:-/home/ubuntu/xianglic/swiftmap}
PORT=${SWIFTMAP_PORT:-43322}
CONTROL=${SWIFTMAP_CONTROL_PORT:-43323}
START_TIMEOUT=${SWIFTMAP_START_TIMEOUT:-300}
FINISH_TIMEOUT=${SWIFTMAP_FINISH_TIMEOUT:-300}
PIDFILE=/tmp/swiftmap-server.pid
LOG=${EXPERIMENT_TRIAL_DIR:-/tmp}/swiftmap_server.log
PYTHON=/usr/bin/python3

say() { echo "[swiftmap_server] $*"; }
listening() { ss -ltnH "sport = :$1" | grep -q .; }
running() { [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; }
control() { curl -sS -m 10 -X "$1" "http://127.0.0.1:$CONTROL$2"; }

stop() {
    if ! running; then
        rm -f "$PIDFILE"
        return 0
    fi
    pid=$(cat "$PIDFILE")
    say "stopping server $pid"
    for sig in INT TERM KILL; do
        kill -"$sig" "$pid" 2>/dev/null
        i=0
        while kill -0 "$pid" 2>/dev/null && [ "$i" -lt 30 ]; do
            sleep 1
            i=$((i + 1))
        done
        kill -0 "$pid" 2>/dev/null || break
        say "server $pid still up after SIG$sig"
    done
    rm -f "$PIDFILE"
}

start() {
    stop
    if listening "$PORT"; then
        say "port $PORT is taken by a server this script didn't start; stop it first"
        exit 1
    fi
    cd "$SWIFTMAP_DIR" || exit 1
    set -a
    [ -f .env ] && . ./.env
    set +a
    PYTHONUNBUFFERED=1 setsid nohup .venv/bin/python -c '
import runpy, signal, sys
signal.signal(signal.SIGINT, signal.default_int_handler)
sys.argv[0] = "launch_server.py"
runpy.run_path("launch_server.py", run_name="__main__")' \
        --port "$PORT" --control-port "$CONTROL" >"$LOG" 2>&1 </dev/null &
    echo $! >"$PIDFILE"
    say "started server $! (log $LOG); waiting for ports $PORT and $CONTROL"
    i=0
    until listening "$PORT" && listening "$CONTROL"; do
        if ! running; then
            say "server exited while starting:"
            tail -n 20 "$LOG"
            rm -f "$PIDFILE"
            exit 1
        fi
        if [ "$i" -ge "$START_TIMEOUT" ]; then
            say "server not up after ${START_TIMEOUT}s"
            stop
            exit 1
        fi
        sleep 1
        i=$((i + 1))
    done
    say "server up after ${i}s"
}

render() {
    say "render: $(control POST /render)"
}

idle() {
    control GET /status | "$PYTHON" -c '
import json, sys
s = json.load(sys.stdin)
sys.exit(0 if not s.get("processing") and not s.get("pending") else 1)'
}

done_mapping() {
    control GET /status 2>/dev/null | "$PYTHON" -c '
import json, sys
s = json.load(sys.stdin)
sys.exit(0 if s.get("completed", 0) >= 1 and not s.get("processing") and not s.get("pending") else 1)' 2>/dev/null
}

finish() {
    say "finish: $(control POST /finish)"
    i=0
    until idle; do
        if [ "$i" -ge "$FINISH_TIMEOUT" ]; then
            say "server still mapping after ${FINISH_TIMEOUT}s; stopping anyway"
            exit 1
        fi
        sleep 2
        i=$((i + 2))
    done
    say "server idle after ${i}s"
}

case "${1:-}" in
    start) start ;;
    render) render ;;
    finish) finish ;;
    done) done_mapping ;;
    stop) stop ;;
    *) echo "usage: $0 start|render|finish|done|stop" >&2; exit 2 ;;
esac
