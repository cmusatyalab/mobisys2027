"""Power timeline of a Particle Tachyon (Qualcomm QCM6490), from its sysfs
power supplies.

    python3 tachyon_power.py [--interval 1] [--sample 0.1] > power.csv

Writes a CSV row to stdout every interval seconds, until stdin is closed
(or reaches EOF), Ctrl-C or SIGTERM, and then a last row for the partial
interval. The experiment runner runs it on a remote daemon over ssh, sending
it with python3 -c, and stops it by closing ssh's stdin; a dropped ssh
connection stops it the same way, so it never outlives the trial.

The Tachyon's battery gauge reports power_now and power_avg, but they don't
follow the load (they hardly change between idle and every core busy), so the
power is worked out from voltage and current instead:

    power = sum of V*I over the external supplies that are online (usb, wireless)
            - V*I into the battery while it is Charging
            + V*I out of the battery while it is Discharging

that is, what the board draws, whether from its input, its battery or both.
The input side is measured before the charger, so it includes the charger's
losses. Each row is the mean of the samples taken in its interval (one every
--sample seconds), as single readings are noisy (the input current reads in
5 mA steps). Readings go through the PMIC firmware and take a few ms each,
so sampling much faster than every 0.05 s adds load of its own.

Columns:
    t_sec           start of the interval, in seconds since the script started
    power_w         mean board power over the interval
    input_w         mean power from the external supplies
    battery_w       mean battery power, positive discharging, negative charging
    battery_status  the battery's status at the end of the interval
    samples         samples in the interval

Uses only the standard library, and runs on Python 3.8 (the Tachyon's
Ubuntu 20.04 python3).
"""

import argparse
import os
import select
import signal
import sys
import time

SUPPLIES = "/sys/class/power_supply"
# Supply types that feed the board from outside
INPUT_TYPES = ("USB", "USB_", "Mains", "Wireless")


def read(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return ""


def read_int(path):
    try:
        return int(read(path))
    except ValueError:
        return None


def find_supplies():
    """The external input supplies' directories, and the battery's (or None)."""
    inputs, battery = [], None
    for name in sorted(os.listdir(SUPPLIES)):
        d = os.path.join(SUPPLIES, name)
        kind = read(os.path.join(d, "type"))
        if kind == "Battery" and battery is None:
            battery = d
        elif kind.startswith(INPUT_TYPES) and os.path.exists(os.path.join(d, "current_now")):
            inputs.append(d)
    return inputs, battery


def sample(inputs, battery):
    """(board W, input W, battery W, battery status), battery W positive while
    it discharges."""
    input_w = 0.0
    for d in inputs:
        # A supply that is offline can still report a stale current
        if read_int(os.path.join(d, "online")) != 1:
            continue
        v = read_int(os.path.join(d, "voltage_now"))
        i = read_int(os.path.join(d, "current_now"))
        if v and i:
            input_w += abs(v * i) / 1e12  # uV * uA
    battery_w, status = 0.0, ""
    if battery is not None:
        status = read(os.path.join(battery, "status"))
        v = read_int(os.path.join(battery, "voltage_now"))
        i = read_int(os.path.join(battery, "current_now"))
        if v and i:
            p = abs(v * i) / 1e12
            # The sign convention of current_now differs between gauges, so
            # the direction comes from the status; with no input online the
            # battery can only be discharging
            if status == "Charging":
                battery_w = -p
            elif status == "Discharging" or input_w == 0.0:
                battery_w = p
    return input_w + battery_w, input_w, battery_w, status


def main():
    ap = argparse.ArgumentParser(description="Log a Particle Tachyon's power draw as CSV.")
    ap.add_argument("--interval", type=float, default=1.0, help="seconds per row (default: %(default)s)")
    ap.add_argument("--sample", type=float, default=0.1, help="seconds between samples (default: %(default)s)")
    ap.add_argument(
        "--no-stdin", action="store_true",
        help="don't stop when stdin closes (for running it by hand in the background)",
    )
    args = ap.parse_args()
    if args.interval <= 0 or args.sample <= 0 or args.sample > args.interval:
        ap.error("want 0 < --sample <= --interval")

    if not os.path.isdir(SUPPLIES):
        sys.exit("no %s; is this a Tachyon?" % SUPPLIES)
    inputs, battery = find_supplies()
    if not inputs and battery is None:
        sys.exit("no input or battery supplies in %s" % SUPPLIES)
    sys.stderr.write(
        "measuring inputs %s, battery %s\n"
        % (", ".join(os.path.basename(d) for d in inputs) or "none",
           os.path.basename(battery) if battery else "none")
    )
    sys.stderr.flush()

    stopping = []
    signal.signal(signal.SIGTERM, lambda *_: stopping.append(1))
    watch = [] if args.no_stdin else [sys.stdin]

    out = sys.stdout
    out.write("t_sec,power_w,input_w,battery_w,battery_status,samples\n")
    out.flush()
    t0 = time.monotonic()
    window_start = 0.0
    acc = [0.0, 0.0, 0.0]
    n, status = 0, ""

    def flush_row(start):
        if n:
            out.write("%.3f,%.3f,%.3f,%.3f,%s,%d\n" % (
                start, acc[0] / n, acc[1] / n, acc[2] / n, status, n,
            ))
            out.flush()

    try:
        next_sample = t0
        while not stopping:
            power, input_w, battery_w, status = sample(inputs, battery)
            acc[0] += power
            acc[1] += input_w
            acc[2] += battery_w
            n += 1
            now = time.monotonic()
            if now - t0 - window_start >= args.interval:
                flush_row(window_start)
                window_start += args.interval
                acc, n = [0.0, 0.0, 0.0], 0
            next_sample += args.sample
            if next_sample < now:  # fell behind: don't try to catch up
                next_sample = now
            wait = next_sample - time.monotonic()
            if watch:
                # Sleeps until the next sample, unless stdin closes first
                ready, _, _ = select.select(watch, [], [], max(wait, 0))
                if ready and not os.read(sys.stdin.fileno(), 4096):
                    break
            elif wait > 0:
                time.sleep(wait)
    except KeyboardInterrupt:
        pass
    except BrokenPipeError:
        return
    flush_row(window_start)


if __name__ == "__main__":
    main()
