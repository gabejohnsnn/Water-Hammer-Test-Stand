"""
daq_labjack.py
--------------
LabJack T7 acquisition script for water hammer test stand.
Captures low-frequency pressure transducers (PT-1, PT-2, PT-4) at ~1 kHz.
Monitors a digital input for the solenoid valve trigger edge (shared with NI cDAQ).

Channels
--------
AIN0  ->  PT-1  (inlet, low-freq)
AIN1  ->  PT-2  (Branch A, low-freq)
AIN2  ->  PT-4  (Branch B, low-freq)
FIO0  ->  Valve trigger input (shared TTL from SV-2 command)

Output
------
CSV file:  LJ_[run_id]_[timestamp].csv
Header row contains metadata including t_trigger_epoch for post-processing sync.

Usage
-----
    python daq_labjack.py --run-id 0B_100psi_SV2_run1 --duration 5.0
    python daq_labjack.py --run-id 0B_100psi_SV2_run1 --duration 5.0 --sample-rate 1000

Dependencies
------------
    pip install labjack-ljm numpy
"""

import argparse
import time
import csv
import datetime
import numpy as np
from labjack import ljm

# ── Configuration ─────────────────────────────────────────────────────────────

CHANNELS       = ["AIN0", "AIN1", "AIN2"]   # PT-1, PT-2, PT-4
CHANNEL_LABELS = ["PT1_inlet", "PT2_branchA_LF", "PT4_branchB_LF"]
TRIGGER_DIO    = "FIO0"                       # Digital input for valve trigger
TRIGGER_DIO_NUM = 2000                        # LJM address for FIO0
DEFAULT_RATE   = 1000                         # Hz
DEFAULT_DURATION = 5.0                        # seconds
PRE_TRIGGER_S  = 0.5                          # seconds of data captured before trigger

# Voltage-to-pressure calibration (update with actual sensor calibration coefficients)
# P_psi = V * GAIN + OFFSET  (linear, per channel)
CALIBRATION = {
    "PT1_inlet":        {"gain": 100.0, "offset": 0.0},   # PLACEHOLDER — update
    "PT2_branchA_LF":   {"gain": 100.0, "offset": 0.0},   # PLACEHOLDER — update
    "PT4_branchB_LF":   {"gain": 100.0, "offset": 0.0},   # PLACEHOLDER — update
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def voltage_to_pressure(label, voltage_array):
    cal = CALIBRATION[label]
    return voltage_array * cal["gain"] + cal["offset"]


def setup_stream(handle, channels, scan_rate):
    """Configure T7 stream mode for the given analog channels."""
    # Set all AIN channels to ±10V range, single-ended (GND reference)
    for ch in channels:
        ch_num = int(ch.replace("AIN", ""))
        ljm.eWriteName(handle, f"AIN{ch_num}_RANGE", 10.0)
        ljm.eWriteName(handle, f"AIN{ch_num}_NEGATIVE_CH", 199)  # GND

    # Build address list for stream
    scan_list = ljm.namesToAddresses(len(channels), channels)[0]
    actual_rate = ljm.eStreamStart(
        handle,
        scans_per_read=int(scan_rate // 10),  # read in ~100 ms chunks
        num_addresses=len(channels),
        scan_list=scan_list,
        scan_rate=scan_rate
    )
    return actual_rate


def wait_for_trigger(handle, dio_name, timeout_s=30.0):
    """
    Busy-wait for rising edge on digital input.
    Returns (triggered: bool, epoch_time: float).
    Set timeout_s=0 to skip trigger waiting (free-run mode).
    """
    if timeout_s == 0:
        return False, time.time()

    print(f"  Waiting for trigger on {dio_name} (timeout {timeout_s}s) ...")
    last_state = int(ljm.eReadName(handle, dio_name))
    t_start = time.time()

    while (time.time() - t_start) < timeout_s:
        state = int(ljm.eReadName(handle, dio_name))
        if state == 1 and last_state == 0:
            t_trigger = time.time()
            print(f"  Trigger detected at epoch {t_trigger:.6f}")
            return True, t_trigger
        last_state = state
        time.sleep(0.0005)  # 0.5 ms polling interval

    print("  WARNING: Trigger timeout — saving data without trigger alignment.")
    return False, None


# ── Main Acquisition ──────────────────────────────────────────────────────────

def acquire(run_id, sample_rate, duration, output_dir="."):
    handle = ljm.openS("T7", "ANY", "ANY")
    info = ljm.getHandleInfo(handle)
    print(f"Opened LabJack T7 — Serial: {info[2]}")

    # Configure trigger digital input
    ljm.eWriteName(handle, TRIGGER_DIO, 0)  # ensure input mode

    n_samples = int(sample_rate * duration)
    n_channels = len(CHANNELS)

    # Circular pre-trigger buffer (PRE_TRIGGER_S worth of data)
    pre_buf_size = int(sample_rate * PRE_TRIGGER_S) * n_channels
    pre_buffer = np.zeros(pre_buf_size)
    pre_idx = 0
    pre_full = False

    print(f"Starting stream: {n_channels} channels @ {sample_rate} Hz for {duration} s")
    actual_rate = setup_stream(handle, CHANNELS, sample_rate)
    print(f"  Actual scan rate: {actual_rate:.1f} Hz")

    scans_per_read = int(actual_rate // 10)
    raw_data = []
    t_stream_start = time.time()
    triggered = False
    t_trigger_epoch = None
    t_trigger_sample = None
    total_scans = 0

    try:
        # ── Phase 1: Fill pre-trigger buffer while watching for trigger ──────
        print("  Phase 1: Pre-trigger buffering ...")
        while not triggered and (time.time() - t_stream_start) < (duration + PRE_TRIGGER_S):
            ret = ljm.eStreamRead(handle)
            chunk = np.array(ret[0]).reshape(-1, n_channels)

            # Check trigger on each scan in chunk
            for i, scan in enumerate(chunk):
                # Store in circular pre-trigger buffer
                flat = scan.flatten()
                for v in flat:
                    pre_buffer[pre_idx % pre_buf_size] = v
                    pre_idx += 1

            # Poll trigger (separate from stream — FIO0 not in stream)
            state = int(ljm.eReadName(handle, TRIGGER_DIO))
            if state == 1 and not triggered:
                triggered = True
                t_trigger_epoch = time.time()
                t_trigger_sample = total_scans + len(chunk)
                print(f"  Trigger at epoch {t_trigger_epoch:.6f} "
                      f"(scan ~{t_trigger_sample})")

            raw_data.append(chunk)
            total_scans += len(chunk)

            if triggered:
                break

        if not triggered:
            print("  WARNING: No trigger detected — continuing in free-run mode.")
            t_trigger_epoch = None

        # ── Phase 2: Capture post-trigger data ───────────────────────────────
        print("  Phase 2: Post-trigger capture ...")
        post_scans_needed = int(actual_rate * duration)
        post_scans = 0

        while post_scans < post_scans_needed:
            ret = ljm.eStreamRead(handle)
            chunk = np.array(ret[0]).reshape(-1, n_channels)
            raw_data.append(chunk)
            post_scans += len(chunk)

    finally:
        ljm.eStreamStop(handle)
        ljm.close(handle)
        print("  Stream stopped.")

    # ── Assemble full data array ──────────────────────────────────────────────
    all_data = np.vstack(raw_data)  # shape: (total_scans, n_channels)

    # Build time axis relative to trigger (or stream start if no trigger)
    dt = 1.0 / actual_rate
    n_total = len(all_data)
    if t_trigger_sample is not None:
        t_axis = (np.arange(n_total) - t_trigger_sample) * dt
    else:
        t_axis = np.arange(n_total) * dt

    # Convert voltages to pressure
    pressure_data = np.zeros_like(all_data)
    for i, label in enumerate(CHANNEL_LABELS):
        pressure_data[:, i] = voltage_to_pressure(label, all_data[:, i])

    # ── Write CSV ─────────────────────────────────────────────────────────────
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/LJ_{run_id}_{ts}.csv"

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)

        # Metadata header block
        writer.writerow(["# WATER HAMMER TEST STAND — LabJack T7 Data"])
        writer.writerow(["# run_id", run_id])
        writer.writerow(["# timestamp_utc", datetime.datetime.utcnow().isoformat()])
        writer.writerow(["# sample_rate_hz", f"{actual_rate:.2f}"])
        writer.writerow(["# n_samples", n_total])
        writer.writerow(["# triggered", str(triggered)])
        writer.writerow(["# t_trigger_epoch", f"{t_trigger_epoch:.6f}" if t_trigger_epoch else "None"])
        writer.writerow(["# t_trigger_sample_index", str(t_trigger_sample) if t_trigger_sample else "None"])
        writer.writerow(["# channels", ", ".join(CHANNEL_LABELS)])
        writer.writerow(["# calibration_note", "Check CALIBRATION dict — placeholders may be active"])
        writer.writerow(["#"])

        # Column headers
        writer.writerow(["t_s", "t_epoch"] +
                        [f"{lbl}_V" for lbl in CHANNEL_LABELS] +
                        [f"{lbl}_psi" for lbl in CHANNEL_LABELS])

        # Data rows
        t_epoch_axis = (t_axis + t_trigger_epoch) if t_trigger_epoch else t_axis
        for i in range(n_total):
            row = ([f"{t_axis[i]:.6f}", f"{t_epoch_axis[i]:.6f}"] +
                   [f"{all_data[i, j]:.5f}" for j in range(n_channels)] +
                   [f"{pressure_data[i, j]:.3f}" for j in range(n_channels)])
            writer.writerow(row)

    print(f"\nSaved: {filename}")
    print(f"  {n_total} scans, {actual_rate:.1f} Hz, trigger={'YES' if triggered else 'NO'}")
    return filename


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LabJack T7 water hammer acquisition")
    parser.add_argument("--run-id",     required=True, help="Run identifier, e.g. 0B_100psi_SV2_run1")
    parser.add_argument("--duration",   type=float, default=DEFAULT_DURATION, help="Post-trigger capture duration (s)")
    parser.add_argument("--sample-rate",type=int,   default=DEFAULT_RATE,     help="Sample rate in Hz (default 1000)")
    parser.add_argument("--output-dir", default=".",                           help="Directory for output CSV")
    args = parser.parse_args()

    acquire(
        run_id=args.run_id,
        sample_rate=args.sample_rate,
        duration=args.duration,
        output_dir=args.output_dir
    )
