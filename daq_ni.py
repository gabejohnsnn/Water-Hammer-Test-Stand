"""
daq_ni.py
---------
NI cDAQ acquisition script for water hammer test stand.
Captures high-frequency PCB Piezotronics 113B03 pressure transducers (PT-3, PT-5)
at 50 kHz. Monitors a digital input for the shared solenoid valve trigger edge.

Channels
--------
Dev1/ai0  ->  PT-3  (Branch A, high-freq, PCB 113B03)
Dev1/ai1  ->  PT-5  (Branch B, high-freq, PCB 113B03)
Dev1/port0/line0  ->  Valve trigger input (shared TTL from SV-2 command)

Hardware note
-------------
PCB 113B03 outputs a charge signal requiring an ICP/charge amplifier before the
cDAQ analog input. If using an NI 9234 IEPE module, set IEPE excitation to 4 mA.
If using a separate charge amp, set input range to match amp output (typically ±5V).

Output
------
CSV file:  NI_[run_id]_[timestamp].csv
Header contains t_trigger_epoch for post-processing sync with LabJack data.

Usage
-----
    python daq_ni.py --run-id 0B_100psi_SV2_run1 --duration 1.0
    python daq_ni.py --run-id 0B_100psi_SV2_run1 --duration 1.0 --sample-rate 50000

Dependencies
------------
    pip install nidaqmx numpy
"""

import argparse
import time
import csv
import datetime
import threading
import numpy as np
import nidaqmx
from nidaqmx.constants import (
    AcquisitionType, TerminalConfiguration, Edge, WAIT_INFINITELY
)
from nidaqmx.stream_readers import AnalogMultiChannelReader

# ── Configuration ─────────────────────────────────────────────────────────────

AI_DEVICE       = "Dev1"
AI_CHANNELS     = ["ai0", "ai1"]               # PT-3, PT-5
CHANNEL_LABELS  = ["PT3_branchA_HF", "PT5_branchB_HF"]
DI_TRIGGER_LINE = f"{AI_DEVICE}/port0/line0"   # Digital trigger input
DEFAULT_RATE    = 50_000                        # Hz — appropriate for 113B03
DEFAULT_DURATION = 1.0                          # seconds post-trigger
PRE_TRIGGER_S   = 0.1                          # seconds of pre-trigger data

# Voltage-to-pressure calibration for PCB 113B03 via charge amp
# P_psi = V * GAIN + OFFSET  (update with actual charge amp gain setting)
# 113B03 sensitivity: ~10 mV/psi (with typical ICP amp at unity gain)
CALIBRATION = {
    "PT3_branchA_HF": {"gain": 100.0, "offset": 0.0},   # PLACEHOLDER — update
    "PT5_branchB_HF": {"gain": 100.0, "offset": 0.0},   # PLACEHOLDER — update
}

VOLTAGE_MIN = -5.0   # Input range minimum (V) — match charge amp output range
VOLTAGE_MAX =  5.0   # Input range maximum (V)

# ── Helpers ───────────────────────────────────────────────────────────────────

def voltage_to_pressure(label, voltage_array):
    cal = CALIBRATION[label]
    return voltage_array * cal["gain"] + cal["offset"]


class TriggerWatcher:
    """
    Runs in a background thread, polling the DI line for a rising edge.
    Sets self.triggered = True and records self.t_trigger_epoch on detection.
    """
    def __init__(self, di_line, poll_interval=0.0005):
        self.di_line = di_line
        self.poll_interval = poll_interval
        self.triggered = False
        self.t_trigger_epoch = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2.0)

    def _run(self):
        try:
            with nidaqmx.Task() as di_task:
                di_task.di_channels.add_di_chan(self.di_line)
                last_state = di_task.read()
                while not self._stop.is_set():
                    state = di_task.read()
                    if state and not last_state:
                        self.t_trigger_epoch = time.time()
                        self.triggered = True
                        print(f"\n  [Trigger] Rising edge at epoch {self.t_trigger_epoch:.6f}")
                        break
                    last_state = state
                    time.sleep(self.poll_interval)
        except Exception as e:
            print(f"  [TriggerWatcher] Error: {e}")


# ── Main Acquisition ──────────────────────────────────────────────────────────

def acquire(run_id, sample_rate, duration, output_dir="."):
    n_channels = len(AI_CHANNELS)
    total_samples = int(sample_rate * (PRE_TRIGGER_S + duration))
    pre_samples   = int(sample_rate * PRE_TRIGGER_S)
    post_samples  = int(sample_rate * duration)

    print(f"NI cDAQ acquisition: {n_channels} ch @ {sample_rate} Hz")
    print(f"  Pre-trigger: {pre_samples} samples ({PRE_TRIGGER_S} s)")
    print(f"  Post-trigger: {post_samples} samples ({duration} s)")

    # Start trigger watcher thread
    watcher = TriggerWatcher(DI_TRIGGER_LINE)
    watcher.start()
    print(f"  Trigger watcher armed on {DI_TRIGGER_LINE}")

    # Ring buffer for pre-trigger data
    ring = np.zeros((n_channels, pre_samples + 1))
    ring_idx = 0

    all_data = np.zeros((n_channels, total_samples))
    t_trigger_epoch = None
    t_trigger_sample = None

    try:
        with nidaqmx.Task() as task:
            # Add analog input channels
            for ch in AI_CHANNELS:
                task.ai_channels.add_ai_voltage_chan(
                    f"{AI_DEVICE}/{ch}",
                    terminal_config=TerminalConfiguration.RSE,
                    min_val=VOLTAGE_MIN,
                    max_val=VOLTAGE_MAX
                )

            # Configure sample clock — continuous, large buffer
            task.timing.cfg_samp_clk_timing(
                rate=sample_rate,
                active_edge=Edge.RISING,
                sample_mode=AcquisitionType.CONTINUOUS,
                samps_per_chan=total_samples * 2
            )

            reader = AnalogMultiChannelReader(task.in_stream)
            chunk_size = sample_rate // 100  # 10 ms chunks
            chunk_buf = np.zeros((n_channels, chunk_size))

            task.start()
            t_stream_start = time.time()
            print("  Streaming ...")

            collected = 0
            triggered_local = False

            # ── Phase 1: Pre-trigger ring buffer ──────────────────────────
            timeout = PRE_TRIGGER_S + 30.0  # 30 s trigger timeout
            while not watcher.triggered:
                if (time.time() - t_stream_start) > timeout:
                    print("  WARNING: Trigger timeout — saving data without sync.")
                    break
                try:
                    reader.read_many_sample(chunk_buf, chunk_size, timeout=1.0)
                    start = ring_idx % ring.shape[1]
                    end   = start + chunk_size
                    if end <= ring.shape[1]:
                        ring[:, start:end] = chunk_buf
                    else:
                        wrap = ring.shape[1] - start
                        ring[:, start:]    = chunk_buf[:, :wrap]
                        ring[:, :end % ring.shape[1]] = chunk_buf[:, wrap:]
                    ring_idx += chunk_size
                except Exception:
                    pass

            t_trigger_epoch  = watcher.t_trigger_epoch
            t_trigger_sample = min(ring_idx, pre_samples)

            # Unroll ring buffer into pre-trigger portion of all_data
            if ring_idx >= pre_samples:
                start = (ring_idx - pre_samples) % ring.shape[1]
                for i in range(pre_samples):
                    all_data[:, i] = ring[:, (start + i) % ring.shape[1]]
            else:
                all_data[:, :ring_idx] = ring[:, :ring_idx]

            collected = pre_samples

            # ── Phase 2: Post-trigger capture ─────────────────────────────
            print("  Post-trigger capture ...")
            while collected < total_samples:
                remaining = total_samples - collected
                read_n = min(chunk_size, remaining)
                buf = np.zeros((n_channels, read_n))
                reader.read_many_sample(buf, read_n, timeout=5.0)
                all_data[:, collected:collected + read_n] = buf
                collected += read_n

            task.stop()

    finally:
        watcher.stop()

    # ── Build time axis ───────────────────────────────────────────────────────
    dt = 1.0 / sample_rate
    t_axis = (np.arange(total_samples) - t_trigger_sample) * dt

    # Convert to pressure
    pressure_data = np.zeros_like(all_data)
    for i, label in enumerate(CHANNEL_LABELS):
        pressure_data[i, :] = voltage_to_pressure(label, all_data[i, :])

    # ── Write CSV ─────────────────────────────────────────────────────────────
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/NI_{run_id}_{ts}.csv"

    print(f"  Writing {total_samples} samples to {filename} ...")

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(["# WATER HAMMER TEST STAND — NI cDAQ Data"])
        writer.writerow(["# run_id",             run_id])
        writer.writerow(["# timestamp_utc",      datetime.datetime.utcnow().isoformat()])
        writer.writerow(["# sample_rate_hz",     str(sample_rate)])
        writer.writerow(["# n_samples",          str(total_samples)])
        writer.writerow(["# pre_trigger_samples",str(t_trigger_sample)])
        writer.writerow(["# triggered",          str(watcher.triggered)])
        writer.writerow(["# t_trigger_epoch",    f"{t_trigger_epoch:.6f}" if t_trigger_epoch else "None"])
        writer.writerow(["# channels",           ", ".join(CHANNEL_LABELS)])
        writer.writerow(["# calibration_note",   "Check CALIBRATION dict — placeholders may be active"])
        writer.writerow(["#"])

        writer.writerow(["t_s", "t_epoch"] +
                        [f"{lbl}_V" for lbl in CHANNEL_LABELS] +
                        [f"{lbl}_psi" for lbl in CHANNEL_LABELS])

        t_epoch_axis = (t_axis + t_trigger_epoch) if t_trigger_epoch else t_axis
        for i in range(total_samples):
            row = ([f"{t_axis[i]:.8f}", f"{t_epoch_axis[i]:.6f}"] +
                   [f"{all_data[j, i]:.6f}" for j in range(n_channels)] +
                   [f"{pressure_data[j, i]:.3f}" for j in range(n_channels)])
            writer.writerow(row)

    print(f"\nSaved: {filename}")
    print(f"  {total_samples} samples, {sample_rate} Hz, "
          f"trigger={'YES' if watcher.triggered else 'NO'}")
    return filename


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NI cDAQ water hammer acquisition")
    parser.add_argument("--run-id",      required=True, help="Run ID, e.g. 0B_100psi_SV2_run1")
    parser.add_argument("--duration",    type=float, default=DEFAULT_DURATION, help="Post-trigger duration (s)")
    parser.add_argument("--sample-rate", type=int,   default=DEFAULT_RATE,     help="Sample rate Hz (default 50000)")
    parser.add_argument("--output-dir",  default=".",                           help="Output directory")
    args = parser.parse_args()

    acquire(
        run_id=args.run_id,
        sample_rate=args.sample_rate,
        duration=args.duration,
        output_dir=args.output_dir
    )
