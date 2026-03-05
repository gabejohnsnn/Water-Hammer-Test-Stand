"""
Acquisition script for NI cDAQ. Captures high-frequency pressure data (PT3, PT5) 
and waits for a digital trigger on PFI0/Line0.

Usage: python daq_ni.py --run-id TEST_RUN --duration 1.0
"""

import argparse
import time
import csv
import datetime
import threading
import numpy as np
import nidaqmx
from nidaqmx.constants import AcquisitionType, TerminalConfiguration, Edge
from nidaqmx.stream_readers import AnalogMultiChannelReader

# config
AI_DEVICE       = "Dev1"
AI_CHANNELS     = ["ai0", "ai1"]
LABELS          = ["PT3_branchA_HF", "PT5_branchB_HF"]
TRIGGER_LINE    = f"{AI_DEVICE}/port0/line0"
DEFAULT_RATE    = 50000
DEFAULT_DURATION = 1.0
PRE_TRIGGER_S   = 0.1

# Gain and offset for P_psi = V * gain + offset
CALIBRATION = {
    "PT3_branchA_HF": {"gain": 100.0, "offset": 0.0},
    "PT5_branchB_HF": {"gain": 100.0, "offset": 0.0},
}

VOLTAGE_RANGE = 5.0

class TriggerWatcher(threading.Thread):
    """Polls digital input for a rising edge."""
    def __init__(self, line):
        super().__init__(daemon=True)
        self.line = line
        self.triggered = False
        self.t_trigger = None
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()
        self.join(timeout=2.0)

    def run(self):
        try:
            with nidaqmx.Task() as task:
                task.di_channels.add_di_chan(self.line)
                last_state = task.read()
                while not self._stop.is_set():
                    state = task.read()
                    if state and not last_state:
                        self.t_trigger = time.time()
                        self.triggered = True
                        print(f"Trigger detected!")
                        break
                    last_state = state
                    time.sleep(0.0005)
        except Exception as e:
            print(f"Trigger error: {e}")

def acquire(run_id, rate, duration, output_dir="."):
    n_ch = len(AI_CHANNELS)
    pre_samples = int(rate * PRE_TRIGGER_S)
    post_samples = int(rate * duration)
    total_samples = pre_samples + post_samples

    print(f"NI cDAQ: {n_ch} ch @ {rate} Hz")
    
    watcher = TriggerWatcher(TRIGGER_LINE)
    watcher.start()

    # Data buffers
    ring = np.zeros((n_ch, pre_samples))
    ring_idx = 0
    all_data = np.zeros((n_ch, total_samples))

    try:
        with nidaqmx.Task() as task:
            for ch in AI_CHANNELS:
                task.ai_channels.add_ai_voltage_chan(
                    f"{AI_DEVICE}/{ch}",
                    terminal_config=TerminalConfiguration.RSE,
                    min_val=-VOLTAGE_RANGE, max_val=VOLTAGE_RANGE
                )

            task.timing.cfg_samp_clk_timing(
                rate=rate,
                sample_mode=AcquisitionType.CONTINUOUS,
                samps_per_chan=total_samples * 2
            )

            reader = AnalogMultiChannelReader(task.in_stream)
            
            chunk_size = rate // 100 # 10ms
            chunk = np.zeros((n_ch, chunk_size))
            
            task.start()
            t_start = time.time()
            print("Streaming...")

            # Phase 1: Pre-trigger ring buffer
            while not watcher.triggered:
                if (time.time() - t_start) > (PRE_TRIGGER_S + 30.0):
                    print("Trigger timeout.")
                    break
                reader.read_many_sample(chunk, chunk_size, timeout=1.0)
                
                # Fill ring buffer
                for i in range(chunk_size):
                    ring[:, (ring_idx + i) % pre_samples] = chunk[:, i]
                ring_idx = (ring_idx + chunk_size) % pre_samples

            # Unroll ring buffer
            for i in range(pre_samples):
                all_data[:, i] = ring[:, (ring_idx + i) % pre_samples]

            # Phase 2: Post-trigger capture
            print("Capturing post-trigger...")
            collected = pre_samples
            while collected < total_samples:
                rem = total_samples - collected
                n_read = min(chunk_size, rem)
                buf = np.zeros((n_ch, n_read))
                reader.read_many_sample(buf, n_read, timeout=5.0)
                all_data[:, collected:collected+n_read] = buf
                collected += n_read

            task.stop()

    finally:
        watcher.stop()

    # Process and save
    dt = 1.0 / rate
    t_axis = (np.arange(total_samples) - pre_samples) * dt
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"{output_dir}/NI_{run_id}_{ts}.csv"

    print(f"Saving to {fname}...")
    with open(fname, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["# NI cDAQ Run: " + run_id])
        writer.writerow(["# Triggered: " + str(watcher.triggered)])
        writer.writerow(["t_s", "t_epoch"] + [f"{l}_V" for l in LABELS] + [f"{l}_psi" for l in LABELS])

        for i in range(total_samples):
            v_row = all_data[:, i]
            p_row = [v * CALIBRATION[LABELS[j]]["gain"] + CALIBRATION[LABELS[j]]["offset"] for j, v in enumerate(v_row)]
            t_epoch = (watcher.t_trigger + t_axis[i]) if watcher.t_trigger else (t_start + t_axis[i])
            
            writer.writerow([f"{t_axis[i]:.8f}", f"{t_epoch:.6f}"] + 
                            [f"{v:.6f}" for v in v_row] + 
                            [f"{p:.3f}" for p in p_row])

    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id",      required=True)
    parser.add_argument("--duration",    type=float, default=DEFAULT_DURATION)
    parser.add_argument("--sample-rate", type=int,   default=DEFAULT_RATE)
    args = parser.parse_args()

    acquire(args.run_id, args.sample_rate, args.duration)
