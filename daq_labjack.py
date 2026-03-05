"""
Acquisition script for the LabJack T7. Captures pressure transducer data and waits for a digital trigger on FIO0.

Usage: python daq_labjack.py --run-id TEST_RUN --duration 5.0
"""

import argparse
import time
import csv
import datetime
import numpy as np
from labjack import ljm

# config
CHANNELS = ["AIN0", "AIN1", "AIN2"]
LABELS   = ["PT1_inlet", "PT2_branchA_LF", "PT4_branchB_LF"]
TRIGGER_DIO = "FIO0"
DEFAULT_RATE = 1000
DEFAULT_DURATION = 5.0

# Gain and offset for P_psi = V * gain + offset
CALIBRATION = {
    "PT1_inlet":        {"gain": 100.0, "offset": 0.0},
    "PT2_branchA_LF":   {"gain": 100.0, "offset": 0.0},
    "PT4_branchB_LF":   {"gain": 100.0, "offset": 0.0},
}

def setup_stream(handle, rate):
    """Set ranges and start the stream."""
    for ch in CHANNELS:
        ljm.eWriteName(handle, f"{ch}_RANGE", 10.0)
        ljm.eWriteName(handle, f"{ch}_NEGATIVE_CH", 199) # GND

    addresses = ljm.namesToAddresses(len(CHANNELS), CHANNELS)[0]
    actual_rate = ljm.eStreamStart(
        handle, 
        scans_per_read=rate // 10, 
        num_addresses=len(CHANNELS),
        scan_list=addresses, 
        scan_rate=rate
    )
    return actual_rate

def acquire(run_id, rate, duration, output_dir="."):
    handle = ljm.openS("T7", "ANY", "ANY")
    print(f"Connected to LabJack T7")

    actual_rate = setup_stream(handle, rate)
    print(f"Streaming at {actual_rate:.1f} Hz...")

    data_chunks = []
    total_scans = 0
    triggered = False
    t_trigger_epoch = None
    t_trigger_sample = None
    t_start = time.time()

    try:
        # Loop until duration reached or manual stop
        while total_scans < (actual_rate * (duration + 1)):
            ret = ljm.eStreamRead(handle)
            chunk = np.array(ret[0]).reshape(-1, len(CHANNELS))
            data_chunks.append(chunk)
            total_scans += len(chunk)

            # Check for trigger if not yet found
            if not triggered:
                if ljm.eReadName(handle, TRIGGER_DIO) == 1:
                    triggered = True
                    t_trigger_epoch = time.time()
                    t_trigger_sample = total_scans
                    print(f"Trigger detected!")

            # If triggered, we only need 'duration' more seconds of data
            if triggered and (total_scans - t_trigger_sample) > (actual_rate * duration):
                break

    finally:
        ljm.eStreamStop(handle)
        ljm.close(handle)
        print("Stream stopped.")

    # Process data
    full_data = np.vstack(data_chunks)
    n_rows = len(full_data)
    
    # Create time axis relative to trigger
    dt = 1.0 / actual_rate
    if t_trigger_sample:
        t_axis = (np.arange(n_rows) - t_trigger_sample) * dt
    else:
        t_axis = np.arange(n_rows) * dt

    # Save to CSV
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"{output_dir}/LJ_{run_id}_{ts}.csv"
    
    with open(fname, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["# LabJack T7 Run: " + run_id])
        writer.writerow(["# Triggered: " + str(triggered)])
        writer.writerow(["t_s", "t_epoch"] + [f"{l}_V" for l in LABELS] + [f"{l}_psi" for l in LABELS])
        
        for i in range(n_rows):
            v_row = full_data[i]
            p_row = [v * CALIBRATION[LABELS[j]]["gain"] + CALIBRATION[LABELS[j]]["offset"] for j, v in enumerate(v_row)]
            t_epoch = (t_trigger_epoch + t_axis[i]) if t_trigger_epoch else (t_start + t_axis[i])
            
            writer.writerow([f"{t_axis[i]:.6f}", f"{t_epoch:.6f}"] + 
                            [f"{v:.4f}" for v in v_row] + 
                            [f"{p:.2f}" for p in p_row])

    print(f"Saved to {fname}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION)
    parser.add_argument("--rate", type=int, default=DEFAULT_RATE)
    args = parser.parse_args()

    acquire(args.run_id, args.rate, args.duration)
