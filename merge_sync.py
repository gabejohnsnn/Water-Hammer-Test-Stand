"""
Synchronizes and merges LabJack T7 and NI cDAQ CSV files.
Aligns data using t_trigger_epoch and interpolates onto a common timebase.

Usage: python merge_sync.py --lj <lj.csv> --ni <ni.csv> --output <merged.csv>
"""

import argparse
import csv
import os
import datetime
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# config
PLOT_COLORS = {
    "PT1_inlet_psi":      "#2196F3",
    "PT2_branchA_LF_psi": "#4CAF50",
    "PT4_branchB_LF_psi": "#FF9800",
    "PT3_branchA_HF_psi": "#E91E63",
    "PT5_branchB_HF_psi": "#9C27B0",
}

def read_daq_csv(path):
    """Reads CSV and extracts metadata from headers."""
    meta = {}
    header_count = 0
    with open(path, "r") as f:
        for line in f:
            if line.startswith("#"):
                header_count += 1
                line = line.lstrip("# ").strip()
                if ":" in line: # Support both ':' and ',' as separators
                    key, val = [s.strip() for s in line.split(":", 1)]
                    meta[key] = val
                elif "," in line:
                    key, val = [s.strip() for s in line.split(",", 1)]
                    meta[key] = val
            else:
                break
    
    df = pd.read_csv(path, skiprows=header_count, comment="#")
    df.columns = [c.strip() for c in df.columns]
    
    # Extract trigger epoch
    t_trig = meta.get("t_trigger_epoch") or meta.get("Triggered at epoch")
    try:
        t_trig = float(t_trig) if t_trig and t_trig != "None" else None
    except:
        t_trig = None
        
    return meta, df, t_trig

def merge_data(lj_df, ni_df, t_trig_lj, t_trig_ni, timebase="ni"):
    """Aligns and merges datasets."""
    # Align time axes to trigger (t=0)
    for df, t_trig in [(lj_df, t_trig_lj), (ni_df, t_trig_ni)]:
        if t_trig:
            df["t_s"] = df["t_epoch"] - t_trig
            
    # Select base and source for interpolation
    if timebase == "ni":
        base, src = ni_df, lj_df
    else:
        base, src = lj_df, ni_df

    t_target = base["t_s"].values
    t_src = src["t_s"].values
    
    # Interpolate source columns onto base timebase
    data_cols = [c for c in src.columns if c.endswith(("_V", "_psi"))]
    merged = base.copy()
    
    for col in data_cols:
        fn = interp1d(t_src, src[col].values, bounds_error=False, fill_value=np.nan)
        merged[col] = fn(t_target)

    # Sort columns logically
    t_cols = [c for c in merged.columns if c.startswith("t_")]
    v_cols = sorted([c for c in merged.columns if c.endswith("_V")])
    p_cols = sorted([c for c in merged.columns if c.endswith("_psi")])
    return merged[t_cols + v_cols + p_cols]

def plot_merged(df, run_id, offset_ms, path=None):
    """Generates quick-look plot."""
    p_cols = [c for c in df.columns if c.endswith("_psi")]
    t_ms = df["t_s"].values * 1000
    
    fig, axes = plt.subplots(len(p_cols), 1, figsize=(12, 2.5 * len(p_cols) + 1), sharex=True)
    if len(p_cols) == 1: axes = [axes]
    
    for i, col in enumerate(p_cols):
        axes[i].plot(t_ms, df[col], color=PLOT_COLORS.get(col, f"C{i}"), linewidth=0.8)
        axes[i].axvline(0, color="red", linestyle="--", alpha=0.5)
        axes[i].set_ylabel("psi")
        axes[i].set_title(col.replace("_psi", ""), loc="left", fontsize=10)
        axes[i].grid(True, alpha=0.3)

    plt.xlabel("Time relative to trigger (ms)")
    plt.suptitle(f"Merge Results: {run_id} (offset: {offset_ms:.3f} ms)")
    plt.tight_layout()
    
    if path:
        plt.savefig(path, dpi=150)
        print(f"Plot saved: {path}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lj", required=True)
    parser.add_argument("--ni", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--timebase", choices=["ni", "lj"], default="ni")
    args = parser.parse_args()

    print(f"Merging {args.lj} and {args.ni}...")
    lj_meta, lj_df, t_trig_lj = read_daq_csv(args.lj)
    ni_meta, ni_df, t_trig_ni = read_daq_csv(args.ni)

    offset_ms = (t_trig_ni - t_trig_lj) * 1000 if (t_trig_ni and t_trig_lj) else 0
    print(f"Clock offset: {offset_ms:.3f} ms")

    merged = merge_data(lj_df, ni_df, t_trig_lj, t_trig_ni, args.timebase)
    run_id = lj_meta.get("run_id") or os.path.basename(args.lj)
    
    out_csv = args.output or f"merged_{run_id}.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["# WATER HAMMER MERGED DATA"])
        writer.writerow(["# run_id", run_id])
        writer.writerow(["# offset_ms", f"{offset_ms:.4f}"])
        writer.writerow(["# timestamp", datetime.datetime.now().isoformat()])
    
    merged.to_csv(out_csv, mode="a", index=False)
    print(f"Saved: {out_csv}")

    plot_merged(merged, run_id, offset_ms, out_csv.replace(".csv", ".png"))

if __name__ == "__main__":
    main()
