"""
merge_sync.py
-------------
Post-processing script to synchronize and merge LabJack T7 and NI cDAQ CSV files
from a single water hammer test run.

Synchronization method
----------------------
Both DAQ scripts embed `t_trigger_epoch` (Unix time of trigger detection) in their
CSV headers. This script:
  1. Reads both CSVs and extracts t_trigger_epoch from each header.
  2. Computes the clock offset between the two systems (typically < 10 ms for
     software-triggered systems on the same machine; < 1 ms with shared hardware trigger).
  3. Re-expresses both time axes as t_s relative to the trigger event.
  4. Interpolates the low-rate LabJack data onto the high-rate NI timebase
     (or vice versa — configurable).
  5. Outputs a merged CSV and generates a quick-look plot.

Usage
-----
    python merge_sync.py --lj LJ_0B_100psi_SV2_run1_20260305.csv \
                         --ni NI_0B_100psi_SV2_run1_20260305.csv \
                         --output merged_0B_100psi_SV2_run1.csv

    # Keep NI timebase (default — preserves high-freq data resolution)
    python merge_sync.py --lj <lj_file> --ni <ni_file> --timebase ni

    # Downsample to LabJack rate (smaller file, still aligned)
    python merge_sync.py --lj <lj_file> --ni <ni_file> --timebase lj

    # Plot only, no merge output
    python merge_sync.py --lj <lj_file> --ni <ni_file> --plot-only

Dependencies
------------
    pip install numpy pandas scipy matplotlib
"""

import argparse
import csv
import os
import datetime
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── CSV Parsing ───────────────────────────────────────────────────────────────

def read_wh_csv(filepath):
    """
    Read a water hammer DAQ CSV (LJ or NI format).
    Returns:
        meta  : dict of header key-value pairs
        df    : pandas DataFrame with all data columns
    """
    meta = {}
    header_lines = 0

    with open(filepath, "r") as f:
        for line in f:
            if line.startswith("#"):
                line = line.lstrip("# ").strip()
                if "," in line:
                    parts = line.split(",", 1)
                    key = parts[0].strip()
                    val = parts[1].strip() if len(parts) > 1 else ""
                    if key:
                        meta[key] = val
                header_lines += 1
            else:
                break  # first non-comment line is the column header

    df = pd.read_csv(filepath, skiprows=header_lines, comment="#")
    df.columns = [c.strip() for c in df.columns]
    return meta, df


def parse_trigger_epoch(meta):
    val = meta.get("t_trigger_epoch", "None")
    if val == "None" or val == "":
        return None
    try:
        return float(val)
    except ValueError:
        return None


# ── Synchronization ───────────────────────────────────────────────────────────

def align_to_trigger(df, t_trigger_epoch, t_epoch_col="t_epoch"):
    """
    Re-express time axis relative to trigger epoch.
    Overwrites t_s column and returns the aligned df.
    """
    if t_trigger_epoch is None:
        print("  WARNING: No trigger epoch — using existing t_s column as-is.")
        return df

    if t_epoch_col in df.columns:
        df = df.copy()
        df["t_s"] = df[t_epoch_col] - t_trigger_epoch
    else:
        print(f"  WARNING: '{t_epoch_col}' column not found — cannot re-align.")
    return df


def compute_clock_offset(t_trigger_lj, t_trigger_ni):
    """
    Compute offset = t_trigger_ni - t_trigger_lj.
    Positive means NI trigger was detected later than LJ (NI clock is ahead
    or trigger detection latency is higher on NI side).
    """
    if t_trigger_lj is None or t_trigger_ni is None:
        return None
    return t_trigger_ni - t_trigger_lj


# ── Interpolation & Merge ─────────────────────────────────────────────────────

def get_pressure_cols(df):
    return [c for c in df.columns if c.endswith("_psi")]


def get_voltage_cols(df):
    return [c for c in df.columns if c.endswith("_V")]


def interpolate_to_timebase(source_df, target_t, data_cols, kind="linear"):
    """
    Interpolate source_df columns onto target_t array.
    Returns a dict of {col: interpolated_array}.
    """
    src_t = source_df["t_s"].values
    result = {}
    t_min, t_max = src_t.min(), src_t.max()
    # Clip target to source range to avoid extrapolation artifacts
    mask = (target_t >= t_min) & (target_t <= t_max)

    for col in data_cols:
        interp_fn = interp1d(src_t, source_df[col].values, kind=kind,
                             bounds_error=False, fill_value=np.nan)
        arr = np.full(len(target_t), np.nan)
        arr[mask] = interp_fn(target_t[mask])
        result[col] = arr
    return result


def merge_datasets(lj_df, ni_df, timebase="ni"):
    """
    Merge LJ and NI dataframes onto a common time axis.
    timebase: 'ni' (keep NI 50 kHz grid) or 'lj' (keep LJ 1 kHz grid)
    """
    if timebase == "ni":
        base_df  = ni_df
        interp_df = lj_df
        print(f"  Timebase: NI ({len(ni_df)} samples)")
    else:
        base_df  = lj_df
        interp_df = ni_df
        print(f"  Timebase: LabJack ({len(lj_df)} samples)")

    target_t = base_df["t_s"].values
    interp_cols = get_pressure_cols(interp_df) + get_voltage_cols(interp_df)
    interpolated = interpolate_to_timebase(interp_df, target_t, interp_cols)

    merged = base_df.copy()
    for col, arr in interpolated.items():
        merged[col] = arr

    # Sort columns: t_s, t_epoch, then all _V columns, then all _psi columns
    t_cols   = [c for c in merged.columns if c.startswith("t_")]
    v_cols   = sorted([c for c in merged.columns if c.endswith("_V")])
    psi_cols = sorted([c for c in merged.columns if c.endswith("_psi")])
    other    = [c for c in merged.columns if c not in t_cols + v_cols + psi_cols]
    merged   = merged[t_cols + other + v_cols + psi_cols]

    return merged


# ── Quick-look Plot ───────────────────────────────────────────────────────────

PLOT_COLORS = {
    "PT1_inlet_psi":        "#2196F3",
    "PT2_branchA_LF_psi":  "#4CAF50",
    "PT4_branchB_LF_psi":  "#FF9800",
    "PT3_branchA_HF_psi":  "#E91E63",
    "PT5_branchB_HF_psi":  "#9C27B0",
}

def quicklook_plot(merged_df, run_id, clock_offset, output_path=None):
    psi_cols = [c for c in merged_df.columns if c.endswith("_psi")]
    n = len(psi_cols)
    t = merged_df["t_s"].values

    fig = plt.figure(figsize=(14, 3 * n + 2))
    gs  = gridspec.GridSpec(n + 1, 1, height_ratios=[1] * n + [0.3], hspace=0.4)

    for i, col in enumerate(psi_cols):
        ax = fig.add_subplot(gs[i])
        color = PLOT_COLORS.get(col, f"C{i}")
        ax.plot(t * 1000, merged_df[col].values, color=color, linewidth=0.8, label=col)
        ax.axvline(0, color="red", linewidth=1.2, linestyle="--", label="t=0 (trigger)")
        ax.set_ylabel("Pressure (psi)", fontsize=9)
        ax.set_title(col.replace("_psi", ""), fontsize=10, loc="left")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)
        if i < n - 1:
            ax.set_xticklabels([])

    ax_info = fig.add_subplot(gs[-1])
    ax_info.axis("off")
    offset_str = (f"{clock_offset * 1000:.3f} ms" if clock_offset is not None
                  else "N/A (one or both systems lacked trigger epoch)")
    info = (f"Run: {run_id}   |   "
            f"NI–LJ clock offset: {offset_str}   |   "
            f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    ax_info.text(0.5, 0.5, info, ha="center", va="center",
                 fontsize=9, transform=ax_info.transAxes,
                 bbox=dict(boxstyle="round", facecolor="#f0f0f0", alpha=0.8))

    fig.suptitle(f"Water Hammer Quick-Look — {run_id}", fontsize=13, fontweight="bold")
    fig.text(0.5, 0.01, "Time relative to trigger (ms)", ha="center", fontsize=10)

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"  Plot saved: {output_path}")
    else:
        plt.show()
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def main(lj_file, ni_file, output_file, timebase, plot_only, plot_file):
    print(f"\n{'='*60}")
    print("Water Hammer DAQ Merge & Sync")
    print(f"  LJ file: {lj_file}")
    print(f"  NI file: {ni_file}")
    print(f"{'='*60}\n")

    # Load both files
    print("Loading LabJack data ...")
    lj_meta, lj_df = read_wh_csv(lj_file)
    print(f"  {len(lj_df)} samples, columns: {list(lj_df.columns)}")

    print("Loading NI cDAQ data ...")
    ni_meta, ni_df = read_wh_csv(ni_file)
    print(f"  {len(ni_df)} samples, columns: {list(ni_df.columns)}")

    # Extract trigger epochs
    t_trig_lj = parse_trigger_epoch(lj_meta)
    t_trig_ni = parse_trigger_epoch(ni_meta)
    print(f"\nTrigger epochs:")
    print(f"  LabJack: {t_trig_lj}")
    print(f"  NI cDAQ: {t_trig_ni}")

    clock_offset = compute_clock_offset(t_trig_lj, t_trig_ni)
    if clock_offset is not None:
        print(f"  Clock offset (NI - LJ): {clock_offset * 1000:.3f} ms")
        if abs(clock_offset) > 0.010:
            print(f"  WARNING: Offset > 10 ms — consider hardware trigger for tighter sync.")
    else:
        print("  WARNING: Cannot compute clock offset (missing trigger epoch in one or both files).")

    # Align both to trigger
    print("\nAligning time axes to trigger ...")
    lj_df = align_to_trigger(lj_df, t_trig_lj)
    ni_df = align_to_trigger(ni_df, t_trig_ni)

    run_id = lj_meta.get("run_id", os.path.basename(lj_file))

    if plot_only:
        # Use NI as base, interpolate LJ for plotting
        merged = merge_datasets(lj_df, ni_df, timebase="ni")
        quicklook_plot(merged, run_id, clock_offset,
                       output_path=plot_file)
        return

    # Merge
    print(f"\nMerging onto {timebase.upper()} timebase ...")
    merged = merge_datasets(lj_df, ni_df, timebase=timebase)
    print(f"  Merged shape: {merged.shape}")

    # Write merged CSV
    if not output_file:
        output_file = f"merged_{run_id}.csv"

    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["# WATER HAMMER TEST STAND — Merged DAQ Data"])
        writer.writerow(["# run_id",          run_id])
        writer.writerow(["# timestamp_utc",   datetime.datetime.utcnow().isoformat()])
        writer.writerow(["# lj_source",       os.path.basename(lj_file)])
        writer.writerow(["# ni_source",       os.path.basename(ni_file)])
        writer.writerow(["# timebase",        timebase.upper()])
        writer.writerow(["# t_trigger_epoch_lj", str(t_trig_lj)])
        writer.writerow(["# t_trigger_epoch_ni", str(t_trig_ni)])
        writer.writerow(["# clock_offset_ms", f"{clock_offset * 1000:.4f}" if clock_offset else "N/A"])
        writer.writerow(["# lj_sample_rate_hz", lj_meta.get("sample_rate_hz", "?")])
        writer.writerow(["# ni_sample_rate_hz", ni_meta.get("sample_rate_hz", "?")])
        writer.writerow(["#"])

    merged.to_csv(output_file, mode="a", index=False)
    print(f"\nMerged CSV saved: {output_file}")

    # Quick-look plot
    plot_out = plot_file or output_file.replace(".csv", ".png")
    print(f"Generating quick-look plot -> {plot_out}")
    quicklook_plot(merged, run_id, clock_offset, output_path=plot_out)

    print("\nDone.")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sync and merge LabJack + NI cDAQ water hammer CSV files"
    )
    parser.add_argument("--lj",        required=True, help="LabJack CSV file path")
    parser.add_argument("--ni",        required=True, help="NI cDAQ CSV file path")
    parser.add_argument("--output",    default=None,  help="Merged output CSV path")
    parser.add_argument("--timebase",  choices=["ni", "lj"], default="ni",
                        help="Which system's timebase to use for merged output (default: ni)")
    parser.add_argument("--plot-only", action="store_true",
                        help="Generate plot without writing merged CSV")
    parser.add_argument("--plot-file", default=None,
                        help="Save plot to this path instead of displaying")
    args = parser.parse_args()

    main(
        lj_file=args.lj,
        ni_file=args.ni,
        output_file=args.output,
        timebase=args.timebase,
        plot_only=args.plot_only,
        plot_file=args.plot_file
    )
