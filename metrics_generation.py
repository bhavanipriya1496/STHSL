#!/usr/bin/env python3
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

EPOCH_RE = re.compile(r"_epoch_(\d+)_", re.IGNORECASE)

def parse_epoch(path: Path) -> int:
    m = EPOCH_RE.search(path.name)
    if not m:
        raise ValueError(f"Can't parse epoch from filename: {path.name}")
    return int(m.group(1))

def main():
    ap = argparse.ArgumentParser(
    description=(
        "Generate evaluation metrics by running test.py on all saved model\n"
        "checkpoints for a given dataset. Checkpoints are processed in\n"
        "ascending epoch order, and each epoch's output is written to a\n"
        "separate file (epoch_<N>_main.txt)."
    ),
    formatter_class=argparse.RawTextHelpFormatter,
    epilog=(
        "USAGE EXAMPLES:\n"
        "  # Generate metrics for all CHI checkpoints\n"
        "  python metrics_generation.py --data CHI --use_ode_option baseline\n\n"
        "  # Specify custom Save directory and output folder\n"
        "  python metrics_generation.py --data NYC --save-root ./Save --out-dir ./metrics\n\n"
        "  # Dry run: print evaluation commands without executing\n"
        "  python metrics_generation.py --data CHI --dry-run\n\n"
        "  # Use a specific Python executable (virtualenv / conda)\n"
        "  python metrics_generation.py --data SYN --python /usr/bin/python3\n\n"
        "EXPECTED CHECKPOINT NAMING:\n"
        "  <save-root>/<data>/*_epoch_<N>_*.pth\n"
        "  Example: Save/CHI/model_epoch_12_best.pth\n"
    ),
)
    ap.add_argument("--data", default="CHI", help="Dataset name (e.g., CHI, NYC, SYN)")
    ap.add_argument("--save-root", default="./Save", help="Root folder that contains <data>/ checkpoints")
    ap.add_argument("--out-dir", default=".", help="Where to write epoch_*_main.txt outputs")
    ap.add_argument("--use_ode_option", default="baseline", help="different model options: baseline, option1, option2")
    ap.add_argument("--pattern", default="*.pth", help="Checkpoint glob pattern (default: *.pth)")
    ap.add_argument("--python", default=sys.executable, help="Python executable to use (default: current)")
    ap.add_argument("--dry-run", action="store_true", help="Print commands without running")
    args = ap.parse_args()

    ckpt_dir = Path(args.save_root) / args.data
    if not ckpt_dir.is_dir():
        raise SystemExit(f"Checkpoint folder not found: {ckpt_dir.resolve()}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # collect checkpoints that match the file naming convention
    checkpoints = []
    for p in ckpt_dir.glob(args.pattern):
        if p.is_file() and EPOCH_RE.search(p.name):
            checkpoints.append(p)

    if not checkpoints:
        raise SystemExit(f"No matching checkpoints found in {ckpt_dir.resolve()} with pattern {args.pattern}")

    # sort by epoch number
    checkpoints.sort(key=parse_epoch)

    print(f"Found {len(checkpoints)} checkpoints in {ckpt_dir} (sorted by epoch).")

    for ckpt in checkpoints:
        epoch = parse_epoch(ckpt)
        out_file = out_dir / f"epoch_{epoch}_main.txt"

        cmd = [args.python, "test.py", "--data", args.data, "--use_ode_option", args.use_ode_option, "--checkpoint", str(ckpt)]
        print(f"\n[epoch {epoch}] Running: {' '.join(cmd)}")
        print(f"[epoch {epoch}] Writing: {out_file}")

        if args.dry_run:
            continue

        # run and redirect stdout+stderr into the same file
        with out_file.open("w", encoding="utf-8", errors="replace") as f:
            proc = subprocess.run(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=os.getcwd(),
            )

        if proc.returncode != 0:
            print(f"[epoch {epoch}] FAILED (return code {proc.returncode}). Check: {out_file}")
        else:
            print(f"[epoch {epoch}] OK")

if __name__ == "__main__":
    main()
