#!/usr/bin/env python3
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple

# Checkpoint names now include: _seed_<S>_epoch_<E>
SEED_EPOCH_RE = re.compile(r"_seed_(\d+)_epoch_(\d+)", re.IGNORECASE)
# Backward compatible: filenames that only have: _epoch_<E>
EPOCH_ONLY_RE = re.compile(r"_epoch_(\d+)", re.IGNORECASE)


def parse_seed_epoch(path: Path) -> Tuple[Optional[int], int]:
    """
    Returns (seed, epoch) parsed from filename.

    Accepts either:
      - *_seed_<S>_epoch_<E>*.pth   -> (S, E)
      - *_epoch_<E>*.pth           -> (None, E)  # backward compatible
    """
    m = SEED_EPOCH_RE.search(path.name)
    if m:
        return int(m.group(1)), int(m.group(2))

    m2 = EPOCH_ONLY_RE.search(path.name)
    if m2:
        return None, int(m2.group(1))

    raise ValueError(f"Can't parse seed/epoch from filename: {path.name}")


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Generate evaluation metrics by running test.py on saved model checkpoints.\n"
            "Supports per-seed checkpoint folders and writes per-epoch outputs.\n"
            "If a specific --seed is given, only that seed folder is processed.\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "USAGE EXAMPLES:\n"
            "  # Process ALL seeds (expects <save-root>/<data>/seed_<S>/*.pth)\n"
            "  python metrics_generation.py --data CHI --use_ode_option baseline\n\n"
            "  # Process ONLY seed 42\n"
            "  python metrics_generation.py --data CHI --use_ode_option option2 --seed 42\n\n"
            "  # Custom Save directory and output folder\n"
            "  python metrics_generation.py --data NYC --save-root ./Save --out-dir ./metrics\n\n"
            "  # Dry run\n"
            "  python metrics_generation.py --data CHI --dry-run\n\n"
            "EXPECTED CHECKPOINT LAYOUT (new):\n"
            "  <save-root>/<data>/seed_<S>/*_seed_<S>_epoch_<E>*.pth\n"
            "  Example: Save/CHI/seed_0/_seed_0_epoch_12_MicroF1_0.1234.pth\n\n"
            "BACKWARD COMPAT (old):\n"
            "  <save-root>/<data>/*_epoch_<E>*.pth\n"
        ),
    )

    ap.add_argument("--data", default="CHI", help="Dataset name (e.g., CHI, NYC, SYN)")
    ap.add_argument("--save-root", default="./Save", help="Root folder that contains <data>/ checkpoints")
    ap.add_argument("--out-dir", default=".", help="Where to write metrics outputs")
    ap.add_argument("--use_ode_option", default="baseline", help="different model options: baseline, option1, option2")
    ap.add_argument("--pattern", default="*.pth", help="Checkpoint glob pattern (default: *.pth)")
    ap.add_argument("--python", default=sys.executable, help="Python executable to use (default: current)")
    ap.add_argument("--dry-run", action="store_true", help="Print commands without running")

    # NEW: seed selection (optional)
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help="If set, only process checkpoints under <data>/seed_<seed>/",
    )

    args = ap.parse_args()

    data_dir = Path(args.save_root) / args.data
    if not data_dir.is_dir():
        raise SystemExit(f"Checkpoint folder not found: {data_dir.resolve()}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine which directories to scan:
    # - If --seed specified -> only that seed folder
    # - Else -> prefer seed_* folders if present; otherwise fallback to old flat layout
    seed_dirs = []
    if args.seed is not None:
        sd = data_dir / f"seed_{args.seed}"
        if not sd.is_dir():
            raise SystemExit(f"Seed folder not found: {sd.resolve()}")
        seed_dirs = [sd]
    else:
        seed_dirs = sorted([p for p in data_dir.glob("seed_*") if p.is_dir()])

    # If no seed_* folders exist, fallback to old layout (flat in <data>/)
    scan_dirs = seed_dirs if seed_dirs else [data_dir]

    all_checkpoints = []
    for d in scan_dirs:
        for p in d.glob(args.pattern):
            if not p.is_file():
                continue
            # must have at least epoch parseable
            try:
                _seed, _epoch = parse_seed_epoch(p)
                all_checkpoints.append(p)
            except ValueError:
                continue

    if not all_checkpoints:
        raise SystemExit(
            f"No matching checkpoints found under {data_dir.resolve()} "
            f"(scan dirs: {[str(d) for d in scan_dirs]}) with pattern {args.pattern}"
        )

    # Sort by (seed, epoch) where seed None comes last
    def sort_key(p: Path):
        seed, epoch = parse_seed_epoch(p)
        seed_key = seed if seed is not None else 10**9
        return (seed_key, epoch)

    all_checkpoints.sort(key=sort_key)

    print(
        f"Found {len(all_checkpoints)} checkpoints under {data_dir} "
        f"({'seed folders' if seed_dirs else 'flat layout'}) (sorted by seed, epoch)."
    )

    for ckpt in all_checkpoints:
        seed, epoch = parse_seed_epoch(ckpt)

        # Output filename includes seed if available
        if seed is None:
            out_file = out_dir / f"epoch_{epoch}_main.txt"
            tag = f"epoch {epoch}"
        else:
            out_file = out_dir / f"seed_{seed}_epoch_{epoch}_main.txt"
            tag = f"seed {seed} epoch {epoch}"

        cmd = [
            args.python,
            "test.py",
            "--data",
            args.data,
            "--use_ode_option",
            args.use_ode_option,
            "--checkpoint",
            str(ckpt),
        ]

        print(f"\n[{tag}] Running: {' '.join(cmd)}")
        print(f"[{tag}] Writing: {out_file}")

        if args.dry_run:
            continue

        with out_file.open("w", encoding="utf-8", errors="replace") as f:
            proc = subprocess.run(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=os.getcwd(),
            )

        if proc.returncode != 0:
            print(f"[{tag}] FAILED (return code {proc.returncode}). Check: {out_file}")
        else:
            print(f"[{tag}] OK")


if __name__ == "__main__":
    main()