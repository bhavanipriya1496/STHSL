#!/usr/bin/env python3
"""
tune_STHSL.py — Random/Grid tuning for STHSL baseline/option1/option2 WITHOUT Ray.

What this script does:
- Runs multiple trials with different hyperparameters (and ODE params for option1/option2)
- Ensures *trial isolation* by running each trial in a separate OS process (multiprocessing spawn)
- Uses a trial-unique args.save directory to prevent checkpoint collisions
- Writes:
    - config.json per trial
    - result.json per trial (when not dry-run)
    - best.json per trial (when not dry-run and an epoch improves objective)
    - BEST.json at experiment root (overall best; when not dry-run)
    - summary.csv at experiment root (when not dry-run)

Dry-run mode:
- Samples configs, creates trial folders, writes config.json
- Does NOT build trainer/model, load data, touch GPU, or call train/eval
- Does NOT write result.json/best.json/summary.csv/BEST.json

Usage examples:
  # Real tuning (random search)
  python tune_STHSL.py --data NYC --arch option2 --num-trials 30 --max-epochs 5 --workers 2

  # Dry run (just generate configs + folders)
  python tune_STHSL.py --data NYC --arch option2 --num-trials 10 --dry-run

Notes:
- Objective minimized:
    rmse | mae | mape | rmse_mae (default rmse_mae = RMSE + MAE)
- workers controls parallelism. workers=1 runs sequentially.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import multiprocessing as mp

# ------------------------------------------------------------
# IMPORTANT: Your codebase parses argv at import-time in Params.
# We sanitize argv before importing Params, then restore it.
# ------------------------------------------------------------
_ORIG_ARGV = sys.argv.copy()
sys.argv = [sys.argv[0]]
from Params import args  # noqa: E402
sys.argv = _ORIG_ARGV

from engine import trainer  # noqa: E402
from utils import seed_torch  # noqa: E402


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float(str(x))


def _set_args_from_config(cfg: Dict[str, Any]) -> None:
    """Apply sampled config to Params.args (only if that attribute exists)."""
    for k, v in cfg.items():
        if hasattr(args, k):
            setattr(args, k, v)

    # Normalize common numeric strings if any
    if hasattr(args, "odeint_atol"):
        args.odeint_atol = _safe_float(getattr(args, "odeint_atol"))
    if hasattr(args, "odeint_rtol"):
        args.odeint_rtol = _safe_float(getattr(args, "odeint_rtol"))


def _objective(res_eval: Dict[str, float], mode: str) -> float:
    """Return scalar to minimize."""
    if mode == "rmse":
        return float(res_eval.get("RMSE", 1e18))
    if mode == "mae":
        return float(res_eval.get("MAE", 1e18))
    if mode == "mape":
        return float(res_eval.get("MAPE", 1e18))
    # default: rmse_mae
    return float(res_eval.get("RMSE", 1e18) + res_eval.get("MAE", 1e18))


def _cleanup_torch() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# -----------------------------
# Search space (Ray-free)
# -----------------------------
@dataclass(frozen=True)
class Space:
    # Each entry: ("type", params...)
    # types: "choice", "uniform", "loguniform"
    spec: Dict[str, Tuple[str, Any, Any]]


def _sample(space: Space, rng: random.Random) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, (typ, a, b) in space.spec.items():
        if typ == "choice":
            out[k] = rng.choice(list(a))
        elif typ == "uniform":
            out[k] = float(a) + (float(b) - float(a)) * rng.random()
        elif typ == "loguniform":
            lo, hi = float(a), float(b)
            if lo <= 0 or hi <= 0:
                raise ValueError(f"loguniform requires positive bounds for {k}: {lo}, {hi}")
            x = math.log(lo) + (math.log(hi) - math.log(lo)) * rng.random()
            out[k] = float(math.exp(x))
        else:
            raise ValueError(f"Unknown space type: {typ} for key={k}")
    return out


def build_space(arch: str, cpu_only: bool) -> Space:
    """
    Define your search space here.

    IMPORTANT: This is a generic example space. Ideally you should align keys exactly
    to your Params/model.py expectations (the same thing you did for Ray).
    """
    device_choices = ["cpu"] if cpu_only else (["cuda"] if torch.cuda.is_available() else ["cpu"])

    spec: Dict[str, Tuple[str, Any, Any]] = {
        "use_ode_option": ("choice", [arch], None),
        "device": ("choice", device_choices, None),

        "lr": ("loguniform", 1e-4, 5e-3),
        "weight_decay": ("loguniform", 1e-6, 1e-3),

        "batch": ("choice", [8, 16, 32], None),
        "latdim": ("choice", [8, 16, 32, 64], None),

        "dropRateL": ("uniform", 0.0, 0.4),
        "dropRateG": ("uniform", 0.0, 0.3),

        # You mentioned engine.py uses cr and ir, keep them searchable.
        "cr": ("uniform", 0.4, 1.0),
        "ir": ("uniform", 0.5, 2.0),
        "t": ("loguniform", 0.02, 0.2),

        "kernelSize": ("choice", [3, 5], None),
        "hyperNum": ("choice", [64, 128, 256], None),
    }

    if arch in ("option1", "option2"):
        spec.update(
            {
                "n_traj_samples": ("choice", [1, 2, 4], None),
                "ode_method": ("choice", ["euler", "rk4", "dopri5"], None),
                "odeint_atol": ("loguniform", 1e-6, 1e-3),
                "odeint_rtol": ("loguniform", 1e-6, 1e-3),

                "gen_layers": ("choice", [1, 2, 3], None),
                "gen_dim": ("choice", [32, 64, 128], None),

                "gcn_step": ("choice", [1, 2, 3], None),
                "horizon": ("choice", [6, 12, 24], None),
            }
        )

    return Space(spec=spec)


# -----------------------------
# Trial runner (runs INSIDE each subprocess)
# -----------------------------
def run_one_trial(
    trial_id: str,
    cfg: Dict[str, Any],
    run_dir: str,
    max_epochs: int,
    eval_every: int,
    objective: str,
    seed: int,
) -> Dict[str, Any]:
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)

    # Always write config, even in dry-run
    (run_path / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    # ---------------- DRY RUN: exit BEFORE touching trainer/GPU/data ----------------
    if bool(getattr(args, "dry_run", False)):
        return {
            "trial_id": trial_id,
            "dry_run": True,
            "seed": seed,
            "run_dir": str(run_path),
            "config": cfg,
        }

    # deterministic seeding for this trial
    seed_torch(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # apply cfg -> Params.args
    _set_args_from_config(cfg)

    # trial-unique save folder (prevents collisions)
    args.save = str(run_path / "Save") + os.sep
    Path(args.save, args.data).mkdir(parents=True, exist_ok=True)

    # Build trainer AFTER args set (so it reads correct values)
    device = torch.device(getattr(args, "device", "cuda" if torch.cuda.is_available() else "cpu"))
    eng = trainer(device)

    best_val_obj = float("inf")
    best_epoch = -1
    best_val: Dict[str, float] = {}
    best_test: Dict[str, float] = {}
    last_train_loss = math.nan

    t0 = time.time()
    for ep in range(1, max_epochs + 1):
        _, train_loss = eng.train()
        last_train_loss = float(train_loss)

        if ep % eval_every == 0:
            res_val = eng.eval(True, True)
            res_test = eng.eval(False, True)

            val_obj = _objective(res_val, objective)
            if val_obj < best_val_obj:
                best_val_obj = float(val_obj)
                best_epoch = int(ep)
                best_val = dict(res_val)
                best_test = dict(res_test)
                
                ckpt_dir = Path(args.save) / args.data
                ckpt_dir.mkdir(parents=True, exist_ok=True)

                torch.save(
                    eng.model.state_dict(),
                    ckpt_dir / f"_epoch_{ep}_VALOBJ_{best_val_obj:.6f}.pth"
                )
                
                # write best.json
                (run_path / "best.json").write_text(
                    json.dumps(
                        {
                            "trial_id": trial_id,
                            "best_epoch": best_epoch,
                            "best_val_obj": best_val_obj,
                            "best_val_metrics": best_val,
                            "best_test_metrics": best_test,
                            "config": cfg,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )

    elapsed_s = time.time() - t0

    result = {
        "trial_id": trial_id,
        "dry_run": False,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_obj": best_val_obj,
        "best_val_RMSE": float(best_val.get("RMSE", math.nan)),
        "best_val_MAE": float(best_val.get("MAE", math.nan)),
        "best_val_MAPE": float(best_val.get("MAPE", math.nan)),
        "best_test_RMSE": float(best_test.get("RMSE", math.nan)),
        "best_test_MAE": float(best_test.get("MAE", math.nan)),
        "best_test_MAPE": float(best_test.get("MAPE", math.nan)),
        "last_train_loss": float(last_train_loss),
        "elapsed_s": float(elapsed_s),
        "config": cfg,
    }

    (run_path / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    del eng
    _cleanup_torch()
    return result


def _trial_worker(payload: Tuple[str, Dict[str, Any], str, int, int, str, int]) -> Dict[str, Any]:
    return run_one_trial(*payload)


def main() -> None:
    p = argparse.ArgumentParser(description="Ray-free tuner for STHSL (with dry-run).")
    p.add_argument("--data", required=True)
    p.add_argument("--arch", choices=["baseline", "option1", "option2"], required=True)
    p.add_argument("--num-trials", type=int, default=20)
    p.add_argument("--max-epochs", type=int, default=5)
    p.add_argument("--eval-every", type=int, default=1)
    p.add_argument("--objective", choices=["rmse", "mae", "mape", "rmse_mae"], default="rmse_mae")
    p.add_argument("--workers", type=int, default=1, help="Parallel trial workers. Use 1 for sequential.")
    p.add_argument("--results-dir", type=str, default="./tune_results")
    p.add_argument("--name", type=str, default=None)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--mode", choices=["random"], default="random")
    p.add_argument("--gpus", type=float, default=0.0, help="Only used to infer cpu_only if 0.")

    # ---- DRY RUN FLAG ----
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Only sample configs and create trial folders (write config.json). No training/eval.",
    )

    args_cli = p.parse_args()

    # Put dry-run flag into Params.args so subprocesses can see it
    args.dry_run = bool(args_cli.dry_run)

    # Apply base args
    args.data = args_cli.data
    args.use_ode_option = args_cli.arch

    cpu_only = (args_cli.gpus == 0.0)
    space = build_space(args_cli.arch, cpu_only=cpu_only)

    exp_name = args_cli.name or f"STHSL_TUNE_NORAY_{args_cli.arch}_{args_cli.data}"
    root = Path(args_cli.results_dir) / exp_name
    root.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args_cli.seed)

    # build trial payloads
    payloads: List[Tuple[str, Dict[str, Any], str, int, int, str, int]] = []
    for i in range(args_cli.num_trials):
        cfg = _sample(space, rng)
        trial_id = f"trial_{i:04d}"
        run_dir = str(root / trial_id)
        seed = (args_cli.seed * 1000003 + i) % (2**31 - 1)
        payloads.append((trial_id, cfg, run_dir, args_cli.max_epochs, args_cli.eval_every, args_cli.objective, seed))

    # run trials
    results: List[Dict[str, Any]] = []
    if args_cli.workers <= 1:
        for pl in payloads:
            r = _trial_worker(pl)
            results.append(r)
            if r.get("dry_run"):
                print(f"[dry-run] {r['trial_id']} wrote {Path(r['run_dir']) / 'config.json'}")
            else:
                print(f"[done] {pl[0]} best_val_obj={r['best_val_obj']:.6f} best_epoch={r['best_epoch']}")
    else:
        ctx = mp.get_context("spawn")  # safer with torch
        with ctx.Pool(processes=args_cli.workers) as pool:
            for r in pool.imap_unordered(_trial_worker, payloads):
                results.append(r)
                if r.get("dry_run"):
                    print(f"[dry-run] {r['trial_id']} wrote {Path(r['run_dir']) / 'config.json'}")
                else:
                    print(f"[done] {r['trial_id']} best_val_obj={r['best_val_obj']:.6f} best_epoch={r['best_epoch']}")

    # If dry-run, stop here (no ranking, no CSV)
    if args_cli.dry_run:
        print("\nDry run complete.")
        print(f"Generated {len(results)} trial configs under: {root}")
        return

    # rank and write summary
    results_sorted = sorted(results, key=lambda d: float(d.get("best_val_obj", 1e18)))
    best = results_sorted[0] if results_sorted else None

    (root / "BEST.json").write_text(json.dumps(best, indent=2), encoding="utf-8")

    # CSV summary
    csv_path = root / "summary.csv"
    cols = [
        "trial_id",
        "seed",
        "best_epoch",
        "best_val_obj",
        "best_val_RMSE",
        "best_val_MAE",
        "best_val_MAPE",
        "best_test_RMSE",
        "best_test_MAE",
        "best_test_MAPE",
        "last_train_loss",
        "elapsed_s",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in results_sorted:
            w.writerow({k: r.get(k) for k in cols})

    print("\n=== BEST (by best_val_obj) ===")
    if best:
        print("trial_id:", best["trial_id"])
        print("best_val_obj:", best["best_val_obj"])
        print("best_epoch:", best["best_epoch"])
        print("config:", json.dumps(best["config"], indent=2))
    print(f"\nWrote: {csv_path}")


if __name__ == "__main__":
    main()
