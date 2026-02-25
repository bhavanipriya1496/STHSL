import re
import sys
import os
import pandas as pd
from typing import Optional, Tuple

CATEGORY_MAP = {
    "0": "Burglary",
    "1": "Larceny",
    "2": "Robbery",
    "3": "Assault",
}

SPARSITY_MAP = {
    "1": "<=0.25",
    "2": "<=0.5",
    "3": "<=0.75",
    "4": "<=1.0",
}


def parse_metrics_from_file(filepath: str) -> dict:
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    best_match = re.search(r"Best:(.*)", text)
    if not best_match:
        print(f"No 'Best:' line found in {filepath}")
        return {}

    metrics_line = best_match.group(1)

    loss_match = re.search(r"epochLoss\s*=\s*([0-9.]+)", text)
    epoch_loss_value = float(loss_match.group(1)) if loss_match else None

    pairs = re.findall(r"([A-Za-z0-9_]+)\s*=\s*([0-9.]+)", metrics_line)

    metrics = {}

    for key, value in pairs:
        value = float(value)

        if key in [
            "RMSE",
            "MAE",
            "MAPE",
            "MicroF1",
            "MacroF1",
            "AP",
            "Acc",
            "MacroAP_over_categories",
            "MacroF1_over_categories",
            "MacroAcc_over_categories",
        ]:
            metrics[key] = value
            continue

        m = re.match(r"(RMSE|MAE|MAPE|F1_cate_|AP_cate|Acc_cate)_([0-9]+)", key)
        if m:
            mtype, cid = m.groups()
            cname = CATEGORY_MAP.get(cid, cid)
            metrics[f"{mtype}_{cname}"] = value
            continue

        s = re.match(r"(RMSE|MAE|MAPE|F1|AP|Acc)_mask_([0-9]+)", key)
        if s:
            mtype, sid = s.groups()
            sname = SPARSITY_MAP.get(sid, sid)
            metrics[f"{mtype}_sparsity_{sname}"] = value
            continue

    if epoch_loss_value is not None:
        metrics["epochLoss"] = epoch_loss_value

    return metrics


def parse_seed_epoch_from_filename(filename: str) -> Optional[Tuple[int, int]]:
    """
    New filenames:
      seed_<S>_epoch_<E>_main.txt  -> (S, E)

    Backward compatible:
      epoch_<E>_main.txt -> (0, E)
    """
    m = re.fullmatch(r"seed_(\d+)_epoch_(\d+)_main\.txt", filename)
    if m:
        return int(m.group(1)), int(m.group(2))

    m2 = re.fullmatch(r"epoch_(\d+)_main\.txt", filename)
    if m2:
        return 0, int(m2.group(1))

    return None


def flat_col_name(seed: int, epoch: int) -> str:
    return f"seed{seed}_epoch{epoch}"


def main():
    if len(sys.argv) < 2:
        print("Usage: python metrics_extraction.py <folder>")
        sys.exit(1)

    folder = sys.argv[1]
    if not os.path.isdir(folder):
        print("Folder not found.")
        sys.exit(1)

    print(f"\nReading folder: {folder}\n")

    files: list[tuple[int, int, str]] = []
    for filename in os.listdir(folder):
        if not filename.endswith(".txt"):
            continue

        parsed = parse_seed_epoch_from_filename(filename)
        if parsed is None:
            continue

        seed, epoch = parsed
        files.append((seed, epoch, os.path.join(folder, filename)))

    if not files:
        print("No matching metrics files found (expected: seed_<S>_epoch_<E>_main.txt).")
        sys.exit(1)

    files.sort(key=lambda x: (x[0], x[1]))

    # Build dict: column_name -> metrics dict
    col_metrics: dict[str, dict] = {}
    for seed, epoch, filepath in files:
        col = flat_col_name(seed, epoch)
        print(f"Processing {col}: {os.path.basename(filepath)}")
        col_metrics[col] = parse_metrics_from_file(filepath)

    df = pd.DataFrame.from_dict(col_metrics, orient="columns")

    # ---- Sort columns by (seed, epoch) using the column naming scheme ----
    def sort_key(col: str):
        m = re.fullmatch(r"seed(\d+)_epoch(\d+)", col)
        if not m:
            return (10**9, 10**9)
        return (int(m.group(1)), int(m.group(2)))

    df = df.reindex(sorted(df.columns, key=sort_key), axis=1)

    # ---- OPTIONAL: make epochs continuous per seed (fills missing epochs with NaN) ----
    # Build map seed -> epochs present
    present = {}
    for col in df.columns:
        m = re.fullmatch(r"seed(\d+)_epoch(\d+)", col)
        if not m:
            continue
        s, e = int(m.group(1)), int(m.group(2))
        present.setdefault(s, []).append(e)

    # Create complete column list in sorted order
    full_cols = []
    for s in sorted(present.keys()):
        ep_list = sorted(set(present[s]))
        if not ep_list:
            continue
        min_ep, max_ep = ep_list[0], ep_list[-1]
        for e in range(min_ep, max_ep + 1):
            full_cols.append(flat_col_name(s, e))

    # Reindex to include missing epoch columns (NaN)
    df = df.reindex(columns=full_cols)

    # ---- Seed averages (avg across epochs for each seed) ----
    seed_avg_cols = {}
    for s in sorted(present.keys()):
        cols_s = [c for c in df.columns if c.startswith(f"seed{s}_epoch")]
        if cols_s:
            seed_avg_cols[f"seed_avg_seed{s}"] = df[cols_s].mean(axis=1, skipna=True)

    df_seed_avg = pd.DataFrame(seed_avg_cols)

    # ---- Overall average across seeds (mean of seed averages) ----
    if not df_seed_avg.empty:
        df_seed_avg["all_seeds_avg"] = df_seed_avg.mean(axis=1, skipna=True)
    else:
        df_seed_avg = pd.DataFrame({"all_seeds_avg": pd.Series(index=df.index, dtype=float)})

    # Merge and round
    df_all = pd.concat([df, df_seed_avg], axis=1).round(4)

    out_path = os.path.join(folder, "metrics_all_epochs_pivot.csv")
    df_all.to_csv(out_path)

    print(f"\nCreated: {out_path}\n")

    with pd.option_context("display.max_columns", 40, "display.width", 200):
        print(df_all)


if __name__ == "__main__":
    main()