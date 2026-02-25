import pandas as pd
import sys
import re
from typing import Optional, Tuple, Dict, List


"""
Lower is better (min): RMSE/MAE/MAPE/Loss
Higher is better (max): F1/AP/Acc
"""
def metric_goal(metric_name: str) -> str:
    """
    Objective direction for your metric names.

    Maximize: F1 / AP / Acc (including category/sparsity and macro-over-categories variants)
    Minimize: RMSE / MAE / MAPE / epochLoss (including category/sparsity variants)
    """
    m = str(metric_name).strip()

    # --- minimize (errors) ---
    if (
        m.startswith("RMSE") or m.startswith("MAE") or m.startswith("MAPE")
        or m == "epochLoss"
        or m.startswith("RMSE_") or m.startswith("MAE_") or m.startswith("MAPE_")
        or m.startswith("RMSE_sparsity_") or m.startswith("MAE_sparsity_") or m.startswith("MAPE_sparsity_")
    ):
        return "min"

    # --- maximize (scores) ---
    if (
        m == "MicroF1" or m == "MacroF1"
        or m == "AP" or m == "Acc"
        or m.startswith("Acc_") or m.startswith("AP_") or m.startswith("F1_")
        or m.startswith("Acc_cate_") or m.startswith("AP_cate_") or m.startswith("F1_cate_")
        or m.startswith("Acc_sparsity_") or m.startswith("AP_sparsity_") or m.startswith("F1_sparsity_")
        or m in ("MacroF1_over_categories", "MacroAP_over_categories", "MacroAcc_over_categories")
    ):
        return "max"

    # Fallback: treat unknowns as "min" (safer than accidentally maximizing an error)
    return "min"

def try_read_multiindex_csv(csv_path: str) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None

    if not isinstance(df.columns, pd.MultiIndex) or df.columns.nlevels != 2:
        return None

    # Heuristic: MultiIndex CSV produced by pandas often has "Unnamed: 0_level_0"
    # as first column (metric names column)
    return df


def flatten_col_multi(col0, col1) -> str:
    """
    Convert multiindex column (seed, epoch) into a readable single string.
    """
    s0 = str(col0).strip()
    s1 = str(col1).strip()

    if s0.isdigit() and s1.isdigit():
        return f"seed{s0}_epoch{s1}"

    if s0.lower() == "seed_avg" and s1.isdigit():
        return f"seed_avg_seed{s1}"

    if s0.lower() == "all_seeds_avg":
        return "all_seeds_avg"

    # fallback
    return f"{s0}_{s1}"


def parse_flat_seed_epoch(col: str) -> Optional[Tuple[int, int]]:
    m = re.fullmatch(r"seed(\d+)_epoch(\d+)", str(col))
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def parse_flat_seed_avg(col: str) -> Optional[int]:
    m = re.fullmatch(r"seed_avg_seed(\d+)", str(col))
    if not m:
        return None
    return int(m.group(1))


def load_any_metrics_csv(csv_path: str) -> pd.DataFrame:
    """
    Loads your metrics_all_epochs_pivot.csv in either:
      - MultiIndex (2 header rows) OR
      - Flat header

    Returns a DataFrame with:
      - a real "Metric" column
      - flattened per-epoch columns like seed0_epoch10
    """
    df_mi = try_read_multiindex_csv(csv_path)

    if df_mi is not None:
        # IMPORTANT FIX:
        # First column contains metric names but its header is ("Unnamed: 0_level_0", "Unnamed: 0_level_1")
        metric_series = df_mi.iloc[:, 0]
        values = df_mi.iloc[:, 1:].copy()

        # Flatten remaining columns
        flat_cols = [flatten_col_multi(c[0], c[1]) for c in values.columns]
        values.columns = flat_cols

        # Insert Metric column
        values.insert(0, "Metric", metric_series)
        return values

    # Flat CSV fallback
    df = pd.read_csv(csv_path, header=0)
    df = df.rename(columns={df.columns[0]: "Metric"})
    return df


def analyze_metrics(csv_path: str):
    df = load_any_metrics_csv(csv_path)

    # Identify columns
    per_epoch_cols: List[str] = []
    seed_avg_cols: List[str] = []
    other_cols: List[str] = []

    for c in df.columns:
        if c == "Metric":
            continue
        if parse_flat_seed_epoch(c) is not None:
            per_epoch_cols.append(c)
        elif parse_flat_seed_avg(c) is not None:
            seed_avg_cols.append(c)
        else:
            other_cols.append(c)

    if not per_epoch_cols and not seed_avg_cols:
        raise SystemExit(
            "No seed/epoch columns found. Expected columns like seed0_epoch10 and/or seed_avg_seed0.\n"
            "If your CSV is MultiIndex, ensure it has 2 header rows as produced by pandas."
        )

    # Convert to numeric
    for c in per_epoch_cols + seed_avg_cols + other_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # seed -> epoch columns map
    seed_to_cols: Dict[int, List[str]] = {}
    for c in per_epoch_cols:
        s, e = parse_flat_seed_epoch(c)
        seed_to_cols.setdefault(s, []).append(c)

    for s in seed_to_cols:
        seed_to_cols[s].sort(key=lambda col: parse_flat_seed_epoch(col)[1])

    seeds = sorted(seed_to_cols.keys())

    # ===============================
    # (A) Best (seed, epoch) per metric across all seeds/epochs
    # ===============================
    best_overall_rows = []
    for _, row in df.iterrows():
        metric = row["Metric"]
        goal = metric_goal(metric)

        series = row[per_epoch_cols].dropna()
        if series.empty:
            best_overall_rows.append([metric, None, None, None, goal])
            continue

        best_col = series.idxmin() if goal == "min" else series.idxmax()
        best_val = float(series[best_col])
        s, e = parse_flat_seed_epoch(best_col)
        best_overall_rows.append([metric, s, e, best_val, goal])

    best_overall_df = pd.DataFrame(
        best_overall_rows,
        columns=["Metric", "BestSeed", "BestEpoch", "BestValue", "Goal(min/max)"],
    )

    se_counts = (
        best_overall_df.dropna(subset=["BestSeed", "BestEpoch"])
        .assign(SeedEpoch=lambda x: x["BestSeed"].astype(int).astype(str) + "_" + x["BestEpoch"].astype(int).astype(str))
        ["SeedEpoch"]
        .value_counts()
    )

    overall_best_seed_epoch = se_counts.idxmax() if not se_counts.empty else None
    overall_best_seed_epoch_wins = int(se_counts.max()) if not se_counts.empty else 0

    # ===============================
    # (B) Best epoch per metric within each seed
    # ===============================
    per_seed_best_rows = []
    for _, row in df.iterrows():
        metric = row["Metric"]
        goal = metric_goal(metric)

        for s in seeds:
            cols_s = seed_to_cols[s]
            series = row[cols_s].dropna()
            if series.empty:
                per_seed_best_rows.append([metric, s, None, None, goal])
                continue

            best_col = series.idxmin() if goal == "min" else series.idxmax()
            best_val = float(series[best_col])
            _, e = parse_flat_seed_epoch(best_col)
            per_seed_best_rows.append([metric, s, e, best_val, goal])

    per_seed_best_df = pd.DataFrame(
        per_seed_best_rows,
        columns=["Metric", "Seed", "BestEpoch", "BestValue", "Goal(min/max)"],
    )

    per_seed_epoch_winner = []
    for s in seeds:
        sub = per_seed_best_df[(per_seed_best_df["Seed"] == s) & (per_seed_best_df["BestEpoch"].notna())]
        vc = sub["BestEpoch"].value_counts()
        if vc.empty:
            per_seed_epoch_winner.append([s, None, 0])
        else:
            per_seed_epoch_winner.append([s, int(vc.idxmax()), int(vc.max())])

    per_seed_epoch_winner_df = pd.DataFrame(
        per_seed_epoch_winner,
        columns=["Seed", "OverallBestEpochWithinSeed", "WinsCount"],
    )

    # ===============================
    # (C) Best seed by avg-over-epochs columns (optional)
    # ===============================
    best_seed_avg_df = pd.DataFrame()
    if seed_avg_cols:
        best_seed_avg_rows = []
        for _, row in df.iterrows():
            metric = row["Metric"]
            goal = metric_goal(metric)

            series = row[seed_avg_cols].dropna()
            if series.empty:
                best_seed_avg_rows.append([metric, None, None, goal])
                continue

            best_col = series.idxmin() if goal == "min" else series.idxmax()
            best_val = float(series[best_col])
            s = parse_flat_seed_avg(best_col)
            best_seed_avg_rows.append([metric, s, best_val, goal])

        best_seed_avg_df = pd.DataFrame(
            best_seed_avg_rows,
            columns=["Metric", "BestSeedByAvgEpochs", "BestValueByAvgEpochs", "Goal(min/max)"],
        )

    # ===============================
    # Print
    # ===============================
    print("\n==============================")
    print("BEST (SEED, EPOCH) PER METRIC (across all seeds/epochs)")
    print("==============================\n")
    print(best_overall_df.to_string(index=False))

    print("\n==============================")
    print("OVERALL BEST (SEED, EPOCH) BY 'MOST METRICS WON'")
    print("==============================")
    if overall_best_seed_epoch is None:
        print("No overall winner (no valid per-epoch values found).")
    else:
        s_str, e_str = overall_best_seed_epoch.split("_")
        print(f"Best (seed, epoch) overall: seed={int(s_str)} epoch={int(e_str)} (wins {overall_best_seed_epoch_wins} metrics)\n")

    print("\n==============================")
    print("OVERALL BEST EPOCH WITHIN EACH SEED (wins across metrics)")
    print("==============================\n")
    print(per_seed_epoch_winner_df.to_string(index=False))

    if not best_seed_avg_df.empty:
        print("\n==============================")
        print("BEST SEED BY AVERAGE-OVER-EPOCHS (per metric)")
        print("==============================\n")
        print(best_seed_avg_df.to_string(index=False))

    # ===============================
    # Save outputs
    # ===============================
    base = csv_path.replace(".csv", "")

    out_best_overall = base + "_ranking_seed_epoch.csv"
    best_overall_df.to_csv(out_best_overall, index=False)
    print(f"\nSaved: {out_best_overall}")

    out_per_seed = base + "_ranking_per_seed.csv"
    per_seed_best_df.to_csv(out_per_seed, index=False)
    print(f"Saved: {out_per_seed}")

    out_seed_summary = base + "_summary_per_seed.csv"
    per_seed_epoch_winner_df.to_csv(out_seed_summary, index=False)
    print(f"Saved: {out_seed_summary}")

    out_overall_summary = base + "_summary_overall_seed_epoch.csv"
    if overall_best_seed_epoch is None:
        summary_df = pd.DataFrame([{
            "OverallBestSeed": None,
            "OverallBestEpoch": None,
            "WinsCount": 0,
            "NumMetrics": int(len(best_overall_df)),
        }])
    else:
        s_str, e_str = overall_best_seed_epoch.split("_")
        summary_df = pd.DataFrame([{
            "OverallBestSeed": int(s_str),
            "OverallBestEpoch": int(e_str),
            "WinsCount": int(overall_best_seed_epoch_wins),
            "NumMetrics": int(len(best_overall_df)),
        }])
    summary_df.to_csv(out_overall_summary, index=False)
    print(f"Saved: {out_overall_summary}")

    if not best_seed_avg_df.empty:
        out_best_seed_avg = base + "_ranking_best_seed_by_avg.csv"
        best_seed_avg_df.to_csv(out_best_seed_avg, index=False)
        print(f"Saved: {out_best_seed_avg}")

    return {
        "best_overall_df": best_overall_df,
        "per_seed_best_df": per_seed_best_df,
        "per_seed_epoch_winner_df": per_seed_epoch_winner_df,
        "best_seed_avg_df": best_seed_avg_df,
        "overall_best_seed_epoch": overall_best_seed_epoch,
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyse_metrics_seeds.py <metrics_all_epochs_pivot.csv>")
        sys.exit(1)

    csv_file = sys.argv[1]
    analyze_metrics(csv_file)