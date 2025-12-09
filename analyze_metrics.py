import pandas as pd
import sys

def analyze_metrics(csv_path):

    # Load pivot CSV (metrics in rows, epochs in columns)
    df = pd.read_csv(csv_path)

    # First column is metric names
    df = df.rename(columns={df.columns[0]: "Metric"})

    # Identify epoch columns
    epoch_cols = [c for c in df.columns if c != "Metric"]

    results = []

    for idx, row in df.iterrows():
        metric_name = row["Metric"]

        # For each metric, find the epoch with the MIN value
        best_epoch = row[epoch_cols].idxmin()   # epoch column name
        best_value = row[best_epoch]

        results.append([metric_name, best_epoch, best_value])

    ranking_df = pd.DataFrame(results, columns=["Metric", "BestEpoch", "BestValue"])

    # Count how many times each epoch wins
    epoch_score = ranking_df["BestEpoch"].value_counts().sort_index()

    # Find overall winner
    overall_best_epoch = epoch_score.idxmax()
    overall_best_count = epoch_score.max()

    print("\n==============================")
    print("🚀 BEST EPOCH PER METRIC")
    print("==============================\n")
    print(ranking_df.to_string(index=False))

    print("\n==============================")
    print("🏆 OVERALL BEST EPOCH")
    print("==============================")
    print(f"Best epoch overall: {overall_best_epoch} (wins {overall_best_count} metrics)\n")

    # Save rankings
    out_path = csv_path.replace(".csv", "_ranking.csv")
    ranking_df.to_csv(out_path, index=False)
    print(f"📄 Saved detailed ranking: {out_path}")

    return ranking_df, overall_best_epoch


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_metrics.py <metrics_all_epochs_pivot.csv>")
        sys.exit(1)

    csv_file = sys.argv[1]
    analyze_metrics(csv_file)
