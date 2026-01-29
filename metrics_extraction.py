import re
import sys
import os
import pandas as pd

CATEGORY_MAP = {
    "0": "Burglary",
    "1": "Larceny",
    "2": "Robbery",
    "3": "Assault"
}

SPARSITY_MAP = {
    "1": "<=0.25",
    "2": "<=0.5",
    "3": "<=0.75",
    "4": "<=1.0"
}


def parse_metrics_from_file(filepath):
    with open(filepath, "r") as f:
        text = f.read()

    # Extract Best line
    best_match = re.search(r"Best:(.*)", text)
    if not best_match:
        print(f"No 'Best:' line found in {filepath}")
        return {}

    metrics_line = best_match.group(1)

    # Extract last value = epochLoss
    loss_match = re.search(r"epochLoss\s*=\s*([0-9.]+)", text)
    epoch_loss_value = float(loss_match.group(1)) if loss_match else None

    pairs = re.findall(r"([A-Za-z0-9_]+)\s*=\s*([0-9.]+)", metrics_line)

    metrics = {}

    # Global metrics
    for key, value in pairs:
        value = float(value)

        if key in ["RMSE", "MAE", "MAPE", "MicroF1", "MacroF1", "AP", "Acc", "MacroAP_over_categories", "MacroF1_over_categories", "MacroAcc_over_categories"]:
            metrics[key] = value
            continue

        # Category-based (RMSE_0)
        m = re.match(r"(RMSE|MAE|MAPE|F1_cate_|AP_cate|Acc_cate)_([0-9]+)", key)
        if m:
            mtype, cid = m.groups()
            cname = CATEGORY_MAP.get(cid, cid)
            metrics[f"{mtype}_{cname}"] = value
            continue

        # Sparsity-based (RMSE_mask_1)
        s = re.match(r"(RMSE|MAE|MAPE|F1|AP|Acc)_mask_([0-9]+)", key)
        if s:
            mtype, sid = s.groups()
            sname = SPARSITY_MAP.get(sid, sid)
            metrics[f"{mtype}_sparsity_{sname}"] = value
            continue

    # Add epochLoss
    if epoch_loss_value is not None:
        metrics["epochLoss"] = epoch_loss_value

    return metrics


# MAIN: folder input
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python metrics_pivot.py <folder>")
        sys.exit(1)

    folder = sys.argv[1]

    if not os.path.isdir(folder):
        print("Folder not found.")
        sys.exit(1)

    all_epochs = {}

    print(f"\n Reading folder: {folder}\n")

    for filename in os.listdir(folder):
        if filename.endswith(".txt"):

            filepath = os.path.join(folder, filename)

            # extract epoch number
            m = re.search(r"epoch[_\-]?(\d+)", filename)
            epoch = int(m.group(1)) if m else None

            print(f"Processing: {filename}")

            metrics = parse_metrics_from_file(filepath)
            if epoch is not None:
                all_epochs[epoch] = metrics

    # Convert to DataFrame with one row per metric, columns = epochs
    df = pd.DataFrame(all_epochs)

    # Save pivot-style CSV
    out_path = os.path.join(folder, "metrics_all_epochs_pivot.csv")
    df.to_csv(out_path)

    print(f"\nCreated: {out_path}\n")
    print(df)
