"""Export the full per-update history of wandb runs to CSV (one file per run).

Usage (from the repo folder, with the crax env active):
    python safety_analysis/download_wandb.py                  # all runs in groups starting with "safety" (safety_*, safety2_*)
    python safety_analysis/download_wandb.py --group safety_ppo_lag_desync
    python safety_analysis/download_wandb.py --name <run display name>

Files are written to safety_analysis/wandb_exports/<group>/<run name>.csv
(git-ignored). Runs that already have
a CSV are skipped, so the script can be re-run as more runs finish.
"""
import argparse
import csv
import os

import wandb

PROJECT = "crax-srl"
# Exports live next to this script, independent of the working directory.
EXPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wandb_exports")


def export_run(run, out_dir):
    """Write every logged row of one run to <out_dir>/<run name>.csv."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{run.name}.csv")
    if os.path.exists(path):
        print(f"  skip (exists): {path}")
        return
    # history() fetches the logged rows in one request. It only subsamples when a
    # run has more rows than `samples`; our runs have ~150 rows (one per policy
    # update), so a large `samples` value returns every row. This is much faster
    # than scan_history(), which pages through the rows one request at a time.
    print(f"  downloading {run.name} ...", flush=True)
    rows = run.history(samples=100_000, pandas=False)
    # Rows can have different keys (eval metrics only at some steps), so take the
    # union of all column names in first-seen order; missing values stay empty.
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  saved {len(rows):4d} rows -> {path}  (state: {run.state})")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--group", help="export all runs in this wandb group")
    parser.add_argument("--name", help="export the single run with this display name")
    parser.add_argument("--group_prefix", default="safety",
                        help="if neither --group nor --name is given, export all groups starting with this")
    args = parser.parse_args()

    api = wandb.Api()
    path = f"{api.default_entity}/{PROJECT}"   # entity = the logged-in wandb account

    if args.name:
        runs = api.runs(path, filters={"display_name": args.name})
    elif args.group:
        runs = api.runs(path, filters={"group": args.group})
    else:
        runs = api.runs(path, filters={"group": {"$regex": f"^{args.group_prefix}"}})

    runs = list(runs)
    print(f"Found {len(runs)} run(s) in {path}")
    for run in runs:
        export_run(run, os.path.join(EXPORT_DIR, run.group or "no_group"))


if __name__ == "__main__":
    main()
