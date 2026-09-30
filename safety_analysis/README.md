# safety_analysis

Tools to export, check and analyse the training-time safety runs (wandb project `crax-srl`).
Everything here runs on a laptop; it only reads the exported CSV files.

| File | What it does |
|---|---|
| `download_wandb.py` | Exports every run in wandb groups starting with `safety` to `wandb_exports/<group>/<run>.csv` (one row per logged wandb row). Runs that already have a CSV are skipped. |
| `summarise_runs.py` | Quick one-line-per-run summary (standard library only) written to `run_summary.csv`. |
| `analysis_lib.py` | Shared loading, cleaning and metric definitions. The figures and the quoted numbers both use it, so they cannot disagree. |
| `make_figures.py` | Draws all 21 figures of the in-depth report to `figures/figNN_*.png` (`--pdf` also writes vector PDFs for the paper). `NN` is the figure number in the report. |
| `key_numbers.py` | Prints every number quoted in the reports and writes them to `figures/key_numbers.json`. |

`wandb_exports/`, `figures/` and `*.csv` are git-ignored: they are large or can be regenerated.

## Reproduce or verify the reports

From the repo folder:

```bash
python -m pip install numpy matplotlib          # only if they are missing
python safety_analysis/download_wandb.py        # once, or again after new runs finish
python safety_analysis/make_figures.py          # all 21 figures (under a minute)
python safety_analysis/make_figures.py --pdf    # the same, plus vector PDF copies
python safety_analysis/make_figures.py fig07    # one figure
python safety_analysis/key_numbers.py           # all quoted numbers
python safety_analysis/key_numbers.py levels    # only the sections whose name contains 'levels'
```

What to expect on another computer:

* `figures/key_numbers.json` should be identical, and every number in the reports appears in
  the printed output of `key_numbers.py` (search for it).
* The figures should look the same. The PNG files themselves can differ in a few edge
  pixels between computers (for example Apple silicon vs Intel), because the drawing library
  rounds slightly differently on different processors; the plotted values are the same.
* A full run of `make_figures.py` deletes `figNN_*` files that no current figure writes.
  Figure numbers shifted when figures were added, so an old file could otherwise be
  mistaken for the new figure with the same number.

## How the data is cleaned (analysis_lib.py)

* **Averaged rows are removed.** Before commit 720b86c, the trainer also logged the average
  of each epoch's updates at the step of the epoch's last update, which is also the step of
  the epoch's evaluation, overwriting that update (1 row in 7: 19 rows per run, one per epoch;
  PPO-Saute 1, as it trained as a single epoch). All 113 runs of the batch used that code.
  `averaged_rows` drops them in two steps: a row with a non-whole number of robots with cost
  or of finished episodes is certainly averaged; and in any run with such a row, every
  evaluation row is averaged too (an average is whole by chance about 1 time in 7, so the
  first test alone missed 0-6 rows per run). Evaluation values and episodic values in those
  rows are genuine and kept. `summarise_runs.py` applies the same rule.
* **Curves bridge the removed rows** by linear interpolation (plots only, never metrics).
* **Episode metrics start after about 2M steps**: with desynced robots the first, shortened
  episodes are excluded, so `safety_ep/*` is empty until full episodes finish.

## Metric definitions

| Name in the code | Meaning |
|---|---|
| budget `d` | Allowed cost per episode: 25 in every experiment (1000-step Goal Point episodes, 2000-step Ant episodes). |
| `avg_cost` | Mean over updates of the update's mean cost per episode (`training/safety/env_cost_mean`). |
| `avg_excess` | Mean over updates of `max(0, mean cost - d)`: how far above the budget, on average. |
| `pct_over`, `pct_over_10pct` | % of updates whose mean cost is above `d`, or above `1.1 d`. |
| `clear_crossings` | Times the mean moves from below `0.9 d` to above `1.1 d` or back (wobbles near `d` are not counted). |
| `first_under_m` | First update (million steps) whose mean cost is at or below `d`. |
| `ep_share_over`, `ep_share_over_2x` | % of all finished training episodes whose true cost exceeded `d` or `2 d` (weighted by episodes per update). |
| `ep_final_*` | The same, or episode-cost quantiles (mean, p50, p90, p99), over the last 25% of updates. |
| `slice_final_*` | Per-update slice statistics over the last 25% of updates. Slice = one robot's cost over the steps it ran in one update, scaled to an episode. The mean is accurate; slice tails overstate episode tails. |
| `reward_final` | Mean of the last 10 logged training-episode rewards (`episodic/sum_reward`). |
| `eval_cost_final` | Cost of the last evaluation (`eval/episode_cost`; stochastic policy, fresh episodes). |

Windows: the seven-algorithm comparison uses updates up to 30.2M steps (PPO-Saute ran for
30M); everything else uses the full run.
