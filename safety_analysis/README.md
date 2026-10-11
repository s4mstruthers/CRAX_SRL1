# safety_analysis

Tools to export, check and analyse the training-time safety runs (wandb project `crax-srl`).
Everything here runs on a laptop; it only reads the exported CSV files.

| File | What it does |
|---|---|
| `download_wandb.py` | Exports every run in wandb groups starting with `safety` to `wandb_exports/<group>/<run>.csv` (one row per logged wandb row), plus `<run>.summary.json` (the final-policy evaluation) and, for runs from 6 Oct on, `<run>.final_eval_episodes.npz` (raw per-episode final-evaluation costs and rewards). Files that already exist are skipped. |
| `summarise_runs.py` | Quick one-line-per-run summary (standard library only) written to `run_summary.csv`. |
| `analysis_lib.py` | Shared loading, cleaning and metric definitions. The figures and the quoted numbers both use it, so they cannot disagree. |
| `make_figures.py` | Draws the paper's figures to `figures/figNN_*.png` (`--pdf` also writes vector PDFs). Numbering is kept from the two in-depth reports, so it has gaps: fig04, 07, 08, 10, 11, 12, 14, 15 and 20 were removed on branch `paper-trim` (11 Oct) because they are not in the paper; `git show main:safety_analysis/make_figures.py` has them. fig34 is new (see below). |
| `key_numbers.py` | Prints every number quoted in the reports and writes them to `figures/key_numbers.json`. |

`wandb_exports/`, `figures/` and `*.csv` are git-ignored: they are large or can be regenerated.

## Reproduce or verify the reports

From the repo folder:

```bash
python -m pip install numpy matplotlib          # only if they are missing
python safety_analysis/download_wandb.py        # once, or again after new runs finish
python safety_analysis/make_figures.py          # all paper figures (under a minute)
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

## Paper figures (branch paper-trim, 11 Oct)

The paper (direction brief in `Research Topics/4_Phase3_Research_Project/Notes/CRAX_project_direction_brief.pdf`)
asks two questions. Which figure answers what:

| Question | Figures |
|---|---|
| Measurement: per-update slice vs true episode metrics, desynced robots, reproducibility | fig01, fig02, fig33 |
| RQ1 structure: V is set by the mean, not the algorithm; harder tasks and another robot | **fig34** (all tasks), fig21 (Goal only), fig16, fig17, fig18, fig19, fig09, fig03, fig05 |
| RQ2 intervention: target, controller, tail signal, fixed penalty; reward cost | fig32, fig26, fig27, fig22, fig31, fig06, fig13 |
| Training vs final policy; Spoor et al. metrics, tiers, CDFs | fig24, fig23, fig25, fig29, fig30 |
| Discussion: CPO bias | fig28 |

`fig34_mean_vs_share_all_tasks` is the cross-task version of fig21 (step A1 of the brief): x = mean episode
cost / budget, y = % of episodes over budget, final quarter, one dot per run, colour = task, marker =
algorithm. It also prints, per task, the share of violating episodes for runs whose mean is within 5% of
the budget. Reading rule, fixed before looking: one band across tasks means V is set by the mean alone;
separate bands mean V at a given mean depends on the task's cost distribution.

### Experiment 14: levers on Button (`scripts/snellius/14_levers_button.sh`)

PPO-Lag with target 12.5 (measured against 25) and P3O with fixed kappa = 20 on Button, T = 2000, 3 seeds,
50M steps (~140 SBU). `bash scripts/snellius/14_levers_button.sh list` previews, `sbatch` submits from the
repo root, then `python safety_analysis/download_wandb.py`. The runs appear automatically in fig34 and in
the Button panel of fig32 (`BUTTON_LEVERS` in `make_figures.py`); `analysis_lib.parse_group` knows the
group name `safety_levers_button_...`.

## Second batch (6 Oct): figures 22-33

Experiments `core_rerun` (10), `fixed_baselines` (09), `cpo` (13) and `other_tasks` (11). The
first report's figures and `key_numbers.py` never use these runs (`analysis_lib.batch1`), so
they stay exactly as before.

| Figure | Shows |
|---|---|
| fig22_final_eval_goal | Final policy on Goal Point L1, all 13 methods: V, mean cost, D_norm+, reward; stochastic (open) vs greedy (filled) |
| fig23_cdfs | CDF of D_norm (Spoor et al. Fig. 2 style): during training vs final stochastic vs final greedy; methods and variants |
| fig24_training_vs_final | Per run: % of training episodes over budget vs % of final greedy episodes over budget (all tasks) |
| fig25_safety_tiers | Spoor et al. safety tiers (0-4) with D_norm, V, D_norm+, during training and for the final greedy policy |
| fig26_fixed_baselines | Fixed baselines vs their CRAX defaults (PID, P3O, Saute, tighter target, tail-driven λ): cost and reward |
| fig27_fixed_penalties | λ / κ of the fixed baselines |
| fig28_cpo_diagnostics | CPO vs PPO-Lag/FOCOPS (cost, reward), share of updates per optimisation case, line search, KL, constraint value c |
| fig29_tasks_curves | Circle, Push, Button (T = 2000): cost and reward per update; Goal Point PPO-Lag with T = 1000 vs 2000 |
| fig30_tasks_final | Final policy on the other tasks: V, D_norm+, reward; stochastic vs greedy |
| fig31_tails_goal | Worst cases on Goal Point L1: training p99 (final quarter), final CVaR95 and worst episode |
| fig32_tradeoff_final | Final greedy reward vs V, per task |
| fig33_reproducibility | First batch (03) vs rerun (10) with identical settings |

Notes: the final-policy CDF uses the 9 logged thresholds (dots), because wandb dropped the raw
1000-value lists of this batch; runs from 6 Oct on also attach the raw values as a file.
`safety_ep/cost_cvar95` is computed by rank since commit e50cf04 (the earlier threshold rule
returned the mean when >95% of episodes had zero cost; first-batch values are essentially
unaffected).

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
