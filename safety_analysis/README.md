# safety_analysis

Tools to export, check and analyse the training-time safety runs (wandb project `crax-srl`).
Everything here runs on a laptop; it only reads the exported CSV files.

| File | What it does |
|---|---|
| `download_wandb.py` | Exports every run in wandb groups starting with `safety` to `wandb_exports/<group>/<run>.csv` (one row per logged wandb row), plus `<run>.summary.json` (the final-policy evaluation) and, for runs from 6 Oct on, `<run>.final_eval_episodes.npz` (raw per-episode final-evaluation costs and rewards). Files that already exist are skipped. |
| `summarise_runs.py` | Quick one-line-per-run summary (standard library only) written to `run_summary.csv`. |
| `analysis_lib.py` | Shared loading, cleaning and metric definitions. The figures and the quoted numbers both use it, so they cannot disagree. |
| `make_figures.py` | Draws the paper's 2 main figures to `figures/paper/`, its 3 supplement figures to `figures/supplement/` and every other report figure to `figures/not_in_paper/` (`--paper` for the paper's figures only, `--pdf` also writes vector PDFs), plus `figures/INDEX.md`. The `PAPER`, `SUPPLEMENT` and `NOT_IN_PAPER` lists and `PAPER_FILES` (file names) at the bottom of the file are the single source of what exists. |
| `illustrations.py` | Explanatory figures with no run data (for talks and the introduction) to `figures/illustration/`: `intro_same_mean_different_risk`, three cost distributions with the same mean (= budget) but different V and tails. |
| `key_numbers.py` | Prints every number quoted in the reports and writes them to `figures/key_numbers.json`. |

`wandb_exports/`, `figures/` and `*.csv` are git-ignored: they are large or can be regenerated.

## Reproduce or verify the reports

From the repo folder:

```bash
python -m pip install numpy matplotlib          # only if they are missing
python safety_analysis/download_wandb.py        # once, or again after new runs finish
python safety_analysis/make_figures.py --pdf    # every figure, with vector PDFs (under a minute)
python safety_analysis/make_figures.py --paper  # only the 5 figures the paper uses
python safety_analysis/make_figures.py fig1_    # one figure, by file name (or function name, e.g. fig34)
python safety_analysis/illustrations.py         # the introduction illustration
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

## Paper figures (Paper Plan, 20 Oct; renamed 21 Oct)

The paper asks two questions: (RQ1) how the per-episode violation rate V depends on the mean cost,
task and robot; (RQ2) which training change reduces V at what reward cost. Since 21 Oct the figures
the paper uses are written under names that say where they appear; `figures/INDEX.md` (written by
`make_figures.py`) lists them with their LaTeX labels. Ready-made LaTeX blocks with captions are in
`Research Topics/4_Phase3_Research_Project/SRL_1/figure_blocks.tex`.

| In the paper | File (`figures/...`) | Drawing function | Shows |
|---|---|---|---|
| Fig. 1 (RQ1) | `paper/fig1_violation_rate_vs_mean_cost` | `fig34_mean_vs_share_all_tasks` | Every run of every task: mean cost / budget vs % of episodes over budget. One band: V is set by the mean. Prints the per-task numbers quoted in the paper. |
| Fig. 2 (RQ2) | `paper/fig2_penalty_floor_cost_distributions` | `fig37_floor_distributions` | Final-policy episode-cost distributions, CRAX default vs penalty floor, PPO-Lag / FOCOPS / P3O on Goal and Button |
| Fig. S1 | `supplement/figS1_goal_levels_metrics` | `fig17_levels_metrics` | Goal Levels 1-3: the mean holds while the tail (p99, % over 2d) grows; backs Table 1 |
| Fig. S2 | `supplement/figS2_training_vs_final_violation_rate` | `fig24_training_vs_final` | Training-time vs final-policy V, per run; justifies training-time data for RQ1 |
| Fig. S3 (optional) | `supplement/figS3_penalty_floor_summary` | `fig36_penalty_floor` | One-glance summary of Fig. 2: V and reward kept, default vs floor |
| Intro (optional) | `illustration/intro_same_mean_different_risk` | `illustrations.py` | Three made-up cost distributions with the same mean (= budget); not run data |

Paper figures are drawn at their printed width (single column 3.33 in, full width 7 in) with
7-8 pt text and carry no notes or long titles: what is plotted belongs in the LaTeX caption.

`figures/not_in_paper/` holds every other figure (the in-depth reports' figures, including fig03,
fig22, fig23, fig26 and fig32, which the paper dropped on 20 Oct) under its old `figNN_` name; the
`NOT_IN_PAPER` list in `make_figures.py` says which.

### Experiment 14: levers on Button (`scripts/snellius/14_levers_button.sh`)

PPO-Lag with target 12.5 (measured against 25) and P3O with fixed kappa = 20 on Button, T = 2000, 3 seeds,
50M steps (~30 SBU per run). Ran 11 Oct. `bash scripts/snellius/14_levers_button.sh list` previews (and checks
wandb's 128-character group-name limit), `sbatch` submits from the repo root, `python safety_analysis/download_wandb.py`
fetches. The runs appear in Fig. 1 and in `not_in_paper/fig32_tradeoff_final` (`BUTTON_LEVERS` in `make_figures.py`).

## Second batch (6 Oct): figures 22-33 (historical note)

Experiments `core_rerun` (10), `fixed_baselines` (09), `cpo` (13) and `other_tasks` (11). The
first report's figures and `key_numbers.py` never use these runs (`analysis_lib.batch1`), so
they stay exactly as before.

| Figure | Shows |
|---|---|
| fig22_final_eval_goal | Final policy on Goal Point L1, all 13 methods: V, mean cost, D_norm+, reward; stochastic (open) vs greedy (filled) |
| fig23_cdfs | CDF of D_norm (Spoor et al. Fig. 2 style): during training vs final stochastic vs final greedy; methods and variants |
| fig24_training_vs_final | Per run: % of training episodes over budget vs % of final greedy episodes over budget (all tasks) |
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
