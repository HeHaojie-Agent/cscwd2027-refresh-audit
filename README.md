# Online supplement: Auditing Refresh Scheduling in Collaborative Sensing

Online supplement for the CSCWD 2027 submission *Auditing Refresh Scheduling in
Collaborative Sensing* (Haojie He and Xin Zhong, Tianjin University of Science and Technology).
It contains the supplementary PDF, the code for the matched offline schedule and
the causal-imitation experiment, and the summary tables behind the main paper's
numbers and figures.

## Files

| File | What it is |
|---|---|
| `supplement.pdf` | Supplementary material: detailed setup, additional results of the original protocol (sensitivity checks, participant-specific payloads), the matched offline schedule, the causal-imitation summary, and the cost-aware design-selection experiment. |
| `run_matched_oracle_v28.py` | Matched zero-conflict offline schedule, Eq. (8) of the paper (stage V28). |
| `run_oracle_imitation_v29.py` | Causal imitation of the offline schedule, Sections III-E and V-C (stage V29), with `selftest`, `freeze`, `run`, and `summarize` actions. |
| `compare_v29.py` | Joins the imitation results with the threshold forecasters and the offline schedule. |
| `imitation_frontier.py` | Draws Figs. 2 and 3 of the paper from `figure_envelope_points.csv` and `frontier_points.csv`. |
| `figure_envelope_points.csv` | Fixed-trace control budgets plotted in Fig. 2. |
| `frozen.json` | Imitation protocol frozen before any evaluation-subject (A) result; `source_sha256` is the SHA-256 of `run_oracle_imitation_v29.py`. |
| `imitation_dataset_equal_metrics.csv` | Subject-equal results for all 18 imitation settings in all nine dataset-grid scenarios, with the model-package fee charged and waived. |
| `research_decision.json` | Joint-screen decision: no imitation setting passes any scenario. |
| `frontier_points.csv` | All points of Fig. 3: threshold forecasters, offline schedule, and imitation policies. |
| `fee_matched_dataset_equal_metrics.csv` | Fee-matched offline savings (Table II of the paper). |
| `forecasters_original_dataset_equal_metrics.csv` | Threshold-forecaster and baseline policies on PAMAP2 and MHEALTH (stage V20). |
| `forecasters_external_dataset_equal_metrics.csv` | The same policies on the external OPPORTUNITY replication (stage V23). |

## Reproducing

The scripts are reference copies from the full experiment pipeline: they import
other pipeline modules and read frozen classifier-score frames derived from
PAMAP2, MHEALTH, and OPPORTUNITY (UCI Machine Learning Repository), which are not
redistributed here. Python 3.13 with numpy, pandas, scikit-learn, numba, and
threadpoolctl was used. Inside the pipeline folder:

```
python run_oracle_imitation_v29.py selftest
python run_oracle_imitation_v29.py run --workers 8
python run_oracle_imitation_v29.py summarize
python compare_v29.py
```

`run` refuses to start if the script's hash differs from `frozen.json`. The
external OPPORTUNITY cases were also rerun in a second environment with a
different scikit-learn version; savings differed by at most 0.64 and
disagreement by at most 0.27 percentage points. Reported numbers come from the
full run in the original environment. `imitation_frontier.py` runs directly in
this folder.

## Scope

All datasets had prior development exposure; grids and repeated views add no new
individuals. The offline schedule uses future scores and is a diagnostic, not a
deployable policy. No physical-network, radio, or energy measurement is included.
