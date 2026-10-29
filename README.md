# Online supplement: Auditing Refresh Scheduling in Collaborative Sensing

This repository is the online supplement for the CSCWD 2027 submission
*Auditing Refresh Scheduling in Collaborative Sensing*. It contains the
supplementary PDF, the code for the matched offline schedule (V28) and the
causal-imitation experiment (V29), and the summary tables behind the main
paper's numbers and figures.

## Contents

| Path | What it is |
|---|---|
| `supplement.pdf` | Supplementary material: protocol details, subject-level design outcomes, participant-specific payload analysis, aggregate design-selection table, and the causal-imitation summary (Section V). |
| `code/run_matched_oracle_v28.py` | Matched zero-conflict offline schedule, Equation (8) of the paper. |
| `code/run_oracle_imitation_v29.py` | Causal imitation of the offline schedule, Section III-E / V-C. Includes `selftest`, `freeze`, `run` and `summarize` actions. |
| `code/compare_v29.py` | Joins V29 with the frozen V20/V23 forecaster policies and the V28 offline points. |
| `code/imitation_frontier.py` | Draws Fig. 3 from `results/oracle_imitation_v29/frontier_points.csv`. |
| `results/oracle_imitation_v29/frozen.json` | Protocol frozen before any evaluation-subject (A) result; `source_sha256` is the SHA-256 of `code/run_oracle_imitation_v29.py`. |
| `results/oracle_imitation_v29/dataset_equal_metrics.csv` | Subject-equal results for all 18 imitation settings on all nine dataset-grid scenarios, with the package fee charged and waived. |
| `results/oracle_imitation_v29/research_decision.json` | Joint-screen decision: no imitation setting passes any scenario. |
| `results/oracle_imitation_v29/frontier_points.csv` | All points of the saving-disagreement frontier (V20/V23 forecasters, V28 offline, V29 imitation). |
| `results/matched_oracle_v28/fee_matched_dataset_equal_metrics.csv` | Fee-matched offline savings (Table II of the paper). |
| `results/matched_calibration_v20/dataset_equal_metrics.csv`, `results/opportunity_v23/dataset_equal_metrics.csv` | Frozen forecaster and baseline policies for the original and external datasets. |

## Reproducing

The scripts import modules from the full experiment pipeline and read frozen
classifier-score frames derived from PAMAP2, MHEALTH, and OPPORTUNITY (UCI
Machine Learning Repository). Python 3.13 with numpy, pandas, scikit-learn,
numba, and threadpoolctl was used. Within the pipeline folder:

```
python run_oracle_imitation_v29.py selftest
python run_oracle_imitation_v29.py run --workers 8
python run_oracle_imitation_v29.py summarize
python compare_v29.py
```

`run` refuses to start if the script's hash differs from `frozen.json`.
The external OPPORTUNITY cases were also rerun in a second environment with a
different scikit-learn version; savings differed by at most 0.64 and
disagreement by at most 0.27 percentage points. Reported numbers come from the
full run in the original environment.

## Scope

All datasets had prior development exposure; grids and repeated views add no
new individuals. The offline schedule uses future scores and is a diagnostic,
not a deployable policy. No physical-network, radio, or energy measurement is
included.
