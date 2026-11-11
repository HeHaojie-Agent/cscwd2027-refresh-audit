# Online supplement: Auditing Refresh Scheduling in Collaborative Sensing

Online supplement for the CSCWD 2027 submission *Auditing Refresh Scheduling in
Collaborative Sensing* (Haojie He, Tianjin University of Science and Technology).
It contains the supplementary PDF, the code for the matched offline schedule, the
causal-imitation experiment and the auxiliary self-triggered push, the figure
scripts, and the summary tables behind the main paper's numbers and figures.

## Files

| File | What it is |
|---|---|
| `supplement.pdf` | Supplementary material: detailed setup, additional results of the original protocol (sensitivity checks, participant-specific payloads, decision-region table), the matched offline schedule, the causal-imitation summary, the auxiliary self-triggered push, and the cost-aware design-selection experiment. |
| `run_matched_oracle_v28.py` | Matched zero-conflict offline schedule, Eq. (7) of the paper (stage V28). |
| `run_oracle_imitation_v29.py` | Causal imitation of the offline schedule, Sections III-E and V-C (stage V29), with `selftest`, `freeze`, `run`, and `summarize` actions. |
| `compare_v29.py` | Joins the imitation results with the threshold forecasters and the offline schedule. |
| `run_aux_trigger_v30.py` | Auxiliary self-triggered push, Sections III-F and V-D (stage V30), with `selftest`, `freeze`, `run`, and `summarize` actions. |
| `run_aux_trigger_v30b.py` | Declared post hoc frontier extension of V30 with lower thresholds (stage V30b); it reuses the V30 code and B-fitted scales unchanged. |
| `run_learned_trigger_v31.py` | Learned auxiliary-side change detector (stage V31), designed after V30 results and frozen before its own evaluation; reported in the supplement. |
| `run_twoside_trigger_v32.py` | Two-sided trigger (stage V32): auxiliary push plus a primary margin guard, the method reported in Sections III-F and V-D. Designed after V30 and V31 and frozen before its own evaluation. |
| `system_overview.py` | Draws Fig. 1. Its timelines are schematic; bars and disagreement are measured PAMAP2 primary-grid values. |
| `cost_envelope.py` | Draws Fig. 2 from `figure_envelope_points.csv`. |
| `imitation_frontier.py` | Draws Fig. 3 from `frontier_points.csv`, the two `aux_trigger_*_points.csv` files, and `twoside_trigger_v32_points.csv`. |
| `figure_envelope_points.csv` | Fixed-trace control budgets plotted in Fig. 2. |
| `frontier_points.csv` | Threshold forecasters, offline schedule, and imitation policies in Fig. 3. |
| `frozen.json` | Imitation protocol frozen before any evaluation-subject (A) result; `source_sha256` is the SHA-256 of `run_oracle_imitation_v29.py`. |
| `research_decision.json` | Imitation joint-screen decision: no imitation setting passes any scenario. |
| `imitation_dataset_equal_metrics.csv` | Subject-equal results for all 18 imitation settings in all nine dataset-grid scenarios, with the model-package fee charged and waived. |
| `v30_frozen.json` | Self-trigger protocol frozen before any A result; `source_sha256` is the SHA-256 of `run_aux_trigger_v30.py`. |
| `v30_research_decision.json` | Self-trigger joint-screen decision and the B-selected thresholds. |
| `aux_trigger_v30_points.csv` | Subject-equal results of the frozen self-trigger (B-selected and nine fixed thresholds) in all nine scenarios. |
| `aux_trigger_v30b_points.csv` | Subject-equal results of the post hoc thresholds 0.5, 0.6, 0.75, and 0.9. |
| `v31_frozen.json` | Learned-detector protocol frozen before any A result; `source_sha256` is the SHA-256 of `run_learned_trigger_v31.py`. |
| `v31_research_decision.json` | Learned-detector joint-screen decision and the B-selected thresholds. |
| `learned_trigger_v31_points.csv` | Subject-equal results of the learned detector (B-selected and nine fixed thresholds). |
| `v32_frozen.json` | Two-sided trigger protocol frozen before any A result; `source_sha256` is the SHA-256 of `run_twoside_trigger_v32.py`. |
| `v32_research_decision.json` | Two-sided trigger joint-screen decision and the B-selected pairs. |
| `twoside_trigger_v32_points.csv` | Subject-equal results of the two-sided trigger (B-selected and 30 fixed pairs). |
| `fee_matched_dataset_equal_metrics.csv` | Fee-matched offline savings reported in Section V-B. |
| `forecasters_original_dataset_equal_metrics.csv` | Threshold-forecaster and baseline policies on PAMAP2 and MHEALTH (stage V20). |
| `forecasters_external_dataset_equal_metrics.csv` | The same policies on the external OPPORTUNITY replication (stage V23). |

## Reproducing

The experiment scripts are reference copies from the full pipeline: they import
other pipeline modules and read frozen classifier-score frames and cleaned raw
signals derived from PAMAP2, MHEALTH, and OPPORTUNITY (UCI Machine Learning
Repository), which are not redistributed here. Python 3.13 with numpy, pandas,
scikit-learn, numba, and threadpoolctl was used. Inside the pipeline folder:

```
python run_oracle_imitation_v29.py selftest
python run_oracle_imitation_v29.py run --workers 8
python run_oracle_imitation_v29.py summarize
python run_aux_trigger_v30.py selftest
python run_aux_trigger_v30.py run --workers 8
python run_aux_trigger_v30.py summarize
python run_aux_trigger_v30b.py run --workers 8
python run_aux_trigger_v30b.py summarize
python run_learned_trigger_v31.py selftest
python run_learned_trigger_v31.py run --workers 8
python run_learned_trigger_v31.py summarize
python run_twoside_trigger_v32.py selftest
python run_twoside_trigger_v32.py run --workers 8
python run_twoside_trigger_v32.py summarize
```

Each `run` refuses to start if the frozen script's hash differs from its frozen
protocol. The external OPPORTUNITY imitation cases were also rerun in a second
environment with a different scikit-learn version; savings differed by at most
0.64 and disagreement by at most 0.27 percentage points. Reported numbers come
from the full runs in the original environment. The three figure scripts run
directly in this folder.

## Scope

All datasets had prior development exposure; grids and repeated views add no new
individuals. The offline schedule uses future scores and is a diagnostic, not a
deployable policy. Self-trigger thresholds below 1 were added after the frozen
results were seen and describe the trade-off only. V30, V31, and V32 were
designed in sequence on the same subjects; each was frozen before its own
evaluation, and all three are reported. No physical-network, radio,
or energy measurement is included.
