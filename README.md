# Assignment A: Network Telemetry Fault Detection & Root Cause Attribution

This repository is a self-contained solution for the Nomiso AI/ML internship assignment.

## Folder structure

```text
assignment_telemetry/
  code/                  Python source code
  data/                  generated telemetry, anomaly scores, detections, metrics
  ground_truth/          true fault schedule and labels - used only by evaluation
  plots/                 verification and analysis plots
  report.pdf             short report
  README.md
  requirements.txt
  DECLARATIONS.md
```

## How to run

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
python code/run_all.py
```

The command regenerates the dataset, runs unsupervised detection, evaluates the results, creates plots, and rebuilds `report.pdf`.

## Design summary

- 10 cell sites, 21 days, 15-minute telemetry, 12 KPIs.
- Normal data includes daily and weekly seasonality, site heterogeneity, KPI correlation, heteroscedastic noise, and benign missing values.
- 30 injected fault events cover F1-F6: RF degradation, clock drift, power instability, sleeping cell, interference burst, and capacity overload.
- Days 1-10 are used for training. Days 11-21 are used for evaluation.
- Detection is unsupervised and does **not** import or read `ground_truth/`.
- `ground_truth/` is read only in `code/evaluate.py`.

## Main outputs

- `data/telemetry.csv`: generated KPI data.
- `ground_truth/fault_events.csv`: event-level truth file.
- `data/scores.csv`: per-site, per-timestamp anomaly scores and alerts.
- `data/detected_events.csv`: detected intervals with top-3 KPI attribution and predicted fault type.
- `data/metrics.json`: event-level, point-level, point-adjust, delay, false-alarm, and attribution metrics.
- `data/per_archetype_metrics.csv`: per-fault-type breakdown.

## Evaluation notes

Event-level detection counts a detection as correct when it overlaps a true fault interval on the same site. Point-level F1 is reported both with and without point-adjust. The non-adjusted value is more conservative because point-adjust can inflate performance for long anomalies.
