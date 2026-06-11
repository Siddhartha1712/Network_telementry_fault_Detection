# Network Telemetry Fault Detection and Root Cause Attribution

This repository contains a self-contained solution for **Network Telemetry Fault Detection and Root Cause Attribution**.

The project creates realistic synthetic telecom network data, injects different types of network faults, detects anomalies using an unsupervised machine learning approach, and explains the possible root cause of each detected fault event.

---

## 1. Project Overview

Mobile network operators continuously monitor cell sites using different performance counters, also called KPIs. These KPIs include traffic volume, active users, call success rate, handover success rate, latency, signal quality, CPU load, temperature, voltage, and interference.

In real networks, faults may not always appear as sudden alarms. Some faults slowly affect multiple KPIs before customers notice a service problem. Traditional threshold-based alarms can create too many false alerts, which makes it difficult for operations teams to identify the actual issue.

This project aims to build a complete fault detection workflow using synthetic network telemetry data.

The system performs the following tasks:

- Generates realistic mobile network telemetry data
- Injects different network fault events
- Trains an unsupervised anomaly detection model
- Produces anomaly scores for each site and timestamp
- Converts alerts into detected events
- Identifies the top responsible KPIs
- Predicts the possible fault type
- Evaluates the detection performance

---

## 2. Repository Structure

```text
Network_telementry_fault_Detection/
│
├── code/
│   ├── config.yaml
│   ├── generate_data.py
│   ├── train_detect.py
│   ├── evaluate.py
│   ├── plot_utils.py
│   └── run_all.py
│
├── data/
│   ├── telemetry.csv
│   ├── scores.csv
│   ├── detected_events.csv
│   ├── thresholds.json
│   ├── metrics.json
│   ├── per_archetype_metrics.csv
│   └── point_eval_labels.csv
│
├── ground_truth/
│   ├── fault_events.csv
│   └── fault_schedule.json
│
├── plots/
│   ├── site_traffic_profiles.png
│   ├── score_case_study.png
│   ├── per_archetype_recall.png
│   └── missingness.png
│
├── DECLARATIONS.md
├── requirements.txt
├── report.pdf
└── README.md
```

---

## 3. Dataset Design

The project uses a synthetic telecom telemetry dataset generated from scratch.

| Item | Description |
|---|---|
| Number of cell sites | 10 |
| Duration | 21 days |
| Time interval | 15 minutes |
| Rows per site | 2,016 |
| Total rows | 20,160 |
| Number of KPIs | 12 |
| Fault events | 30 |
| Training period | Days 1 to 10 |
| Evaluation period | Days 11 to 21 |

The dataset is designed to be realistic and includes:

- Daily seasonality
- Weekly traffic variation
- Site-to-site differences
- Correlated KPI behaviour
- Heteroscedastic noise
- Benign missing values
- Multiple fault patterns

---

## 4. KPIs Generated

The following KPIs are generated for every cell site:

| KPI Name | Description |
|---|---|
| `downlink_traffic_gb` | Downlink traffic volume in GB |
| `active_users` | Number of active users at the site |
| `call_setup_success_rate_pct` | Percentage of successfully established calls |
| `call_drop_rate_pct` | Percentage of calls that dropped |
| `handover_success_rate_pct` | Percentage of successful handovers |
| `avg_signal_quality_db` | Average signal quality |
| `retransmission_rate_pct` | Percentage of retransmissions |
| `latency_ms` | Network latency in milliseconds |
| `cpu_load_pct` | CPU load of the site equipment |
| `unit_temperature_c` | Equipment temperature in Celsius |
| `supply_voltage_v` | Supply voltage |
| `uplink_interference_dbm` | Uplink interference level |

---

## 5. Fault Types Injected

Six different network fault archetypes are injected into the dataset.

| Fault Type | Fault Name | Description |
|---|---|---|
| F1 | RF Unit Degradation | Signal quality slowly decreases while drop rate and retransmissions increase |
| F2 | Clock Sync Drift | Handover success slowly degrades while other KPIs remain almost normal |
| F3 | Power Supply Instability | Voltage oscillates and CPU/temperature become unstable |
| F4 | Sleeping Cell | Traffic and active users collapse while quality KPIs may still look healthy |
| F5 | Interference Burst | Uplink interference suddenly increases and affects quality and success rates |
| F6 | Capacity Overload | Traffic surge causes latency, CPU load, and drop rate to increase |

Each fault affects multiple KPIs together instead of creating a single unrealistic spike.

---

## 6. Methodology

The complete workflow is shown below:

```text
Configuration
      ↓
Synthetic Data Generation
      ↓
Fault Injection
      ↓
Training Window Selection
      ↓
Unsupervised Anomaly Detection
      ↓
Alert Thresholding
      ↓
Detected Event Creation
      ↓
Root Cause Attribution
      ↓
Final Evaluation
      ↓
Plots and Report
```

---

## 7. Detection Approach

The detection system is unsupervised. This means the model does not use fault labels during training.

The model is trained only on the first 10 days of data. The remaining days are used for evaluation.

The detection approach combines three methods:

| Method | Purpose |
|---|---|
| Isolation Forest | Detects unusual KPI patterns |
| PCA Reconstruction Error | Detects broken relationships between KPIs |
| Robust Z-Score Baseline | Detects simple extreme KPI deviations |

The final anomaly score is calculated by combining these three scores.

This approach was used because a single method may not detect all types of faults. Some faults are sudden, some are gradual, and some affect only a few KPIs.

---

## 8. Leak-Free Design

The project follows a leak-free evaluation design.

The detection code does not read the ground truth labels.

| File/Folder | Usage |
|---|---|
| `data/telemetry.csv` | Used by the detection system |
| `ground_truth/fault_events.csv` | Used only by the evaluation script |
| `code/train_detect.py` | Does not access ground truth |
| `code/evaluate.py` | Reads ground truth only for final scoring |

This ensures that the model is not trained or tuned using the answers.

---

## 9. Thresholding Strategy

Each cell site has its own anomaly threshold.

This is important because different sites behave differently. For example, an urban site may naturally have higher traffic than a rural site.

The threshold is selected using only training data. The goal is to keep the number of false alarms within an acceptable limit.

---

## 10. Root Cause Attribution

After detecting an event, the system identifies the top KPIs responsible for the anomaly.

For every detected event, the system outputs:

- Site ID
- Event start time
- Event end time
- Top 3 responsible KPIs
- Predicted fault type

Example interpretation:

| Observed Pattern | Predicted Fault |
|---|---|
| Interference increases and signal quality drops | F5 Interference Burst |
| Traffic and active users collapse | F4 Sleeping Cell |
| Voltage becomes unstable with temperature and CPU changes | F3 Power Supply Instability |
| Handover success slowly decreases | F2 Clock Sync Drift |

The root cause mapping is rule-based and easy to understand.

---

## 11. Evaluation Metrics

The system is evaluated using both event-level and point-level metrics.

| Metric | Meaning |
|---|---|
| Event Precision | Out of all detected events, how many were real faults |
| Event Recall | Out of all real faults, how many were detected |
| Event F1 | Balance between event precision and event recall |
| Point-Level F1 | Timestamp-level detection quality |
| Point-Adjust F1 | Relaxed score that gives credit if any part of an event is detected |
| Detection Delay | Time taken to detect a fault after it starts |
| False Alarms per Site per Day | Number of false alerts generated per site per day |
| Attribution Hit Rate | Whether the top-3 KPIs include a truly affected KPI |

Point-adjust F1 is reported separately because it can give an overly positive result for long fault windows. Event-level metrics and false alarm rate are more useful for real network operations.

---

## 12. Output Files

| File | Description |
|---|---|
| `data/telemetry.csv` | Generated synthetic network telemetry dataset |
| `ground_truth/fault_events.csv` | True injected fault event labels |
| `ground_truth/fault_schedule.json` | Configurable fault schedule |
| `data/scores.csv` | Anomaly scores and alerts for each timestamp |
| `data/detected_events.csv` | Detected events with root cause attribution |
| `data/thresholds.json` | Site-wise alert thresholds |
| `data/metrics.json` | Final evaluation results |
| `data/per_archetype_metrics.csv` | Fault-wise performance summary |
| `data/point_eval_labels.csv` | Timestamp-level evaluation data |
| `plots/` | Generated plots for analysis |
| `report.pdf` | Final project report |

---

## 13. How to Run the Project

### Step 1: Clone the repository

```bash
git clone https://github.com/Siddhartha1712/Network_telementry_fault_Detection.git
cd Network_telementry_fault_Detection
```

### Step 2: Create a virtual environment

```bash
python -m venv .venv
```

### Step 3: Activate the virtual environment

For Windows:

```bash
.venv\Scripts\activate
```

For Linux/Mac:

```bash
source .venv/bin/activate
```

### Step 4: Install the required libraries

```bash
pip install -r requirements.txt
```

### Step 5: Run the complete pipeline

```bash
python code/run_all.py
```

This command will:

1. Generate synthetic telemetry data
2. Inject fault events
3. Train the anomaly detector
4. Produce anomaly scores
5. Generate detected events
6. Perform root cause attribution
7. Evaluate the results
8. Generate plots

---

## 14. Requirements

The main Python libraries used are:

```text
numpy
pandas
scikit-learn
matplotlib
pyyaml
```

Install all dependencies using:

```bash
pip install -r requirements.txt
```

---

## 15. Reproducibility

The project uses a fixed random seed in the configuration file.

This allows the same dataset and results to be regenerated.

The configuration file is located at:

```text
code/config.yaml
```

Important configurable items include:

- random seed
- number of sites
- number of days
- missing value rate
- number of fault events
- training and evaluation split

---

## 16. Important Design Decisions

### Why synthetic data?

The assignment does not provide external data. Therefore, the dataset is generated synthetically based on realistic telecom KPI behaviour.

### Why unsupervised detection?

In real operations, labelled fault data may be limited or unavailable. An unsupervised detector can learn normal behaviour and flag unusual patterns without requiring labelled examples.

### Why multiple models?

Different faults behave differently. A combined model is more flexible than relying on a single detection method.

### Why root cause attribution?

A simple alert is not enough for a network operations team. The system should also explain which KPIs were responsible for the alert and what type of issue it may represent.

---

## 17. Limitations

This project has some limitations:

- The dataset is synthetic and may not capture every real-world telecom condition.
- Rule-based root cause mapping may not handle mixed or overlapping faults perfectly.
- Subtle faults such as clock sync drift can be difficult to detect early.
- Capacity overload may look similar to genuine high-demand traffic.
- The system does not use external context such as maintenance logs, weather, or local events.
- Neighbouring-site relationships are not deeply modelled.

---

## 18. Future Improvements

With more time, the following improvements can be made:

- Add forecasting-based anomaly detection
- Improve detection of gradual faults
- Add neighbouring-site comparison
- Add adaptive thresholds
- Include event severity scoring
- Improve root cause classification
- Add dashboard visualization
- Simulate maintenance tickets and operator feedback
- Test the pipeline with real telecom telemetry data

---

## 19. Conclusion

This project demonstrates a complete workflow for network telemetry fault detection.

It creates realistic synthetic telecom data, injects multiple types of faults, detects abnormal behaviour using unsupervised learning, attributes the likely root cause, and evaluates the results using clear metrics.

The main focus of the project is not only detecting anomalies, but also making the output explainable and useful for a network operations team.

---

## Author

**Siddhartha1712**

Repository: `Network_telementry_fault_Detection`
