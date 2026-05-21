## Prescription Fraud Detection
**Unsupervised Machine Learning | Anomaly Detection in Drug Prescription Records**

## Project Overview
This project detects unusual and potentially fraudulent patterns in Medicare drug prescription records using unsupervised machine learning. With no labelled fraud examples available, the model learns what "normal" prescribing behaviour looks like — and flags prescribers that deviate significantly from that baseline.
The dataset used is the **CMS Medicare Part D Prescribers dataset**, containing over 600,000 prescription records spanning drug costs, claim volumes, beneficiary counts, and prescriber demographics across US states.

## Objectives

- Preprocess and clean large-scale prescription data for modelling
- Engineer meaningful features that capture abnormal prescription behaviour
- Apply unsupervised anomaly detection to surface high-risk prescribers
- Visualise and interpret flagged anomalies for clinical or regulatory review


## Feature Engineering
Several fraud-indicative features were derived from raw prescription data:
| Feature | Fraud Signal |
|---|---|
| Avg_Cost_per_Claim | Inflated billing per claim |
| Cost_per_Beneficiary | Phantom claims or over-billing |
| Claims_per_Beneficiary | Overprescribing behaviour |
| Avg_Days_Supply_per_Claim | Potential drug diversion |
| Cost_per_Day | Abnormally costly daily treatment |

Drug-category flags were also engineered to highlight high-risk prescribing patterns:

- **Opioid Flag** — presence of known opioid drugs (Fentanyl, Oxycodone, Morphine, etc.)
- **Long-Acting Opioid Flag** — subset of opioids with higher diversion risk
- **Antibiotic Flag** — antibiotic overprescribing patterns
- **Antipsychotic Flag** — inappropriate antipsychotic prescribing (especially in 65+ patients)
- **NDC Conflict Flag** — drugs with known National Drug Code conflicts


## Model: Isolation Forest
```python
IsolationForest(
    n_estimators=100,
    contamination=0.01,   # 1% of records flagged as anomalous
    random_state=42,
    n_jobs=-1
)
```
**How it works:** Isolation Forest isolates anomalies by randomly partitioning features. Records that are easier to isolate receive higher anomaly scores — corresponding to prescribers behaving unusually relative to peers.
**Results:**

- Total records analysed: **600,000**
- Flagged as anomalous: **~6,000 (1%)**
- Anomaly scores normalised to [0, 1] for interpretability


## Key Visualisation
A distribution plot of Total Drug Cost was produced comparing normal vs. anomalous prescribers, showing that flagged prescribers cluster at significantly higher drug costs than the general population.

## Sample Output — Top Anomalous Prescribers
The model outputs a ranked list of the most anomalous prescribers, including:

- Prescriber NPI, name, city, and state
- Drug names (generic and brand)
- Total claims, drug cost, and beneficiary count
- Drug category flags (opioid, antibiotic, antipsychotic, NDC conflict)
- Normalised anomaly score

