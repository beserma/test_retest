# Test-Retest Results — Code & Data

**Author:** Maria Beser · maria_beser@iislafe.es

Repository containing scripts, processed imaging data, and clinical/questionnaire 
data for test–retest reliability analyses (ICC, connectivity stability, 
signal quality metrics, and brain–behaviour correlations).

---

## ⚠️ Data Availability

The following data are available in this repository under `/input`:

| File / Folder | Contents | Status |
|---|---|---|
| `conn_project01/results/firstlevel/SBC_01/` | Processed fMRI connectivity matrices (CONN toolbox output, per condition) | ✅ Available |
| `Clinical_results.xlsx` | Per-subject clinical and psychometric scores (test & retest sessions) | ✅ Available |
| `Raw_Clinical_results.xlsx` | Raw behavioural/questionnaire data | ✅ Available |

> Raw fMRI DICOM/NIfTI files are not included due to patient data privacy 
> constraints (GDPR). Access can be requested from the corresponding author.

---

## Summary

- **Purpose:** Collection of tools and scripts used to compute connectivity 
  consistency, ICC, signal quality metrics, and related analyses for 
  test–retest studies.
- **Language:** Python

---

## Repository Structure
```
/
├── input/                              ← Place all input data here
│   ├── conn_project01/
│   │   └── results/firstlevel/SBC_01/
│   │       ├── resultsROI_Condition002.mat
│   │       ├── resultsROI_Condition003.mat
│   │       ├── resultsROI_Condition004.mat
│   │       ├── _list_conditions.txt
│   │       └── _list_sources.mat
│   └── Clinical_results.xlsx           ← Clinical & questionnaire data
│
├── output/                             ← Generated tables, figures, results
│
├── calculate_signal_quality_metrics.py
├── compare_icc_age_correction.py
├── connectivity_stability.py
├── correlations_conn.py
├── generate_icc_table_clean.py
├── heatmap.py
├── icc_conn.py
├── icc_figure.py
├── repeated_measures_conn.py
├── session_consistency_analysis.py
├── variance_decomposition_analysis.py
└── README.md
```

### Script descriptions

| Script | Description |
|---|---|
| `calculate_signal_quality_metrics.py` | Compute signal quality metrics from CONN ROI files |
| `compare_icc_age_correction.py` | Compare ICC with/without age correction |
| `connectivity_stability.py` | Kendall's W, Frobenius distances, Procrustes disparities |
| `correlations_conn.py` | Correlations between connectivity matrices and clinical data |
| `generate_icc_table_clean.py` | Generate and clean ICC tables |
| `heatmap.py` | Heatmap visualisations |
| `icc_conn.py` | ICC calculations for connectivity matrices |
| `icc_figure.py` | ICC-related figures and plots |
| `repeated_measures_conn.py` | Repeated-measures connectivity analyses |
| `session_consistency_analysis.py` | Session-to-session consistency evaluation |
| `variance_decomposition_analysis.py` | Variance decomposition analyses |

---

## Setup
```bash
python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate  # macOS/Linux
pip install --upgrade pip
pip install numpy pandas scipy matplotlib seaborn statsmodels pingouin openpyxl
```

---

## Recommended Execution Order

1. `calculate_signal_quality_metrics.py`
2. `generate_icc_table_clean.py`
3. `icc_conn.py`
4. `compare_icc_age_correction.py`
5. `connectivity_stability.py`
6. `correlations_conn.py`
7. `repeated_measures_conn.py`
8. `negative_control_brain_behavior.py`
9. `icc_figure.py` and `heatmap.py`
10. `session_consistency_analysis.py`, `variance_decomposition_analysis.py` 
    (flexible order)

---

## Contact

For questions about the data or scripts, open a GitHub issue or contact 
the corresponding author: maria_beser@iislafe.es

## License

MIT License (to be added).
