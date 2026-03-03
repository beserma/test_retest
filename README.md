**Author** Maria Beser maria_beser@iisalfe.es

**Test-Retest Results — Code**

Repository containing scripts for test–retest analyses (reliability, ICC, correlations and signal quality metrics).

**Summary**
- **Purpose:** Collection of tools and scripts used to compute connectivity consistency, ICC, signal quality metrics, and related analyses for test–retest studies.
- **Language:** Python

**Project Structure**
- `calculate_signal_quality_metrics.py`: Compute signal quality metrics.
- `compare_icc_age_correction.py`: Compare ICC with/without age correction.
- `connectivity_stability.py`: Connectivity stability analysis.
- `correlations_conn.py`: Correlations between connectivity matrices and clinical data.
- `generate_icc_table_clean.py`: Generate and clean ICC tables.
- `heatmap.py`: Heatmap visualizations.
- `icc_conn.py`: ICC calculations for connectivity.
- `icc_figure.py`: ICC-related figures and plots.
- `repeated_measures_conn.py`: Repeated-measures connectivity analyses.
- `session_consistency_analysis.py`: Session-to-session consistency evaluation.
- `variance_decomposition_analysis.py`: Variance decomposition analyses.

**Paths and I/O**
- This repository standardizes input and output folders at the project root as `/input` and `/output`.
- Place raw input files (MAT files, Excel, clinical data) under `/input`.
- Generated tables, figures and processed results will be written to `/output`.

**Input Folder Contents**
- Expected structure under `/input` (CONN toolbox results):
	- `conn_project01\results\firstlevel\SBC_01` — directory containing per-condition .mat files
		- `resultsROI_Condition002.mat`, `resultsROI_Condition003.mat`, `resultsROI_Condition004.mat` (one file per condition)
		- `_list_conditions.txt` (text file listing condition names/order)
		- `_list_sources.mat` (optional source metadata from CONN)
	- `Clinical_results.xlsx` (clinical/demographic spreadsheet; script examples expect an Excel file with subjects in rows)

Place the entire `conn_project01` tree (or its `results\firstlevel\SBC_01` folder) inside `/input` so scripts can load the MAT files and Excel files using the standardized `input_path`.

Notes:
- Filenames are case-sensitive on non-Windows systems; keep the file names consistent.
- If your files use different names, either rename them or update the script top-level `input_path` and filenames accordingly.

**Suggested Requirements**
- Python 3.8 or newer
- Typical packages: numpy, pandas, scipy, matplotlib, seaborn, statsmodels, pingouin
- Example setup:

```bash
python -m venv venv
venv\Scripts\activate    # Windows
pip install --upgrade pip
pip install numpy pandas scipy matplotlib seaborn statsmodels pingouin
```

Add additional packages (e.g., nibabel, nilearn, openpyxl) as needed.

**Basic Usage**
- Run a script directly with Python (examples assume you are in the repository root):

```bash
python compare_icc_age_correction.py
```

- Many scripts accept arguments or expect files in `/input`; check the top of each script for specific filenames.

**Recommended Workflow**
- Create and activate a virtual environment for the project.
- Keep raw data under `/input` and outputs under `/output` (do not commit large data files to the repo).
- Add a `requirements.txt` or `pyproject.toml` to pin package versions for reproducibility.

**Execution Order (recommended)**
Run scripts in the following order to reproduce analyses and outputs. Some scripts are independent and can be run in parallel once their inputs exist.

1. `calculate_signal_quality_metrics.py` — extract signal-quality metrics from CONN ROI files and save amplitude summaries to `/output`.
2. `generate_icc_table_clean.py` — generate cleaned ICC tables from the CONN results (.mat files).
3. `icc_conn.py` — compute intraclass correlations (ICC) across conditions; writes ICC tables to `/output`.
4. `compare_icc_age_correction.py` — run age-correction/residualization and compare adjusted vs unadjusted ICCs; saves Supplementary tables and figures to `/output`.
5. `connectivity_stability.py` — compute global stability metrics (Kendall's W, Frobenius distances, Procrustes disparities).
6. `correlations_conn.py` — produce subject-wise connectivity Excel files and compute correlations with clinical data in `Clinical_results.xlsx`.
7. `repeated_measures_conn.py` — repeated-measures correlations and rm_corr analyses between FC and psychological variables.
8. `negative_control_brain_behavior.py` — negative-control analyses comparing brain–behavior correlations for high-ICC vs low-ICC connections; saves figures and Excel outputs to `/output`.
9. `icc_figure.py` and `heatmap.py` — generate plots and heatmaps from previously saved ICC/tables in `/output`.
10. `session_consistency_analysis.py`, `variance_decomposition_analysis.py`, `test_rmcorr_assumptions.py` — run these diagnostics and additional analyses as needed (order is flexible, they rely on previously generated data).

If you modify filenames or change how inputs are organized, update the `input_path`/`output_path` variables at the top of each script accordingly.

**Contributing**
- Open an issue to discuss changes before submitting pull requests.
- Add minimal examples or tests when introducing new functionality.

**License**
- Add your preferred license (e.g., MIT). No license is currently specified.

**Contact**
- For questions or improvements, contact the project author or open an issue.

--
This README is an initial template. Update the requirements and execution examples according to each script's actual parameters.
