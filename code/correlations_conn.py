import pingouin as pg
import numpy as np
import pandas as pd
from scipy.io import loadmat
import os
from scipy import stats


def load_matrices(mat_files, path):
    """Load connectivity matrices from .mat files and return z-scored arrays and region names."""
    functional_connectivity = "Z"
    names_key = "names"
    names2_key = "names2"

    matrix_zscore = []
    names, names2 = None, None

    for file in mat_files:
        file_path = os.path.join(path, file)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file} was not found in path {path}.")

        mat_data = loadmat(file_path)
        if functional_connectivity not in mat_data:
            raise KeyError(f"Key '{functional_connectivity}' not found in {file}.")

        functional_matrix = stats.zscore(mat_data[functional_connectivity], nan_policy='omit')
        matrix_zscore.append(np.array(functional_matrix))

        if names is None and names_key in mat_data:
            names = [str(name[0]) for name in mat_data[names_key].flatten()]
        if names2 is None and names2_key in mat_data:
            names2 = [str(name[0]) for name in mat_data[names2_key].flatten()]

    return matrix_zscore, names, names2


def save_combined_matrices_to_excel(matrices, names, output_prefix, output_path):
    """Save combined region-pair connectivity values across subjects to Excel files.

    Each file contains rows for subjects and conditions for a specific region pair.
    """
    num_regions = len(names)

    for i in range(num_regions):
        for j in range(i + 1, num_regions):
            data = []

            for condition, matrix in enumerate(matrices):
                region_values = matrix[i, j, :]

                for subject_idx, connectivity_value in enumerate(region_values):
                    data.append({
                        'Subject': f'Subject_{subject_idx + 1}',
                        'Condition': f'Condition_{condition + 1}',
                        'Connectivity': connectivity_value
                    })

            df = pd.DataFrame(data)
            file_name = f"{output_prefix}_Region_{i+1}_vs_Region_{j+1}.xlsx"
            output_file = os.path.join(output_path, file_name)
            df.to_excel(output_file, index=True)


def calculate_clinical_correlations(clinical_df, generated_df, file_name):
    """Calculate repeated-measures correlations between connectivity and clinical variables.

    Returns a DataFrame with significant results (p < 0.05) and FDR-corrected p-values.
    """
    results = []
    combined_df = pd.merge(generated_df, clinical_df, on='Subject', how='left')

    for clinical_variable in clinical_df.columns:
        if clinical_variable != 'Subject':
            try:
                rmcorr_results = pg.rm_corr(
                    data=combined_df,
                    x='Connectivity',
                    y=clinical_variable,
                    subject='Subject'
                )

                region_name = os.path.basename(file_name).replace('.xlsx', '')
                pval = float(rmcorr_results['pval'].iloc[0])

                if pval < 0.05:
                    results.append({
                        'Region': region_name,
                        'Clinical_Variable': clinical_variable,
                        'R': rmcorr_results['r'].iloc[0],
                        'CI95%': rmcorr_results['CI95%'].iloc[0],
                        'P-value': rmcorr_results['pval'].iloc[0]
                    })

            except ValueError as e:
                print(f"Error processing {file_name} with variable {clinical_variable}: {e}")

    results_df = pd.DataFrame(results)

    if len(results_df) > 0:
        rejected, p_values_corrected = pg.multicomp(results_df['P-value'].values, method='fdr_bh')
        results_df['P-value_corrected'] = p_values_corrected
        results_df['Significant'] = rejected

    return results_df


if __name__ == '__main__':
    # Standardized input/output paths
    input_path = '/input'
    output_path = '/output'

    os.makedirs(output_path, exist_ok=True)

    # Read clinical variables from input folder (expected filename Clinical_results.xlsx)
    clinical_data_path = os.path.join(input_path, 'Clinical_results.xlsx')
    clinical_df = pd.read_excel(clinical_data_path)

    results = pd.DataFrame()

    # Iterate over generated combined matrix files in the output folder
    for file_name in os.listdir(output_path):
        if file_name.startswith('combined_matrix') and file_name.endswith('.xlsx'):
            file_path = os.path.join(output_path, file_name)
            generated_df = pd.read_excel(file_path, index_col=0)
            results = pd.concat([results, calculate_clinical_correlations(clinical_df, generated_df, file_name)])

    # Save aggregated results
    results.to_excel(os.path.join(output_path, 'all_correlations_results.xlsx'), index=False)


