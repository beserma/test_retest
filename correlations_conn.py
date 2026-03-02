import pingouin as pg
import numpy as np
import pandas as pd
from scipy.io import loadmat
import os
from scipy.stats import pearsonr
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection

# Función para cargar matrices desde archivos .mat
def load_matrices(mat_files, path):
    functional_connectivity = "Z"
    names_key = "names"
    names2_key = "names2"

    matrix_zscore = []
    names, names2 = None, None

    for file in mat_files:
        file_path = os.path.join(path, file)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"El archivo {file} no se encontró en la ruta {path}.")

        mat_data = loadmat(file_path)
        if functional_connectivity not in mat_data:
            raise KeyError(f"La clave '{functional_connectivity}' no se encontró en {file}.")
        
        functional_matrix = stats.zscore(mat_data[functional_connectivity], nan_policy='omit')
        matrix_zscore.append(np.array(functional_matrix))

        if names is None and names_key in mat_data:
            names = [str(name[0]) for name in mat_data[names_key].flatten()]
        if names2 is None and names2_key in mat_data:
            names2 = [str(name[0]) for name in mat_data[names2_key].flatten()]

    return matrix_zscore, names, names2

# Guardar matrices como archivos Excel por combinación de regiones y todos los sujetos
def save_combined_matrices_to_excel(matrices, names, output_prefix, output_path):
    num_regions = len(names)

    # Iterar sobre todas las combinaciones de regiones (triangular inferior)
    for i in range(num_regions):
        for j in range(i + 1, num_regions):  # Solo recorrer la parte triangular superior y evitar duplicados
            # Crear una lista para almacenar las correlaciones de todos los sujetos
            data = []

            for condition, matrix in enumerate(matrices):
                region_values = matrix[i, j, :]
                
                # Crear filas para cada sujeto y condición
                for subject_idx, connectivity_value in enumerate(region_values):
                    data.append({
                        'Subject': f'Subject_{subject_idx + 1}',
                        'Condition': f'Condition_{condition + 1}',
                        'Connectivity': connectivity_value
                    })

                # Crear un DataFrame a partir de los datos
                df = pd.DataFrame(data)
            # Nombre del archivo Excel
            file_name = f"{output_prefix}_Region_{i+1}_vs_Region_{j+1}.xlsx"
            output_file = os.path.join(output_path, file_name)

            # Guardar el DataFrame en un archivo Excel
            df.to_excel(output_file, index=True)

def calculate_clinical_correlations(clinical_df, generated_df, file_name):
    results = []
    combined_df = pd.merge(generated_df, clinical_df, on='Subject', how='left')

    for clinical_variable in clinical_df.columns:
        if clinical_variable != 'Subject':
            try:
                # Calcular rm_corr
                rmcorr_results = pg.rm_corr(
                    data=combined_df,
                    x='Connectivity',
                    y=clinical_variable,
                    subject='Subject'
                )

                # Extraer el nombre de la región
                region_name = os.path.basename(file_name).replace('.xlsx', '')
                pval=float(rmcorr_results['pval'].iloc[0])
                
                if pval<0.05:
                    # Agregar los resultados
                    results.append({
                        'Region': region_name,
                        'Clinical_Variable': clinical_variable,
                        'R': rmcorr_results['r'].iloc[0],
                        'CI95%': rmcorr_results['CI95%'].iloc[0],
                        'P-value': rmcorr_results['pval'].iloc[0]
                    })

            except ValueError as e:
                print(f"Error en {file_name} con variable {clinical_variable}: {e}")

    # Convertir los resultados a un DataFrame
    results_df = pd.DataFrame(results)

    # Aplicar corrección FDR
    if len(results_df)>0:
        rejected, p_values_corrected = pg.multicomp(results_df['P-value'].values, method='fdr_bh')
        results_df['P-value_corrected'] = p_values_corrected
        results_df['Significant'] = rejected

    return results_df

# Configuración de archivos y directorios
output_path = "/excel_outputs"

# Validar directorio de salida
os.makedirs(output_path, exist_ok=True)

# Cargar matrices
#matrix_zscore, names, names2 = load_matrices(mat_files, path)
results=pd.DataFrame()
# Guardar matrices combinadas como archivos Excel
#save_combined_matrices_to_excel(matrix_zscore, names, "combined_matrix", output_path)

# Leer variables clínicas
clinical_data_path = os.path.join(output_path, 'Clinical_data.xlsx')
clinical_df = pd.read_excel(clinical_data_path)

# Iterar sobre cada condición para procesar y guardar resultados
for index, file_name in enumerate(os.listdir(output_path)):
    if file_name.startswith('combined_matrix') and file_name.endswith('.xlsx'):
        file_path = os.path.join(output_path, file_name)
        generated_df = pd.read_excel(file_path, index_col=0)
        results = pd.concat([results, calculate_clinical_correlations(clinical_df, generated_df, file_name)])

# Guardar el DataFrame completo con todos los resultados acumulados
results.to_excel(os.path.join(output_path, 'all_correlations_results.xlsx'), index=False)


