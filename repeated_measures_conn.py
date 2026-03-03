import numpy as np
import pandas as pd
from scipy.io import loadmat
from itertools import combinations_with_replacement
import os
from scipy import stats
from statsmodels.stats.multitest import multipletests
import pingouin as pg

def load_matrices(mat_files, path):
    """Load functional connectivity matrices"""
    names_key = "names"
    names2_key = "names2"
    functional_connectivity = "Z"
    
    matrix_zscore = []
    matrix = []
    names = None
    names2 = None

    for file in mat_files:
        mat_data = loadmat(os.path.join(path, file))
        if functional_connectivity in mat_data:
            functional_matrix = stats.zscore(mat_data[functional_connectivity], nan_policy='omit')
            matrix_zscore.append(np.array(functional_matrix))
            matrix.append(np.array(mat_data[functional_connectivity]))
        else:
            raise KeyError(f"Key '{functional_connectivity}' not found in {file}.")
        
        # Cargar nombres solo una vez
        if names is None and names_key in mat_data:
            names = [str(name[0]) for name in mat_data[names_key].flatten()]
        if names2 is None and names2_key in mat_data:
            names2 = [str(name[0]) for name in mat_data[names2_key].flatten()]
    
    return matrix_zscore, matrix, names, names2

def load_psychological_variables(excel_path):
    """Load psychological variables from an Excel file"""
    try:
        # Intentar leer el archivo Excel
        df_psych = pd.read_excel(excel_path)
        print(f"Psychological variables loaded: {df_psych.columns.tolist()}")
        return df_psych
    except Exception as e:
        print(f"Error loading Excel file: {e}")
        return None

def correlate_fc_psychology_repeated_measures(matrix_list, psych_data, names, names2, 
                                           significance_threshold=0.05, min_correlation=0.5):
    """
    Compute repeated-measures correlations between functional connectivity and psychological variables
    using pingouin.rm_corr().
    """
    n_regions, _, n_subjects = matrix_list[0].shape
    n_conditions = len(matrix_list)
    
    # Define regions to exclude due to instability
    excluded_regions = [
        'networks.DefaultMode.MPFC (1,55,-3)',
        'networks.Visual.Medial (2,-79,12)',
        'networks.Visual.Occipital (0,-93,-4)',
        'networks.Visual.Lateral (L) (-37,-79,10)',
        'networks.Visual.Lateral (R) (38,-72,13)',
        'networks.DorsalAttention.FEF (L)  (-27,-9,64)',
        'networks.DorsalAttention.FEF (R)  (30,-6,64)',
        'networks.Cerebellar.Anterior (0,-63,-30)',
        'networks.Cerebellar.Posterior (0,-79,-32)'
    ]
    
    print(f"Excluded regions due to instability: {len(excluded_regions)}")
    for region in excluded_regions:
        print(f"  - {region}")
    
    # Verify matching number of subjects
    if len(psych_data) != n_subjects:
        print(f"WARNING: Number of connectivity subjects ({n_subjects}) does not match psychological data ({len(psych_data)})")
        # Tomar el mínimo común
        n_subjects_common = min(len(psych_data), n_subjects)
        psych_data = psych_data.iloc[:n_subjects_common]
        matrix_list = [mat[:, :, :n_subjects_common] for mat in matrix_list]
        n_subjects = n_subjects_common
    
    # Identificar variables psicológicas (excluir columnas de ID si existen)
    psych_columns = [col for col in psych_data.columns 
                    if not col.lower().startswith(('id', 'subject', 'participant'))]
    
    # Almacenar todos los resultados de correlación
    all_correlations = []
    correlation_pairs = []
    excluded_pairs_count = 0
    
    print(f"Analyzing {n_subjects} subjects with {n_conditions} timepoint conditions...")
    
    # Calcular correlaciones para cada par de regiones
    for i, j in combinations_with_replacement(range(n_regions), 2):
        region_1_name = names[i]
        region_2_name = names2[j]
        
        # FILTRO 1: Solo analizar pares donde ambas regiones empiecen por "networks"
        if not (region_1_name.lower().startswith('networks') and 
                region_2_name.lower().startswith('networks')):
            continue  # Saltar este par de regiones
        
        # FILTRO 2: Excluir regiones inestables
        if (region_1_name in excluded_regions or region_2_name in excluded_regions):
            excluded_pairs_count += 1
            continue  # Saltar este par de regiones
        
        # Crear DataFrame con datos de conectividad funcional para las 3 condiciones
        fc_data = []
        for subject in range(n_subjects):
            for cond_idx, matrix in enumerate(matrix_list):
                fc_data.append({
                    'Subject': subject,
                    'Condition': cond_idx + 1,
                    'FC_Value': matrix[i, j, subject]
                })
        
        fc_df = pd.DataFrame(fc_data)
        
        # Calcular correlaciones con cada variable psicológica
        for psych_var in psych_columns:
            # Crear DataFrame expandido con variable psicológica
            expanded_data = []
            for subject in range(n_subjects):
                psych_value = psych_data.iloc[subject][psych_var]
                for cond_idx in range(n_conditions):
                    fc_value = matrix_list[cond_idx][i, j, subject]
                    expanded_data.append({
                        'Subject': subject,
                        'Condition': cond_idx + 1,
                        'FC_Value': fc_value,
                        'Psych_Value': psych_value
                    })
            
            expanded_df = pd.DataFrame(expanded_data)
            
            # Remover filas con NaN
            expanded_df_clean = expanded_df.dropna()
            
            if len(expanded_df_clean) < 30:
                continue
            
            try:
                # Calcular correlación por medidas repetidas usando pingouin
                rm_corr_result = pg.rm_corr(
                    data=expanded_df_clean,
                    x='FC_Value',
                    y='Psych_Value', 
                    subject='Subject'
                )
                
                correlation = rm_corr_result['r'].iloc[0]
                p_value = rm_corr_result['pval'].iloc[0]
                ci_lower = rm_corr_result['CI95%'].iloc[0][0]
                ci_upper = rm_corr_result['CI95%'].iloc[0][1]
                
                all_correlations.append(p_value)
                correlation_pairs.append({
                    'Region_1': region_1_name,
                    'Region_2': region_2_name,
                    'Psychological_Variable': psych_var,
                    'RM_Correlation': correlation,
                    'P_Value': p_value,
                    'CI_Lower': ci_lower,
                    'CI_Upper': ci_upper,
                    'N_Subjects': len(expanded_df_clean['Subject'].unique())
                })
                
            except Exception as e:
                print(f"Error computing RM correlation for {region_1_name}-{region_2_name}, {psych_var}: {e}")
                continue
    
    print(f"Region pairs excluded due to instability: {excluded_pairs_count}")
    
    # Aplicar corrección FDR a todos los p-values
    if all_correlations:
        rejected, corrected_p_values, _, _ = multipletests(all_correlations, method='fdr_bh')
        
        # Añadir p-values corregidos a los resultados
        for i, result in enumerate(correlation_pairs):
            result['Corrected_P_Value'] = corrected_p_values[i]
            result['Significant_FDR'] = rejected[i]
    
    # Convertir a DataFrame
    results_df = pd.DataFrame(correlation_pairs)
    
    # Filtrar resultados significativos y con correlación mínima
    if len(results_df) > 0:
        significant_results = results_df[
            (results_df['Significant_FDR'] == True) & 
            (np.abs(results_df['RM_Correlation']) >= min_correlation)
        ].copy()
    else:
        significant_results = pd.DataFrame()
    
    return results_df, significant_results

def create_correlation_matrix_format(significant_results, psych_columns):
    """
    Crea una matriz en el formato solicitado (similar al ejemplo)
    """
    # Crear una tabla pivote donde las filas son pares de regiones y las columnas son variables psicológicas
    pivot_data = []
    
    # Obtener todos los pares únicos de regiones
    region_pairs = significant_results[['Region_1', 'Region_2']].drop_duplicates()
    
    for _, row in region_pairs.iterrows():
        region_1 = row['Region_1']
        region_2 = row['Region_2']
        
        # Crear fila de datos
        row_data = {'Region_1': region_1, 'Region_2': region_2}
        
        # Buscar correlaciones para cada variable psicológica
        for psych_var in psych_columns:
            correlation_row = significant_results[
                (significant_results['Region_1'] == region_1) & 
                (significant_results['Region_2'] == region_2) & 
                (significant_results['Psychological_Variable'] == psych_var)
            ]
            
            if not correlation_row.empty:
                correlation = correlation_row['RM_Correlation'].iloc[0]  # Cambio aquí: usar 'RM_Correlation'
                # Formatear a 3 decimales como en el ejemplo
                row_data[psych_var] = f"{correlation:.3f}"
            else:
                row_data[psych_var] = ""  # Celda vacía si no hay correlación significativa
        
        # Solo añadir la fila si tiene al menos una correlación
        if any(val != "" for val in row_data.values() if val not in [region_1, region_2]):
            pivot_data.append(row_data)
    
    return pd.DataFrame(pivot_data)

def main():
    """Función principal"""
    
    # Configuración de archivos y directorio
    path = '/input'
    mat_files = ["resultsROI_Condition002.mat", 
                 "resultsROI_Condition003.mat", 
                 "resultsROI_Condition004.mat"]
    
    # Ruta al archivo de variables psicológicas
    psychology_excel_path = os.path.join(path, "Clinical_results.xlsx")  # Ajustar ruta según sea necesario
    
    print("Cargando matrices de conectividad funcional...")
    # Cargar las matrices de conectividad funcional
    matrix_zscore, matrix_raw, names, names2 = load_matrices(mat_files, path)
    
    print("Cargando variables psicológicas...")
    # Cargar variables psicológicas
    psych_data = load_psychological_variables(psychology_excel_path)
    
    if psych_data is None:
        print("No se pudieron cargar las variables psicológicas. Terminando...")
        return
    
    print("Calculando correlaciones por medidas repetidas...")
    # Calcular correlaciones por medidas repetidas usando pingouin
    all_results, significant_results = correlate_fc_psychology_repeated_measures(
        matrix_zscore, psych_data, names, names2,
        significance_threshold=0.05,
        min_correlation=0.5  # Correlación mínima como en el ejemplo
    )
    
    # Identificar variables psicológicas
    psych_columns = [col for col in psych_data.columns 
                    if not col.lower().startswith(('id', 'subject', 'participant'))]
    
    print("Creando matriz de resultados...")
    # Crear matriz en el formato solicitado
    correlation_matrix = create_correlation_matrix_format(significant_results, psych_columns)
    
    # Guardar resultados
    output_path = path
    
    # Guardar todos los resultados
    all_results.to_excel(os.path.join(output_path, "all_rm_correlations_fc_psychology.xlsx"), index=False)
    
    # Guardar solo resultados significativos
    significant_results.to_excel(os.path.join(output_path, "significant_rm_correlations_fc_psychology.xlsx"), index=False)
    
    # Guardar matriz en formato solicitado
    correlation_matrix.to_excel(os.path.join(output_path, "rm_correlation_matrix_formatted.xlsx"), index=False)
    
    # Mostrar resumen
    print(f"\n=== RESUMEN DE RESULTADOS ===")
    print(f"Total de regiones: {len(names)}")
    
    # Contar regiones que empiezan por "networks"
    networks_regions = [name for name in names if name.lower().startswith('networks')]
    print(f"Regiones 'networks': {len(networks_regions)}")
    
    # Mostrar regiones excluidas
    excluded_regions = [
        'networks.DefaultMode.MPFC (1,55,-3)',
        'networks.Visual.Medial (2,-79,12)',
        'networks.Visual.Occipital (0,-93,-4)',
        'networks.Visual.Lateral (L) (-37,-79,10)',
        'networks.Visual.Lateral (R) (38,-72,13)',
        'networks.DorsalAttention.FEF (L)  (-27,-9,64)',
        'networks.DorsalAttention.FEF (R)  (30,-6,64)',
        'networks.Cerebellar.Anterior (0,-63,-30)',
        'networks.Cerebellar.Posterior (0,-79,-32)'
    ]
    
    # Identificar regiones excluidas que realmente estaban en los datos
    excluded_found = [region for region in excluded_regions if region in networks_regions]
    
    print(f"Regiones excluidas por inestabilidad: {len(excluded_found)}")
    for region in excluded_found:
        print(f"  - {region}")
    
    # Regiones incluidas en el análisis
    included_regions = [region for region in networks_regions if region not in excluded_regions]
    print(f"Regiones 'networks' incluidas en análisis: {len(included_regions)}")
    
    print(f"Total de correlaciones RM calculadas: {len(all_results)}")
    print(f"Correlaciones RM significativas (FDR corrected): {len(significant_results)}")
    print(f"Pares de regiones con correlaciones RM significativas: {len(correlation_matrix)}")
    print(f"Variables psicológicas analizadas: {psych_columns}")
    
    print(f"\nRegiones 'networks' incluidas en el análisis:")
    for region in sorted(set(included_regions)):
        print(f"  - {region}")
    
    print(f"\nArchivos guardados en: {output_path}")
    print("- all_rm_correlations_fc_psychology.xlsx: Todas las correlaciones RM")
    print("- significant_rm_correlations_fc_psychology.xlsx: Solo correlaciones RM significativas")
    print("- rm_correlation_matrix_formatted.xlsx: Matriz RM en formato solicitado")
    
    # Mostrar preview de la matriz formateada
    print(f"\n=== PREVIEW DE MATRIZ FORMATEADA (REPEATED MEASURES) ===")
    print(correlation_matrix.head(10))
    
    print(f"\n=== INFORMACIÓN SOBRE CORRELACIONES POR MEDIDAS REPETIDAS ===")
    print("- Las correlaciones RM controlan por las diferencias entre sujetos")
    print("- Evalúan la asociación estable a través de los 3 momentos temporales") 
    print("- Proporcionan mayor validez para relaciones longitudinales")
    print("- Se excluyeron 9 regiones por falta de estabilidad temporal")
    print("- Solo se analizaron conexiones entre regiones 'networks' estables")
    
    return all_results, significant_results, correlation_matrix

if __name__ == "__main__":
    # Verificar que pingouin esté instalado
    try:
        import pingouin as pg
        print("Pingouin encontrado. Iniciando análisis de correlaciones por medidas repetidas...")
    except ImportError:
        print("ERROR: pingouin no está instalado. Instálalo con: pip install pingouin")
        exit()
    
    all_results, significant_results, correlation_matrix = main()

