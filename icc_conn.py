import pingouin as pg

import numpy as np
import pandas as pd
from scipy.io import loadmat
from itertools import combinations_with_replacement
import os
from statsmodels.stats.anova import AnovaRM
from scipy import stats
from statsmodels.stats.multitest import multipletests
from scipy.stats import ttest_rel
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as smf
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Carga de matrices
def load_matrices(mat_files, path):
    names_key = "names"
    names2_key = "names2"
    functional_connectivity = "Z"
    
    matrix_zscore = []
    matrix=[]
    names = None
    names2 = None

    for file in mat_files:
        mat_data = loadmat(os.path.join(path, file))
        if functional_connectivity in mat_data:
            functional_matrix=stats.zscore(mat_data[functional_connectivity], nan_policy='omit')
            matrix_zscore.append(np.array(functional_matrix))
            matrix.append(np.array(mat_data[functional_connectivity]))
        else:
            raise KeyError(f"La clave '{functional_connectivity}' no se encontró en {file}.")
        
        # Cargar nombres solo una vez (asumimos que son consistentes entre condiciones)
        if names is None and names_key in mat_data:
            names = [str(name[0]) for name in mat_data[names_key].flatten()]
        if names2 is None and names2_key in mat_data:
            names2 = [str(name[0]) for name in mat_data[names2_key].flatten()]
    
    return matrix_zscore, matrix, names, names2

# Configuración de archivos y directorio
path = 'Z:\\mnt\\rimp\\PROJECTS\\TEST-RETEST\\Conectividad funcional\\conn_project01\\results\\firstlevel\\SBC_01'
mat_files = ["resultsROI_Condition002.mat",
             "resultsROI_Condition003.mat",
            "resultsROI_Condition004.mat"]

# Cargar las matrices de conectividad funcional y nombres
matrix_zscore, matrix_raw, names, names2 = load_matrices(mat_files, path)

# Cargar datos de edad desde el archivo Excel
age_file = os.path.join(path, 'Base_María_IRI_RPQ.xlsx')
age_df = pd.read_excel(age_file)
ages = age_df['Age'].values  # Array con las edades para cada sujeto

# Verificar consistencia de las dimensiones
n_regions, _, n_subjects = matrix_zscore[0].shape
assert len(names) == n_regions and len(names2) == n_regions, "Dimensiones de nombres y matriz no coinciden."
assert len(ages) == n_subjects, "El número de edades no coincide con el número de sujetos."

# Preparar los datos para ANOVA de medidas repetidas
data = []
for subject in range(n_subjects):
    for i, j in combinations_with_replacement(range(n_regions), 2):
        for cond_idx, mat in enumerate(matrix_zscore):  # Iterar sobre las condiciones (matrices)
            data.append({
                'Subject': subject,
                'Region_1': names[i],
                'Region_2': names2[j],
                'Condition': f'Condition_{cond_idx+1}',
                'Connectivity': mat[i, j, subject],  # Conectividad entre regiones (i, j) para este sujeto
                'Age': ages[subject]  # Añadir edad para este sujeto
            })

df_zscore = pd.DataFrame(data)

# Función para residualizar la conectividad respecto a la edad
def residualize_by_age(group):
    """Residualiza los valores de conectividad removiendo el efecto de la edad."""
    # Crear una máscara para valores válidos (no NaN)
    valid_mask = ~group['Connectivity'].isna()
    
    if valid_mask.sum() < 2:  # Necesitamos al menos 2 puntos para ajustar
        return group
    
    X = group.loc[valid_mask, 'Age'].values.reshape(-1, 1)
    y = group.loc[valid_mask, 'Connectivity'].values
    
    # Ajustar modelo lineal de edad -> conectividad
    model = LinearRegression()
    model.fit(X, y)
    
    # Calcular residuos solo para valores válidos
    y_pred = model.predict(X)
    residuals = y - y_pred
    
    group = group.copy()
    group.loc[valid_mask, 'Connectivity'] = residuals
    return group
# Aplicar residualización por edad a cada par de regiones (no sobreescribimos df_zscore original)
#filtrar solo pares de regiones donde ambos nombres empiezan por 'networks' antes de residualizar
df_zscore_filtered = df_zscore[(df_zscore['Region_1'].str.startswith('networks')) & (df_zscore['Region_2'].str.startswith('networks'))]

df_zscore_residualized_filtered = df_zscore_filtered.groupby(['Region_1', 'Region_2'], group_keys=False).apply(residualize_by_age)

# =====================================================================
# CÁLCULO DE ICC SIN CORRECCIÓN POR EDAD (solo 'networks')
# =====================================================================
icc_results_no_correction = {}

for (region_1, region_2), group in df_zscore_filtered.groupby(['Region_1', 'Region_2']):
    if region_1 != region_2:
        icc = pg.intraclass_corr(data=group, targets='Subject', raters='Condition', ratings='Connectivity', nan_policy='omit')
        icc_results_no_correction[(region_1, region_2)] = icc

if not df_zscore_filtered.empty:
    global_icc_no_correction = pg.intraclass_corr(data=df_zscore_filtered, targets='Subject', raters='Condition', ratings='Connectivity', nan_policy='omit')
    print("ICC sin corrección por edad (networks):")
    print(global_icc_no_correction)
else:
    print("No hay pares 'networks' en los datos sin corrección.")

# Crear un DataFrame con los resultados del ICC sin corrección
icc_results_df_no_correction = pd.DataFrame()

for (region_1, region_2), value in icc_results_no_correction.items():
    icc_value = value.iloc[4]['ICC']
    ci_lower = value.iloc[4]['CI95%'][0]
    ci_upper = value.iloc[4]['CI95%'][1]
    pval = value.iloc[4]['pval']
    
    temp_df = pd.DataFrame({
        'Region_1': [region_1],
        'Region_2': [region_2],
        'ICC': [icc_value],
        'CI Lower': [ci_lower],
        'CI Upper': [ci_upper],
        'p-val': [pval]
    })
    
    icc_results_df_no_correction = pd.concat([icc_results_df_no_correction, temp_df], ignore_index=True)

icc_results_df_no_correction = icc_results_df_no_correction.replace([np.inf, -np.inf], np.nan).fillna("N/A")
icc_results_df_no_correction.to_excel(os.path.join(path, 'icc_results_without_age_correction_networks.xlsx'), index=False)
print("Los resultados del ICC SIN corrección (networks) se han guardado en 'icc_results_without_age_correction_networks.xlsx'")

# =====================================================================
# CÁLCULO DE ICC CON CORRECCIÓN POR EDAD (solo 'networks')
# =====================================================================
icc_results = {}

for (region_1, region_2), group in df_zscore_residualized_filtered.groupby(['Region_1', 'Region_2']):
    if region_1 != region_2:
        icc = pg.intraclass_corr(data=group, targets='Subject', raters='Condition', ratings='Connectivity', nan_policy='omit')
        icc_results[(region_1, region_2)] = icc

if not df_zscore_residualized_filtered.empty:
    global_icc = pg.intraclass_corr(data=df_zscore_residualized_filtered, targets='Subject', raters='Condition', ratings='Connectivity', nan_policy='omit')
    print("\nICC con corrección por edad (networks):")
    print(global_icc)
else:
    print("No hay pares 'networks' en los datos con corrección.")

# Crear un DataFrame con los resultados del ICC (filtrado y residualizado)
icc_results_df1 = pd.DataFrame()

for (region_1, region_2), value in icc_results.items():
    icc_value = value.iloc[4]['ICC']
    ci_lower = value.iloc[4]['CI95%'][0]
    ci_upper = value.iloc[4]['CI95%'][1]
    pval = value.iloc[4]['pval']
    
    temp_df = pd.DataFrame({
        'Region_1': [region_1],
        'Region_2': [region_2],
        'ICC': [icc_value],
        'CI Lower': [ci_lower],
        'CI Upper': [ci_upper],
        'p-val': [pval]
    })
    
    icc_results_df1 = pd.concat([icc_results_df1, temp_df], ignore_index=True)

icc_results_df1 = icc_results_df1.replace([np.inf, -np.inf], np.nan).fillna("N/A")
icc_results_df1.to_excel(os.path.join(path, 'icc_results_with_age_correction_networks.xlsx'), index=False)
print("Los resultados del ICC CON corrección (networks) se han guardado en 'icc_results_with_age_correction_networks.xlsx'")


# ==========================
# COMPARACIÓN POR CONDICIONES
# ==========================
# Calculamos ICC para cada par de condiciones (p.ej. Condition_1 vs Condition_2) y comparamos
conditions = sorted(df_zscore['Condition'].unique())
cond_pairs = list(combinations_with_replacement(conditions, 2))

comparison_rows = []

for (region_1, region_2), _ in df_zscore_filtered.groupby(['Region_1', 'Region_2']):
    if region_1 == region_2:
        continue
    for c1, c2 in cond_pairs:
        subset_no = df_zscore_filtered[(df_zscore_filtered['Region_1'] == region_1) &
                                       (df_zscore_filtered['Region_2'] == region_2) &
                                       (df_zscore_filtered['Condition'].isin([c1, c2]))]

        subset_yes = df_zscore_residualized_filtered[(df_zscore_residualized_filtered['Region_1'] == region_1) &
                                                     (df_zscore_residualized_filtered['Region_2'] == region_2) &
                                                     (df_zscore_residualized_filtered['Condition'].isin([c1, c2]))]

        if subset_no.empty and subset_yes.empty:
            continue

        icc_no = None
        icc_yes = None
        try:
            if not subset_no.empty:
                res_no = pg.intraclass_corr(data=subset_no, targets='Subject', raters='Condition', ratings='Connectivity', nan_policy='omit')
                icc_no = res_no.iloc[4]['ICC']
        except Exception:
            icc_no = None

        try:
            if not subset_yes.empty:
                res_yes = pg.intraclass_corr(data=subset_yes, targets='Subject', raters='Condition', ratings='Connectivity', nan_policy='omit')
                icc_yes = res_yes.iloc[4]['ICC']
        except Exception:
            icc_yes = None

        diff = None
        if (icc_no is not None) and (icc_yes is not None):
            diff = icc_yes - icc_no

        comparison_rows.append({
            'Region_1': region_1,
            'Region_2': region_2,
            'Condition_A': c1,
            'Condition_B': c2,
            'ICC_no_correction': icc_no,
            'ICC_with_correction': icc_yes,
            'Difference': diff
        })

# DataFrame resumen y guardado
df_comparison = pd.DataFrame(comparison_rows)
# Limpiar valores Inf y NaN antes de guardar a Excel
df_comparison = df_comparison.replace([np.inf, -np.inf], np.nan)
df_comparison = df_comparison.fillna("N/A")
try:
    df_comparison.to_excel(os.path.join(path, 'icc_comparison_by_condition_networks.xlsx'), index=False)
    print("Resumen de comparación por condición guardado en 'icc_comparison_by_condition_networks.xlsx'")
except Exception as e:
    print(f"Error guardando comparación: {e}")

# Imprimir comparaciones consolidadas (por par de regiones y condición)
for _, row in df_comparison.iterrows():
    print(f"{row['Region_1']} - {row['Region_2']} | {row['Condition_A']} vs {row['Condition_B']}: ICC_no={row['ICC_no_correction']}, ICC_yes={row['ICC_with_correction']}, diff={row['Difference']}")

# También guardamos una comparación global entre todas las condiciones (ya calculada arriba como global_icc_no_correction y global_icc)
try:
    # Extraer ICC global (mismo índice usado antes)
    global_icc_no_val = None
    global_icc_yes_val = None
    if 'global_icc_no_correction' in globals():
        global_icc_no_val = global_icc_no_correction.iloc[4]['ICC'] if hasattr(global_icc_no_correction, 'iloc') else None
    if 'global_icc' in globals():
        global_icc_yes_val = global_icc.iloc[4]['ICC'] if hasattr(global_icc, 'iloc') else None

    print("\nComparación global entre todas las condiciones:")
    print(f"ICC sin corrección (global): {global_icc_no_val}")
    print(f"ICC con corrección (global): {global_icc_yes_val}")
    if (global_icc_no_val is not None) and (global_icc_yes_val is not None):
        print(f"Diferencia (with - without): {global_icc_yes_val - global_icc_no_val}")

    # Guardar resumen global en Excel
    df_global_comp = pd.DataFrame([{
        'ICC_no_correction_global': global_icc_no_val,
        'ICC_with_correction_global': global_icc_yes_val,
        'Difference_global': (global_icc_yes_val - global_icc_no_val) if (global_icc_no_val is not None and global_icc_yes_val is not None) else None
    }])
    df_global_comp = df_global_comp.replace([np.inf, -np.inf], np.nan)
    df_global_comp = df_global_comp.fillna("N/A")
    try:
        df_global_comp.to_excel(os.path.join(path, 'icc_global_comparison_networks.xlsx'), index=False)
        print("Resumen global guardado en 'icc_global_comparison_networks.xlsx'")
    except Exception as e:
        print(f"Error guardando comparación global: {e}")
except Exception:
    print("No se pudo calcular la comparación global completa.")

# ==========================
# PRUEBA ESTADÍSTICA: EFECTO DE LA EDAD
# Usamos modelos lineales mixtos: Connectivity ~ Age + C(Condition) + (1|Subject)
# Para cada par 'networks' guardamos coeficiente y p-valor de 'Age' (original y residualizado)
age_effect_rows = []

warnings.filterwarnings('ignore')
for (region_1, region_2), group in df_zscore_filtered.groupby(['Region_1', 'Region_2']):
    if region_1 == region_2:
        continue
    row = {'Region_1': region_1, 'Region_2': region_2}
    # Modelo en datos originales
    try:
        model_no = smf.mixedlm('Connectivity ~ Age + C(Condition)', group, groups=group['Subject'])
        fit_no = model_no.fit(reml=False)
        if 'Age' in fit_no.params.index:
            row['Age_coef_no'] = float(fit_no.params['Age'])
            row['Age_pval_no'] = float(fit_no.pvalues['Age']) if 'Age' in fit_no.pvalues.index else None
        else:
            row['Age_coef_no'] = None
            row['Age_pval_no'] = None
    except Exception:
        row['Age_coef_no'] = None
        row['Age_pval_no'] = None

    # Modelo en datos residualizados (aquí esperamos que Age no tenga efecto)
    try:
        group_res = df_zscore_residualized_filtered[(df_zscore_residualized_filtered['Region_1'] == region_1) & (df_zscore_residualized_filtered['Region_2'] == region_2)]
        if not group_res.empty:
            model_yes = smf.mixedlm('Connectivity ~ Age + C(Condition)', group_res, groups=group_res['Subject'])
            fit_yes = model_yes.fit(reml=False)
            if 'Age' in fit_yes.params.index:
                row['Age_coef_yes'] = float(fit_yes.params['Age'])
                row['Age_pval_yes'] = float(fit_yes.pvalues['Age']) if 'Age' in fit_yes.pvalues.index else None
            else:
                row['Age_coef_yes'] = None
                row['Age_pval_yes'] = None
        else:
            row['Age_coef_yes'] = None
            row['Age_pval_yes'] = None
    except Exception:
        row['Age_coef_yes'] = None
        row['Age_pval_yes'] = None

    age_effect_rows.append(row)

df_age_effect = pd.DataFrame(age_effect_rows)
df_age_effect = df_age_effect.replace([np.inf, -np.inf], np.nan)
df_age_effect = df_age_effect.fillna("N/A")
try:
    df_age_effect.to_excel(os.path.join(path, 'age_effect_networks.xlsx'), index=False)
    print("Resultados del test de efecto de Age guardados en 'age_effect_networks.xlsx'")
except Exception as e:
    print(f"Error guardando age effect: {e}")

warnings.filterwarnings('default')
# =====================================================================
# CÁLCULO DE ICC SIN CORRECCIÓN POR EDAD (datos originales)
# =====================================================================
icc_results_no_correction = {}

for (region_1, region_2), group in df_zscore.groupby(['Region_1', 'Region_2']):
    if region_1 != region_2:
        # Calcular ICC para cada par de regiones
        icc = pg.intraclass_corr(data=group, targets='Subject', raters='Condition', ratings='Connectivity', nan_policy='omit')
        icc_results_no_correction[(region_1, region_2)] = icc

global_icc_no_correction = pg.intraclass_corr(data=df_zscore, targets='Subject', raters='Condition', ratings='Connectivity', nan_policy='omit')
print("ICC sin corrección por edad:")
print(global_icc_no_correction)

# Crear un DataFrame con los resultados del ICC sin corrección
icc_results_df_no_correction = pd.DataFrame()

for (region_1, region_2), value in icc_results_no_correction.items():
    icc_value = value.iloc[4]['ICC']
    ci_lower = value.iloc[4]['CI95%'][0]
    ci_upper = value.iloc[4]['CI95%'][1]
    pval = value.iloc[4]['pval']
    
    temp_df = pd.DataFrame({
        'Region_1': [region_1],
        'Region_2': [region_2],
        'ICC': [icc_value],
        'CI Lower': [ci_lower],
        'CI Upper': [ci_upper],
        'p-val': [pval]
    })
    
    icc_results_df_no_correction = pd.concat([icc_results_df_no_correction, temp_df], ignore_index=True)

icc_results_df_no_correction = icc_results_df_no_correction.replace([np.inf, -np.inf], np.nan).fillna("N/A")
try:
    icc_results_df_no_correction.to_excel(os.path.join(path, 'icc_results_without_age_correction.xlsx'), index=False)
    print("Los resultados del ICC SIN corrección se han guardado en 'icc_results_without_age_correction.xlsx'")
except Exception as e:
    print(f"Error guardando ICC sin corrección: {e}")

# =====================================================================
# CÁLCULO DE ICC CON CORRECCIÓN POR EDAD (datos residualizados)
# =====================================================================
icc_results = {}

for (region_1, region_2), group in df_zscore_residualized_filtered.groupby(['Region_1', 'Region_2']):
    if region_1 != region_2:
        # Calcular ICC para cada par de regiones
        icc = pg.intraclass_corr(data=group, targets='Subject', raters='Condition', ratings='Connectivity', nan_policy='omit')
        icc_results[(region_1, region_2)] = icc
        
global_icc = pg.intraclass_corr(data=df_zscore_residualized_filtered, targets='Subject', raters='Condition', ratings='Connectivity', nan_policy='omit')
print("\nICC con corrección por edad:")
print(global_icc)

# Crear un DataFrame con los resultados del ICC
icc_results_df1 = pd.DataFrame()

for (region_1, region_2), value in icc_results.items():
    # Extraer solo el valor del ICC y su intervalo de confianza
    icc_value = value.iloc[4]['ICC']
    ci_lower = value.iloc[4]['CI95%'][0]
    ci_upper = value.iloc[4]['CI95%'][1]
    pval = value.iloc[4]['pval']
    
    # Crear un DataFrame temporal con los resultados
    temp_df = pd.DataFrame({
        'Region_1': [region_1],
        'Region_2': [region_2],
        'ICC': [icc_value],
        'CI Lower': [ci_lower],
        'CI Upper': [ci_upper],
        'p-val': [pval]
    })
    
    # Concatenar el DataFrame temporal al principal
    icc_results_df1 = pd.concat([icc_results_df1, temp_df], ignore_index=True)

icc_results_df1 = icc_results_df1.replace([np.inf, -np.inf], np.nan).fillna("N/A")
try:
    icc_results_df1.to_excel(os.path.join(path, 'icc_results_with_age_correction.xlsx'), index=False)
    print("Los resultados del ICC CON corrección se han guardado en 'icc_results_with_age_correction.xlsx'")
except Exception as e:
    print(f"Error guardando ICC con corrección: {e}")

global_correlations = []

for subject in range(n_subjects):
    # Extraer las matrices de conectividad del sujeto en formato vectorial
    subject_vectors = []
    for cond_idx, mat in enumerate(matrix_zscore):
        # Vectorizar la matriz (tomar solo la parte triangular superior para evitar redundancia)
        upper_triangle = mat[:, :, subject][np.triu_indices(n_regions, k=1)]
        subject_vectors.append(upper_triangle)
    
    # Calcular las correlaciones entre las condiciones para este sujeto
    corr_matrix = np.corrcoef(subject_vectors)
    global_correlations.append(corr_matrix)

# Mostrar correlaciones promedio entre condiciones
mean_correlation = np.mean([np.mean(corr_matrix[np.triu_indices(len(mat_files), k=1)]) for corr_matrix in global_correlations])
print(f"Correlación promedio entre condiciones: {mean_correlation:.4f}")

# Guardar resultados individuales en Excel
corr_data = []
for subject_idx, corr_matrix in enumerate(global_correlations):
    for i, j in combinations_with_replacement(range(len(mat_files)), 2):
        corr_data.append({
            'Subject': subject_idx,
            'Condition_1': f'Condition_{i+1}',
            'Condition_2': f'Condition_{j+1}',
            'Correlation': corr_matrix[i, j]
        })

df_corr = pd.DataFrame(corr_data)
df_corr.to_excel(os.path.join(path, 'global_correlations.xlsx'), index=False)
print("Las correlaciones globales se han guardado en 'global_correlations.xlsx'")