"""
CÓDIGO LIMPIO PARA GENERAR TABLA DE ICC
Genera la tabla exacta que necesitas con:
- All timepoints (3 sesiones)
- Baseline vs 1 Hour
- Baseline vs 1 Month  
- 1 Hour vs 1 Month
"""

import pingouin as pg
import numpy as np
import pandas as pd
from scipy.io import loadmat
from itertools import combinations_with_replacement
import os
from scipy import stats
import warnings

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
path = 'Y:\\mnt\\rimp\\PROJECTS\\TEST-RETEST\\Conectividad funcional\\conn_project01\\results\\firstlevel\\SBC_01'
mat_files = ["resultsROI_Condition002.mat",  # Baseline
             "resultsROI_Condition003.mat",   # 1 Hour
             "resultsROI_Condition004.mat"]   # 1 Month

# =============================================================================
# PASO 1: CARGAR DATOS
# =============================================================================
def load_matrices(mat_files, path):
    """Carga las matrices de conectividad funcional"""
    matrix_zscore = []
    matrix_raw = []
    names = None
    names2 = None
    
    for file in mat_files:
        mat_data = loadmat(os.path.join(path, file))
        
        # Cargar matriz y aplicar z-score
        Z_matrix = mat_data['Z']
        matrix_raw.append(np.array(Z_matrix))
        matrix_zscore.append(stats.zscore(Z_matrix, axis=2, nan_policy='omit'))
        
        # Cargar nombres (solo una vez)
        if names is None:
            names = [str(name[0]) for name in mat_data['names'].flatten()]
            names2 = [str(name[0]) for name in mat_data['names2'].flatten()]
    
    return matrix_zscore, matrix_raw, names, names2

print("Cargando matrices...")
matrix_zscore, matrix_raw, names, names2 = load_matrices(mat_files, path)
n_regions, _, n_subjects = matrix_zscore[0].shape

print(f"Matrices cargadas: {n_regions} regiones, {n_subjects} sujetos, {len(matrix_zscore)} condiciones")

# =============================================================================
# PASO 2: PREPARAR DATOS EN FORMATO LARGO
# =============================================================================
def prepare_long_format(matrix_list, names, names2, condition_labels=None):
    """
    Convierte matrices 3D a formato largo para pingouin
    
    Returns DataFrame con columnas: Subject, Region_1, Region_2, Condition, Connectivity
    """
    if condition_labels is None:
        condition_labels = [f'Condition_{i+1}' for i in range(len(matrix_list))]
    
    data = []
    n_regions, _, n_subjects = matrix_list[0].shape
    
    for subj in range(n_subjects):
        for i in range(n_regions):
            for j in range(i+1, n_regions):  # Solo triángulo superior (sin diagonal)
                for cond_idx, mat in enumerate(matrix_list):
                    data.append({
                        'Subject': subj,
                        'Region_1': names[i],
                        'Region_2': names2[j],
                        'Condition': condition_labels[cond_idx],
                        'Connectivity': mat[i, j, subj]
                    })
    
    return pd.DataFrame(data)

print("Preparando datos en formato largo...")
df_all = prepare_long_format(matrix_zscore, names, names2, 
                              ['Baseline', '1Hour', '1Month'])

# Filtrar solo conexiones 'networks'
df_networks = df_all[
    (df_all['Region_1'].str.startswith('networks')) & 
    (df_all['Region_2'].str.startswith('networks'))
].copy()

print(f"Total conexiones: {len(df_all['Region_1'].unique()) * len(df_all['Region_2'].unique())}")
print(f"Conexiones 'networks': {len(df_networks.groupby(['Region_1', 'Region_2']))}")

# =============================================================================
# PASO 3: FUNCIÓN PARA CALCULAR ICC POR REGIÓN
# =============================================================================
def calculate_icc_by_region(df, conditions_to_include=None):
    """
    Calcula ICC para cada región (promediando sus conexiones)
    
    Parameters:
    -----------
    df : DataFrame
        Datos en formato largo
    conditions_to_include : list or None
        Si None, usa todas las condiciones. Si lista, filtra por esas condiciones.
    
    Returns:
    --------
    DataFrame con: Region, Mean_ICC, Pct_NonSignificant, N_connections
    """
    
    # Filtrar condiciones si se especifica
    if conditions_to_include is not None:
        df = df[df['Condition'].isin(conditions_to_include)].copy()
    
    # Extraer prefijo de red de los nombres (ej: 'networks.DefaultMode.LP_L' -> 'DMN')
    def extract_network(region_name):
        if not region_name.startswith('networks.'):
            return 'Other'
        parts = region_name.split('.')
        if len(parts) >= 2:
            network = parts[1]
            # Mapear nombres largos a abreviaturas
            mapping = {
                'DefaultMode': 'DMN',
                'Salience': 'SN',
                'DorsalAttention': 'DAN',
                'FrontoParietal': 'FPN',
                'Language': 'LN',
                'SensoriMotor': 'SMN',
                'Visual': 'VN',
                'Cerebellar': 'Cerebellar'
            }
            return mapping.get(network, network)
        return 'Other'
    
    def extract_region_name(full_name):
        """Extrae el nombre corto de la región"""
        if not full_name.startswith('networks.'):
            return full_name
        parts = full_name.split('.')
        if len(parts) >= 3:
            return parts[2]  # Ej: 'LP_L', 'MPFC', etc.
        return full_name
    
    # Añadir columnas de red y región
    df['Network_1'] = df['Region_1'].apply(extract_network)
    df['Network_2'] = df['Region_2'].apply(extract_network)
    df['Short_Region_1'] = df['Region_1'].apply(extract_region_name)
    df['Short_Region_2'] = df['Region_2'].apply(extract_region_name)
    
    # Calcular ICC para cada CONEXIÓN individual
    connection_results = []
    
    for (reg1, reg2), group in df.groupby(['Region_1', 'Region_2']):
        try:
            icc_result = pg.intraclass_corr(
                data=group,
                targets='Subject',
                raters='Condition',
                ratings='Connectivity',
                nan_policy='omit'
            )
            
            # Usar ICC(3,1) - two-way mixed, absolute agreement (índice 4)
            icc_row = icc_result.iloc[4]
            
            connection_results.append({
                'Region_1': reg1,
                'Region_2': reg2,
                'Network_1': group['Network_1'].iloc[0],
                'Network_2': group['Network_2'].iloc[0],
                'Short_Region_1': group['Short_Region_1'].iloc[0],
                'Short_Region_2': group['Short_Region_2'].iloc[0],
                'ICC': icc_row['ICC'],
                'pval': icc_row['pval'],
                'CI_lower': icc_row['CI95%'][0],
                'CI_upper': icc_row['CI95%'][1],
                'Significant': icc_row['pval'] < 0.05
            })
        except Exception as e:
            print(f"Error en {reg1}-{reg2}: {e}")
            continue
    
    df_connections = pd.DataFrame(connection_results)
    
    # Agregar por REGIÓN (una región puede tener múltiples conexiones)
    # Para cada región, calculamos el promedio de ICC de todas sus conexiones
    region_results = []
    
    # Obtener todas las regiones únicas
    all_regions = set(df_connections['Region_1'].unique()) | set(df_connections['Region_2'].unique())
    
    for region in all_regions:
        # Encontrar todas las conexiones que involucran esta región
        region_conns = df_connections[
            (df_connections['Region_1'] == region) | 
            (df_connections['Region_2'] == region)
        ]
        
        if len(region_conns) == 0:
            continue
        
        # Extraer network y nombre corto
        network = region_conns['Network_1'].iloc[0] if region_conns['Region_1'].iloc[0] == region else region_conns['Network_2'].iloc[0]
        short_name = region_conns['Short_Region_1'].iloc[0] if region_conns['Region_1'].iloc[0] == region else region_conns['Short_Region_2'].iloc[0]
        
        # Calcular estadísticas
        mean_icc = region_conns['ICC'].mean()
        pct_nonsig = (region_conns['pval'] > 0.05).mean() * 100
        n_connections = len(region_conns)
        
        region_results.append({
            'Network': network,
            'Region': short_name,
            'Full_Region': region,
            'Mean_ICC': mean_icc,
            'Pct_NonSignificant': pct_nonsig,
            'N_Connections': n_connections
        })
    
    return pd.DataFrame(region_results), df_connections

# =============================================================================
# PASO 4: CALCULAR ICC PARA CADA COMPARACIÓN
# =============================================================================

print("\n" + "="*70)
print("CALCULANDO ICCs PARA TODAS LAS COMPARACIONES")
print("="*70)

# 4.1. ALL TIMEPOINTS (3 sesiones)
print("\n1. Calculando All Timepoints (3 sesiones)...")
regions_all, connections_all = calculate_icc_by_region(df_networks, conditions_to_include=None)
regions_all = regions_all.rename(columns={
    'Mean_ICC': 'ICC_All',
    'Pct_NonSignificant': 'PctNonSig_All'
})

# 4.2. BASELINE vs 1 HOUR
print("2. Calculando Baseline vs 1 Hour...")
regions_b1h, connections_b1h = calculate_icc_by_region(df_networks, conditions_to_include=['Baseline', '1Hour'])
regions_b1h = regions_b1h.rename(columns={
    'Mean_ICC': 'ICC_B1H',
    'Pct_NonSignificant': 'PctNonSig_B1H'
})

# 4.3. BASELINE vs 1 MONTH
print("3. Calculando Baseline vs 1 Month...")
regions_b1m, connections_b1m = calculate_icc_by_region(df_networks, conditions_to_include=['Baseline', '1Month'])
regions_b1m = regions_b1m.rename(columns={
    'Mean_ICC': 'ICC_B1M',
    'Pct_NonSignificant': 'PctNonSig_B1M'
})

# 4.4. 1 HOUR vs 1 MONTH
print("4. Calculando 1 Hour vs 1 Month...")
regions_1h1m, connections_1h1m = calculate_icc_by_region(df_networks, conditions_to_include=['1Hour', '1Month'])
regions_1h1m = regions_1h1m.rename(columns={
    'Mean_ICC': 'ICC_1H1M',
    'Pct_NonSignificant': 'PctNonSig_1H1M'
})

# =============================================================================
# PASO 4.5: EL ANÁLISIS DEL REVISOR 3 (AVERAGED BASELINE vs 1 MONTH)
# =============================================================================
print("\n4.5. Calculando Averaged Baseline (T1+T2) vs 1 Month (T3)...")

# 1. Crear un DataFrame con el promedio de T1 (Baseline) y T2 (1Hour)
df_avg = df_networks[df_networks['Condition'].isin(['Baseline', '1Hour'])].copy()
df_avg = df_avg.groupby(['Subject', 'Region_1', 'Region_2', 'Network_1', 'Network_2', 
                         'Short_Region_1', 'Short_Region_2'])['Connectivity'].mean().reset_index()
df_avg['Condition'] = 'Averaged_Baseline'

# 2. Obtener los datos de T3 (1Month)
df_t3 = df_networks[df_networks['Condition'] == '1Month'].copy()

# 3. Combinar ambos para el cálculo del ICC
df_reviewer_analysis = pd.concat([df_avg, df_t3], ignore_index=True)

# 4. Calcular ICC usando la función que ya definiste
regions_avg_vs_month, connections_avg_vs_month = calculate_icc_by_region(
    df_reviewer_analysis, conditions_to_include=['Averaged_Baseline', '1Month']
)

regions_avg_vs_month = regions_avg_vs_month.rename(columns={
    'Mean_ICC': 'ICC_Avg_vs_Month',
    'Pct_NonSignificant': 'PctNonSig_Avg_vs_Month'
})

print(f"ICC Promedio (Avg Baseline vs 1 Month): {connections_avg_vs_month['ICC'].mean():.3f}")

# =============================================================================
# PASO 5: COMBINAR RESULTADOS EN TABLA FINAL
# =============================================================================

print("\n5. Combinando resultados...")

# Merge todos los dataframes
table = regions_all[['Network', 'Region', 'Full_Region', 'ICC_All', 'PctNonSig_All', 'N_Connections']]
table = table.merge(regions_b1h[['Full_Region', 'ICC_B1H', 'PctNonSig_B1H']], on='Full_Region', how='left')
table = table.merge(regions_b1m[['Full_Region', 'ICC_B1M', 'PctNonSig_B1M']], on='Full_Region', how='left')
table = table.merge(regions_1h1m[['Full_Region', 'ICC_1H1M', 'PctNonSig_1H1M']], on='Full_Region', how='left')

table = table.merge(regions_avg_vs_month[['Full_Region', 'ICC_Avg_vs_Month', 'PctNonSig_Avg_vs_Month']], 
                    on='Full_Region', how='left')

# Ordenar por Network y Region
network_order = ['Cerebellar', 'DMN', 'DAN', 'FPN', 'LN', 'SN', 'SMN', 'VN']
table['Network_Order'] = table['Network'].apply(lambda x: network_order.index(x) if x in network_order else 999)
table = table.sort_values(['Network_Order', 'Region']).drop('Network_Order', axis=1)

# Formatear columnas
for col in ['ICC_All', 'ICC_B1H', 'ICC_B1M', 'ICC_1H1M', 'ICC_Avg_vs_Month']:
    table[col] = table[col].round(3)

for col in ['PctNonSig_All', 'PctNonSig_B1H', 'PctNonSig_B1M', 'PctNonSig_1H1M', 'PctNonSig_Avg_vs_Month']:
    table[col] = table[col].round(0).astype('Int64')  # Int64 permite NaN

# Calcular totales
total_row = pd.DataFrame([{
    'Network': 'Total ROIs',
    'Region': '',
    'Full_Region': 'TOTAL',
    'N_Connections': table['N_Connections'].sum(),
    'ICC_All': connections_all['ICC'].mean(),
    'PctNonSig_All': (connections_all['pval'] > 0.05).mean() * 100,
    'ICC_B1H': connections_b1h['ICC'].mean(),
    'PctNonSig_B1H': (connections_b1h['pval'] > 0.05).mean() * 100,
    'ICC_B1M': connections_b1m['ICC'].mean(),
    'PctNonSig_B1M': (connections_b1m['pval'] > 0.05).mean() * 100,
    'ICC_1H1M': connections_1h1m['ICC'].mean(),
    'PctNonSig_1H1M': (connections_1h1m['pval'] > 0.05).mean() * 100,
    'ICC_Avg_vs_Month': connections_avg_vs_month['ICC'].mean(),
    'PctNonSig_Avg_vs_Month': (connections_avg_vs_month['pval'] > 0.05).mean() * 100
}])

for col in ['ICC_All', 'ICC_B1H', 'ICC_B1M', 'ICC_1H1M', 'ICC_Avg_vs_Month']:
    total_row[col] = total_row[col].round(3)

table_final = pd.concat([table, total_row], ignore_index=True)

# =============================================================================
# PASO 6: GUARDAR RESULTADOS
# =============================================================================

# Guardar tabla principal
output_file = os.path.join(path, 'ICC_Table_Complete.xlsx')
table_final.to_excel(output_file, index=False)
print(f"\n✓ Tabla guardada en: {output_file}")

# Guardar conexiones individuales (para debugging)
connections_all.to_excel(os.path.join(path, 'ICC_Connections_All.xlsx'), index=False)
connections_b1h.to_excel(os.path.join(path, 'ICC_Connections_B1H.xlsx'), index=False)
connections_b1m.to_excel(os.path.join(path, 'ICC_Connections_B1M.xlsx'), index=False)
connections_1h1m.to_excel(os.path.join(path, 'ICC_Connections_1H1M.xlsx'), index=False)
connections_avg_vs_month.to_excel(os.path.join(path, 'ICC_Connections_Avg_vs_Month.xlsx'), index=False)

# =============================================================================
# PASO 7: MOSTRAR RESUMEN
# =============================================================================

print("\n" + "="*70)
print("RESUMEN DE RESULTADOS")
print("="*70)

print(f"\nTotal de regiones analizadas: {len(table)}")
print(f"Total de conexiones: {table['N_Connections'].sum()}")

print("\n--- ICCs GLOBALES ---")
print(f"All timepoints:      ICC = {total_row['ICC_All'].iloc[0]:.3f}")
print(f"Baseline vs 1 Hour:  ICC = {total_row['ICC_B1H'].iloc[0]:.3f}")
print(f"Baseline vs 1 Month: ICC = {total_row['ICC_B1M'].iloc[0]:.3f}")
print(f"1 Hour vs 1 Month:   ICC = {total_row['ICC_1H1M'].iloc[0]:.3f}")

print("\n--- % NON-SIGNIFICANT GLOBALES ---")
print(f"All timepoints:      {total_row['PctNonSig_All'].iloc[0]:.0f}%")
print(f"Baseline vs 1 Hour:  {total_row['PctNonSig_B1H'].iloc[0]:.0f}%")
print(f"Baseline vs 1 Month: {total_row['PctNonSig_B1M'].iloc[0]:.0f}%")
print(f"1 Hour vs 1 Month:   {total_row['PctNonSig_1H1M'].iloc[0]:.0f}%")

print("\n--- TOP 5 REGIONES MÁS FIABLES (All timepoints) ---")
top5 = table.nlargest(5, 'ICC_All')[['Network', 'Region', 'ICC_All', 'PctNonSig_All']]
print(top5.to_string(index=False))

print("\n--- TOP 5 REGIONES MENOS FIABLES (All timepoints) ---")
bottom5 = table.nsmallest(5, 'ICC_All')[['Network', 'Region', 'ICC_All', 'PctNonSig_All']]
print(bottom5.to_string(index=False))

print("\n✓ ANÁLISIS COMPLETADO")
print("="*70)
