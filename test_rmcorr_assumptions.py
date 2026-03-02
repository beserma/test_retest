"""
DIAGNÓSTICO COMPLETO: ¿De dónde vienen 72 y 506?
Compara conteo de conexiones entre diferentes métodos
"""

import numpy as np
import pandas as pd
from scipy.io import loadmat
from itertools import combinations_with_replacement
import os

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
path = 'Y:\\mnt\\rimp\\PROJECTS\\TEST-RETEST\\Conectividad funcional\\conn_project01\\results\\firstlevel\\SBC_01'
mat_files = ["resultsROI_Condition002.mat",
             "resultsROI_Condition003.mat",
             "resultsROI_Condition004.mat"]

# =============================================================================
# CARGAR DATOS
# =============================================================================

def load_matrices(mat_files, path):
    matrix_list = []
    names = None
    names2 = None
    
    for file in mat_files:
        mat_data = loadmat(os.path.join(path, file))
        Z_matrix = mat_data['Z']
        matrix_list.append(Z_matrix)
        
        if names is None:
            names = [str(name[0]) for name in mat_data['names'].flatten()]
            names2 = [str(name[0]) for name in mat_data['names2'].flatten()]
    
    return matrix_list, names, names2

matrix_list, names, names2 = load_matrices(mat_files, path)
n_regions, n_regions2, n_subjects = matrix_list[0].shape

print("="*80)
print("DIAGNÓSTICO: CONTEO DE CONEXIONES")
print("="*80)

# =============================================================================
# REGIONES
# =============================================================================

low_icc_regions = [
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

print(f"\nInformación básica:")
print(f"  Total ROIs (names): {len(names)}")
print(f"  Total ROIs (names2): {len(names2)}")
print(f"  Low-ICC regions definidas: {len(low_icc_regions)}")

# Contar cuántas ROIs "networks" hay
networks_count_names = sum(1 for name in names if name.startswith('networks'))
networks_count_names2 = sum(1 for name in names2 if name.startswith('networks'))

print(f"\nROIs que empiezan con 'networks':")
print(f"  En names: {networks_count_names}")
print(f"  En names2: {networks_count_names2}")

# Identificar índices
low_icc_indices = [i for i, name in enumerate(names) if name in low_icc_regions]
high_icc_indices = [i for i in range(n_regions) 
                    if i not in low_icc_indices and names[i].startswith('networks')]

print(f"\nÍndices identificados:")
print(f"  Low-ICC: {len(low_icc_indices)} ROIs")
print(f"  High-ICC: {len(high_icc_indices)} ROIs")

# =============================================================================
# MÉTODO 1: COMBINATIONS_WITH_REPLACEMENT (tu método original)
# =============================================================================

print(f"\n{'='*80}")
print(f"MÉTODO 1: combinations_with_replacement")
print(f"{'='*80}")

count_low_m1 = 0
count_high_m1 = 0

for i, j in combinations_with_replacement(low_icc_indices, 2):
    region_1 = names[i]
    region_2 = names2[j]
    if region_1.startswith('networks') and region_2.startswith('networks'):
        count_low_m1 += 1
    

for i, j in combinations_with_replacement(high_icc_indices, 2):
    region_1 = names[i]
    region_2 = names2[j]
    if region_1.startswith('networks') and region_2.startswith('networks'):
        count_high_m1 += 1

print(f"Low-ICC: {count_low_m1} conexiones")
print(f"High-ICC: {count_high_m1} conexiones")

# =============================================================================
# MÉTODO 2: TODAS LAS COMBINACIONES CON TODAS LAS REGIONS
# =============================================================================

print(f"\n{'='*80}")
print(f"MÉTODO 2: Low-ICC × ALL networks regions")
print(f"{'='*80}")

count_low_m2 = 0
count_high_m2 = 0

# Low-ICC: cada región low-ICC con TODAS las regiones networks
for i in low_icc_indices:
    for j in range(len(names2)):
        if names2[j].startswith('networks'):
            count_low_m2 += 1

# High-ICC: cada región high-ICC con TODAS las regiones networks
for i in high_icc_indices:
    for j in range(len(names2)):
        if names2[j].startswith('networks'):
            count_high_m2 += 1

print(f"Low-ICC × ALL networks: {count_low_m2} conexiones")
print(f"High-ICC × ALL networks: {count_high_m2} conexiones")

# =============================================================================
# MÉTODO 3: COMO LO HACE EL CÓDIGO ORIGINAL correlate_fc_psychology
# =============================================================================

print(f"\n{'='*80}")
print(f"MÉTODO 3: Simulando código original (itertools sin replacement)")
print(f"{'='*80}")

# El código original probablemente hace esto:
# for i, j in combinations_with_replacement(range(n_regions), 2):
#     if names[i].startswith('networks') and names2[j].startswith('networks'):
#         ... excluir low_icc ...

count_low_m3 = 0
count_high_m3 = 0

all_networks_indices = [i for i in range(n_regions) if names[i].startswith('networks')]

print(f"Total regiones 'networks' en names: {len(all_networks_indices)}")

for i, j in combinations_with_replacement(all_networks_indices, 2):
    region_1 = names[i]
    region_2 = names2[j]
    
    # Clasificar según si alguna región está en low_icc
    if region_1 in low_icc_regions or region_2 in low_icc_regions:
        count_low_m3 += 1
    else:
        count_high_m3 += 1

print(f"Conexiones con al menos una región Low-ICC: {count_low_m3}")
print(f"Conexiones solo entre regiones High-ICC: {count_high_m3}")

# =============================================================================
# MÉTODO 4: VERIFICAR RESULTADOS ORIGINALES
# =============================================================================

print(f"\n{'='*80}")
print(f"VERIFICAR: ¿De dónde vienen 72 y 506 del control negativo original?")
print(f"{'='*80}")

# Cargar resultados previos si existen
results_file = os.path.join(path, 'Negative_Control_Results.xlsx')

if os.path.exists(results_file):
    print(f"\n✓ Encontrado: Negative_Control_Results.xlsx")
    
    try:
        df_low_all = pd.read_excel(results_file, sheet_name='Low_ICC_All')
        df_high_all = pd.read_excel(results_file, sheet_name='High_ICC_All')
        
        print(f"\nConexiones en resultados originales:")
        print(f"  Low_ICC_All: {len(df_low_all)} filas")
        print(f"  High_ICC_All: {len(df_high_all)} filas")
        
        # Ver estructura
        print(f"\nColumnas en Low_ICC_All:")
        for col in df_low_all.columns:
            print(f"  - {col}")
        
        # ¿Hay duplicados de conexiones por variable psicológica?
        if 'Psychological_Variable' in df_low_all.columns:
            n_unique_low = len(df_low_all[['Region_1', 'Region_2']].drop_duplicates())
            n_psych_vars = df_low_all['Psychological_Variable'].nunique()
            print(f"\nConexiones únicas en Low_ICC_All: {n_unique_low}")
            print(f"Variables psicológicas: {n_psych_vars}")
            print(f"Total filas: {len(df_low_all)} (= {n_unique_low} × {n_psych_vars})")
            
            n_unique_high = len(df_high_all[['Region_1', 'Region_2']].drop_duplicates())
            print(f"\nConexiones únicas en High_ICC_All: {n_unique_high}")
            print(f"Variables psicológicas: {df_high_all['Psychological_Variable'].nunique()}")
            print(f"Total filas: {len(df_high_all)} (= {n_unique_high} × {df_high_all['Psychological_Variable'].nunique()})")
        
    except Exception as e:
        print(f"Error leyendo archivo: {e}")
else:
    print(f"\n✗ No encontrado: Negative_Control_Results.xlsx")
    print(f"  Por favor ejecuta primero el análisis de control negativo original")

# =============================================================================
# MÉTODO 5: CÁLCULO TEÓRICO
# =============================================================================

print(f"\n{'='*80}")
print(f"CÁLCULO TEÓRICO: ¿Cuántas conexiones DEBERÍAN ser?")
print(f"{'='*80}")

n_low = len(low_icc_indices)
n_high = len(high_icc_indices)
n_total_networks = len(all_networks_indices)

print(f"\nROIs identificadas:")
print(f"  Low-ICC: {n_low}")
print(f"  High-ICC: {n_high}")
print(f"  Total networks: {n_total_networks}")

print(f"\nSi analizamos connections_with_replacement DENTRO de cada grupo:")
print(f"  Low-ICC solo entre sí: {n_low * (n_low + 1) // 2} = {n_low}×{n_low+1}÷2")
print(f"  High-ICC solo entre sí: {n_high * (n_high + 1) // 2} = {n_high}×{n_high+1}÷2")

print(f"\nSi cada ROI se conecta con TODAS las regiones networks:")
print(f"  Low-ICC × ALL: {n_low} × {n_total_networks} = {n_low * n_total_networks}")
print(f"  High-ICC × ALL: {n_high} × {n_total_networks} = {n_high * n_total_networks}")

print(f"\nSi usamos matriz triangular de TODAS las networks:")
print(f"  Total conexiones: {n_total_networks * (n_total_networks + 1) // 2}")
print(f"  Conexiones con ≥1 low-ICC: depende de intersecciones")
print(f"  Conexiones solo high-ICC: {n_high * (n_high + 1) // 2}")

# =============================================================================
# RESUMEN
# =============================================================================

print(f"\n{'='*80}")
print(f"RESUMEN DE MÉTODOS")
print(f"{'='*80}")

summary = pd.DataFrame({
    'Método': [
        '1. combinations_with_replacement (dentro grupo)',
        '2. Each ROI × ALL networks',
        '3. Clasificar por low/high después',
        '4. Resultados originales (archivo)',
        '5. Teórico: triangular ALL networks'
    ],
    'Low-ICC': [
        count_low_m1,
        count_low_m2,
        count_low_m3,
        'Ver archivo',
        'Depende'
    ],
    'High-ICC': [
        count_high_m1,
        count_high_m2,
        count_high_m3,
        'Ver archivo',
        f'{n_high * (n_high + 1) // 2}'
    ]
})

print(f"\n{summary.to_string(index=False)}")

print(f"\n{'='*80}")
print(f"CONCLUSIÓN")
print(f"{'='*80}")
print(f"""
Para resolver la discrepancia necesitamos saber:

1. ¿Cómo se calcularon originalmente los "72" y "506"?
   - ¿Es por variable psicológica? (72 = 36 conexiones × 2 vars)
   - ¿Es total de análisis rm_corr ejecutados?
   - ¿Es solo conexiones únicas?

2. ¿Qué define una "conexión" en tu análisis?
   - ¿Low-ICC ROI con CUALQUIER otra ROI networks?
   - ¿Low-ICC ROI solo con otras Low-ICC ROIs?
   - ¿Todas las networks, clasificadas después?

Por favor revisa tu código original 'correlate_fc_psychology_repeated_measures'
y dime exactamente qué hace en el loop de conexiones.
""")

# Guardar diagnóstico
summary.to_excel(os.path.join(path, 'Connection_Count_Diagnostic.xlsx'), index=False)
print(f"\n✓ Diagnóstico guardado: Connection_Count_Diagnostic.xlsx")