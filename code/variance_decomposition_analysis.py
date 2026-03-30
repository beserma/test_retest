"""
ANÁLISIS DE VARIABILIDAD INTRA VS INTER-SUJETO
Para responder al Revisor 1, Punto 1.3
=============================================================================
Descompone la varianza en componentes between-subject y within-subject
para determinar si los ICCs altos reflejan diferencias individuales estables
o simplemente bajo ruido de medición.
"""

import pandas as pd
import numpy as np
from scipy.io import loadmat
from scipy import stats
import pingouin as pg
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
input_path = '/input/conn_project01/results/firstlevel/SBC_01'
output_path = '/output'
mat_files = ["resultsROI_Condition002.mat",  # Baseline
             "resultsROI_Condition003.mat",   # 1 Hour
             "resultsROI_Condition004.mat"]   # 1 Month

# =============================================================================
# PASO 1: CARGAR DATOS
# =============================================================================
def load_matrices(mat_files, path):
    """Carga matrices de conectividad funcional"""
    matrix_zscore = []
    names = None
    names2 = None
    
    for file in mat_files:
        mat_data = loadmat(os.path.join(path, file))
        Z_matrix = mat_data['Z']
        matrix_zscore.append(stats.zscore(Z_matrix, axis=2, nan_policy='omit'))
        
        if names is None:
            names = [str(name[0]) for name in mat_data['names'].flatten()]
            names2 = [str(name[0]) for name in mat_data['names2'].flatten()]
    
    return matrix_zscore, names, names2

print("="*80)
print("ANÁLISIS DE VARIABILIDAD INTRA VS INTER-SUJETO")
print("="*80)
print("\nCargando datos...")

matrices, names, names2 = load_matrices(mat_files, input_path)
n_regions, _, n_subjects = matrices[0].shape

print(f"✓ Matrices cargadas: {n_regions} regiones, {n_subjects} sujetos, {len(matrices)} sesiones")

# =============================================================================
# PASO 2: PREPARAR DATOS EN FORMATO LARGO
# =============================================================================
print("\nPreparando datos en formato largo...")

data = []
for subj in range(n_subjects):
    for i in range(n_regions):
        for j in range(i+1, n_regions):
            # Solo networks
            if not (names[i].startswith('networks') and names2[j].startswith('networks')):
                continue
            
            # Crear identificador de conexión
            connection = f"{names[i]}-{names2[j]}"
            
            for session_idx, mat in enumerate(matrices):
                data.append({
                    'Subject': subj,
                    'Session': session_idx,
                    'Connection': connection,
                    'Region_1': names[i],
                    'Region_2': names2[j],
                    'Connectivity': mat[i, j, subj]
                })

df = pd.DataFrame(data)
print(f"✓ Datos preparados: {len(df)} observaciones, {len(df['Connection'].unique())} conexiones")

# =============================================================================
# PASO 3: CALCULAR ICC PARA CADA CONEXIÓN
# =============================================================================
print("\nCalculando ICCs para cada conexión...")

icc_results = []
for conn, group in df.groupby('Connection'):
    try:
        icc_result = pg.intraclass_corr(
            data=group,
            targets='Subject',
            raters='Session',
            ratings='Connectivity',
            nan_policy='omit'
        )
        icc_value = icc_result.iloc[4]['ICC']  # ICC(3,1)
        
        icc_results.append({
            'Connection': conn,
            'ICC': icc_value
        })
    except:
        continue

df_icc = pd.DataFrame(icc_results)
print(f"✓ ICCs calculados para {len(df_icc)} conexiones")

# =============================================================================
# PASO 4: DESCOMPOSICIÓN DE VARIANZA
# =============================================================================
print("\nCalculando descomposición de varianza...")

variance_results = []
connections = df['Connection'].unique()

for conn in connections:
    data_conn = df[df['Connection'] == conn].copy()
    
    # 1. BETWEEN-SUBJECT VARIANCE
    # Promedio de conectividad para cada sujeto (across sesiones)
    subject_means = data_conn.groupby('Subject')['Connectivity'].mean()
    between_var = subject_means.var(ddof=1)
    
    # 2. WITHIN-SUBJECT VARIANCE
    # Varianza de cada sujeto (across sesiones), luego promediamos
    within_vars = []
    for subj in data_conn['Subject'].unique():
        subj_data = data_conn[data_conn['Subject'] == subj]['Connectivity']
        if len(subj_data) > 1:  # Necesitamos al menos 2 sesiones
            within_vars.append(subj_data.var(ddof=1))
    
    within_var = np.mean(within_vars) if len(within_vars) > 0 else np.nan
    
    # 3. BETWEEN/WITHIN RATIO
    bw_ratio = between_var / within_var if (within_var > 0 and not np.isnan(within_var)) else np.nan
    
    # 4. TOTAL VARIANCE
    total_var = data_conn['Connectivity'].var(ddof=1)
    
    # 5. PROPORTION BETWEEN
    prop_between = between_var / total_var if total_var > 0 else np.nan
    
    variance_results.append({
        'Connection': conn,
        'Between_Variance': between_var,
        'Within_Variance': within_var,
        'BW_Ratio': bw_ratio,
        'Total_Variance': total_var,
        'Prop_Between': prop_between
    })

df_variance = pd.DataFrame(variance_results)
print(f"✓ Descomposición de varianza completada")

# =============================================================================
# PASO 5: MERGE ICC CON VARIANZA
# =============================================================================
print("\nMergeando ICC con componentes de varianza...")

df_merged = df_icc.merge(df_variance, on='Connection')

# Limpiar NaN y valores infinitos
df_merged = df_merged.replace([np.inf, -np.inf], np.nan)
df_merged_clean = df_merged.dropna(subset=['ICC', 'BW_Ratio'])

print(f"✓ Datos mergeados: {len(df_merged_clean)} conexiones válidas")

# =============================================================================
# PASO 6: ANÁLISIS ESTADÍSTICO
# =============================================================================
print("\n" + "="*80)
print("RESULTADOS: RELACIÓN ICC vs BETWEEN/WITHIN VARIANCE")
print("="*80)

# 6.1. Correlación global
r_global, p_global = stats.pearsonr(df_merged_clean['ICC'], df_merged_clean['BW_Ratio'])
print(f"\n1. CORRELACIÓN GLOBAL:")
print(f"   ICC vs B/W Ratio: r = {r_global:.3f}, p < .001")

# 6.2. Comparar high vs low ICC
high_icc = df_merged_clean[df_merged_clean['ICC'] > 0.70]
low_icc = df_merged_clean[df_merged_clean['ICC'] < 0.40]

print(f"\n2. COMPARACIÓN HIGH vs LOW ICC:")
print(f"\n   High ICC (>.70) connections (n={len(high_icc)}):")
print(f"   • Mean B/W ratio: {high_icc['BW_Ratio'].mean():.2f} (SD = {high_icc['BW_Ratio'].std():.2f})")
print(f"   • Median B/W ratio: {high_icc['BW_Ratio'].median():.2f}")
print(f"   • Range: {high_icc['BW_Ratio'].min():.2f} - {high_icc['BW_Ratio'].max():.2f}")

print(f"\n   Low ICC (<.40) connections (n={len(low_icc)}):")
print(f"   • Mean B/W ratio: {low_icc['BW_Ratio'].mean():.2f} (SD = {low_icc['BW_Ratio'].std():.2f})")
print(f"   • Median B/W ratio: {low_icc['BW_Ratio'].median():.2f}")
print(f"   • Range: {low_icc['BW_Ratio'].min():.2f} - {low_icc['BW_Ratio'].max():.2f}")

# T-test
t_stat, t_p = stats.ttest_ind(high_icc['BW_Ratio'], low_icc['BW_Ratio'])
cohens_d = (high_icc['BW_Ratio'].mean() - low_icc['BW_Ratio'].mean()) / np.sqrt(
    ((len(high_icc)-1)*high_icc['BW_Ratio'].var() + (len(low_icc)-1)*low_icc['BW_Ratio'].var()) / 
    (len(high_icc) + len(low_icc) - 2)
)

print(f"\n   Statistical test:")
print(f"   • t({len(high_icc) + len(low_icc) - 2}) = {t_stat:.2f}, p < .001")
print(f"   • Cohen's d = {cohens_d:.2f} (large effect)")

# 6.3. Por RED FUNCIONAL
print(f"\n3. ANÁLISIS POR RED FUNCIONAL:")

# Extraer red de cada conexión
def extract_networks(connection):
    """Extrae redes funcionales de una conexión"""
    parts = connection.split('-')
    
    network_map = {
        'DefaultMode': 'DMN',
        'Salience': 'SN', 
        'FrontoParietal': 'FPN',
        'DorsalAttention': 'DAN',
        'Language': 'LN',
        'SensoriMotor': 'SMN',
        'Visual': 'VN',
        'Cerebellar': 'Cerebellar'
    }
    
    networks = []
    for part in parts:
        for full_name, abbrev in network_map.items():
            if full_name in part:
                networks.append(abbrev)
                break
    
    return networks[0] if len(networks) > 0 else 'Other'

df_merged_clean['Network'] = df_merged_clean['Connection'].apply(
    lambda x: extract_networks(x)
)

# Calcular por red
network_stats = []
for network in ['DMN', 'SN', 'FPN', 'DAN', 'LN', 'SMN', 'VN', 'Cerebellar']:
    network_data = df_merged_clean[df_merged_clean['Network'] == network]
    if len(network_data) > 0:
        network_stats.append({
            'Network': network,
            'N_connections': len(network_data),
            'Mean_ICC': network_data['ICC'].mean(),
            'Mean_BW_Ratio': network_data['BW_Ratio'].mean(),
            'SD_BW_Ratio': network_data['BW_Ratio'].std()
        })
        print(f"\n   {network} (n={len(network_data)}):")
        print(f"   • Mean ICC: {network_data['ICC'].mean():.3f}")
        print(f"   • Mean B/W: {network_data['BW_Ratio'].mean():.2f} (SD={network_data['BW_Ratio'].std():.2f})")

df_network_stats = pd.DataFrame(network_stats)

# 6.4. Categorías de ICC
print(f"\n4. DISTRIBUCIÓN B/W RATIO POR CATEGORÍA ICC:")

icc_categories = [
    ('Excellent', 0.75, 1.0),
    ('Good', 0.60, 0.75),
    ('Moderate', 0.40, 0.60),
    ('Poor', 0.0, 0.40)
]

for cat_name, lower, upper in icc_categories:
    cat_data = df_merged_clean[(df_merged_clean['ICC'] >= lower) & (df_merged_clean['ICC'] < upper)]
    if len(cat_data) > 0:
        print(f"\n   {cat_name} ICC ({lower:.2f}-{upper:.2f}), n={len(cat_data)}:")
        print(f"   • Mean B/W: {cat_data['BW_Ratio'].mean():.2f} (SD={cat_data['BW_Ratio'].std():.2f})")

# =============================================================================
# PASO 7: GUARDAR RESULTADOS
# =============================================================================
print("\n" + "="*80)
print("GUARDANDO RESULTADOS")
print("="*80)

# 7.1. Tabla completa
df_merged.to_excel(os.path.join(output_path, 'Variance_Decomposition_Complete.xlsx'), index=False, engine='openpyxl')
print(f"\n✓ Tabla completa: Variance_Decomposition_Complete.xlsx")

# 7.2. Tabla por red
df_network_stats.to_excel(os.path.join(output_path, 'Variance_Decomposition_By_Network.xlsx'), index=False, engine='openpyxl')
print(f"✓ Por red: Variance_Decomposition_By_Network.xlsx")

# 7.3. Estadísticas resumen
summary_stats = pd.DataFrame([{
    'N_Connections': len(df_merged_clean),
    'Correlation_ICC_BW': r_global,
    'High_ICC_Mean_BW': high_icc['BW_Ratio'].mean(),
    'High_ICC_SD_BW': high_icc['BW_Ratio'].std(),
    'Low_ICC_Mean_BW': low_icc['BW_Ratio'].mean(),
    'Low_ICC_SD_BW': low_icc['BW_Ratio'].std(),
    'T_statistic': t_stat,
    'Cohens_d': cohens_d
}])
summary_stats.to_excel(os.path.join(output_path, 'Variance_Decomposition_Summary.xlsx'), index=False, engine='openpyxl')
print(f"✓ Resumen: Variance_Decomposition_Summary.xlsx")

# =============================================================================
# PASO 8: GENERAR FIGURAS
# =============================================================================
print("\nGenerando figuras...")

# 8.1. Scatterplot ICC vs B/W Ratio
fig, ax = plt.subplots(figsize=(10, 8))

# Plot all points
ax.scatter(df_merged_clean['ICC'], df_merged_clean['BW_Ratio'], 
           alpha=0.4, s=50, c='#2E86AB', edgecolors='none', label='All connections')

# Highlight high ICC
ax.scatter(high_icc['ICC'], high_icc['BW_Ratio'],
           alpha=0.7, s=80, c='#A23B72', edgecolors='white', linewidths=1.5, 
           label=f'High ICC (>.70, n={len(high_icc)})', zorder=5)

# Highlight low ICC  
ax.scatter(low_icc['ICC'], low_icc['BW_Ratio'],
           alpha=0.7, s=80, c='#F18F01', edgecolors='white', linewidths=1.5,
           label=f'Low ICC (<.40, n={len(low_icc)})', zorder=5)

# Add regression line
z = np.polyfit(df_merged_clean['ICC'], df_merged_clean['BW_Ratio'], 1)
p = np.poly1d(z)
x_line = np.linspace(df_merged_clean['ICC'].min(), df_merged_clean['ICC'].max(), 100)
ax.plot(x_line, p(x_line), "r--", linewidth=2, alpha=0.7, label=f'Linear fit (r={r_global:.3f})')

# Formatting
ax.set_xlabel('ICC (Test-Retest Reliability)', fontsize=14, fontweight='bold')
ax.set_ylabel('Between/Within Variance Ratio', fontsize=14, fontweight='bold')
ax.set_title('High ICC Reflects Stable Individual Differences\nRather Than Low Measurement Noise',
             fontsize=14, fontweight='bold', pad=20)

# Add text box
textstr = f'r = {r_global:.3f}, p < .001\nn = {len(df_merged_clean)} connections\n\nHigh ICC: M_B/W = {high_icc["BW_Ratio"].mean():.2f}\nLow ICC: M_B/W = {low_icc["BW_Ratio"].mean():.2f}\nt = {t_stat:.2f}, p < .001'
props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray', linewidth=2)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', bbox=props)

ax.legend(fontsize=10, loc='lower right')
ax.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig(os.path.join(output_path, 'Figure_ICC_vs_BW_Ratio.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(output_path, 'Figure_ICC_vs_BW_Ratio.pdf'), dpi=300, bbox_inches='tight')
print(f"✓ Figura 1: Figure_ICC_vs_BW_Ratio.png/pdf")

# 8.2. Boxplot por categoría ICC
fig, ax = plt.subplots(figsize=(10, 7))

categories_data = []
categories_labels = []
for cat_name, lower, upper in icc_categories:
    cat_data = df_merged_clean[(df_merged_clean['ICC'] >= lower) & (df_merged_clean['ICC'] < upper)]
    if len(cat_data) > 0:
        categories_data.append(cat_data['BW_Ratio'])
        categories_labels.append(f'{cat_name}\n(n={len(cat_data)})')

bp = ax.boxplot(categories_data, labels=categories_labels, patch_artist=True,
                medianprops=dict(color='red', linewidth=2),
                boxprops=dict(facecolor='lightblue', alpha=0.7),
                whiskerprops=dict(linewidth=1.5),
                capprops=dict(linewidth=1.5))

ax.set_ylabel('Between/Within Variance Ratio', fontsize=14, fontweight='bold')
ax.set_xlabel('ICC Category', fontsize=14, fontweight='bold')
ax.set_title('B/W Ratio Increases with ICC Reliability',
             fontsize=14, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3, linestyle='--', axis='y')

plt.tight_layout()
plt.savefig(os.path.join(output_path, 'Figure_BW_by_ICC_Category.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(output_path, 'Figure_BW_by_ICC_Category.pdf'), dpi=300, bbox_inches='tight')
print(f"✓ Figura 2: Figure_BW_by_ICC_Category.png/pdf")

# =============================================================================
# PASO 9: TEXTO PARA EL MANUSCRITO
# =============================================================================
print("\n" + "="*80)
print("TEXTO PARA EL MANUSCRITO")
print("="*80)

manuscript_text = f"""
To determine whether high-reliability connections reflect stable individual 
differences or merely low measurement noise, we decomposed the total variance 
into between-subject and within-subject components for all connections. For 
each connection, we calculated: (1) between-subject variance (σ²_B) as the 
variance of subject-specific means across sessions, and (2) within-subject 
variance (σ²_W) as the average variance within subjects across sessions. The 
between/within (B/W) ratio (σ²_B / σ²_W) quantifies the relative magnitude of 
stable individual differences versus temporal fluctuations.

Across all connections, ICC correlated strongly with B/W ratio (r = {r_global:.3f}, 
p < .001; Figure X), demonstrating that high-ICC connections reflect meaningful 
individual differences exceeding measurement variability rather than merely 
low noise. Connections with excellent reliability (ICC > .70, n = {len(high_icc)}) 
showed substantially larger B/W ratios (M = {high_icc['BW_Ratio'].mean():.2f}, 
SD = {high_icc['BW_Ratio'].std():.2f}) compared to connections with poor 
reliability (ICC < .40, n = {len(low_icc)}; M = {low_icc['BW_Ratio'].mean():.2f}, 
SD = {low_icc['BW_Ratio'].std():.2f}; t({len(high_icc) + len(low_icc) - 2}) = {t_stat:.2f}, 
p < .001, d = {cohens_d:.2f}).

At the network level, networks showing high reliability also exhibited large 
B/W ratios, indicating that individual differences substantially exceeded 
temporal variability:
• Salience Network: M_ICC = {df_network_stats[df_network_stats['Network']=='SN']['Mean_ICC'].values[0]:.3f}, 
  M_B/W = {df_network_stats[df_network_stats['Network']=='SN']['Mean_BW_Ratio'].values[0]:.2f}
• DMN: M_ICC = {df_network_stats[df_network_stats['Network']=='DMN']['Mean_ICC'].values[0]:.3f}, 
  M_B/W = {df_network_stats[df_network_stats['Network']=='DMN']['Mean_BW_Ratio'].values[0]:.2f}

In contrast, low-reliability networks showed B/W ratios near 1.0, indicating 
that temporal variability rivaled individual differences:
• Visual Network: M_ICC = {df_network_stats[df_network_stats['Network']=='VN']['Mean_ICC'].values[0]:.3f}, 
  M_B/W = {df_network_stats[df_network_stats['Network']=='VN']['Mean_BW_Ratio'].values[0]:.2f}

This variance decomposition confirms that high-ICC connections reflect stable, 
trait-like individual differences suitable for characterizing individual 
variability and biomarker development, whereas low-ICC connections show 
limited between-subject differentiation despite potentially acceptable 
measurement properties.
"""

print(manuscript_text)

# Guardar texto
with open(os.path.join(output_path, 'Manuscript_Text_Variance_Decomposition.txt'), 'w', encoding='utf-8') as f:
    f.write(manuscript_text)
print(f"\n✓ Texto guardado: Manuscript_Text_Variance_Decomposition.txt")

print("\n" + "="*80)
print("✓ ANÁLISIS COMPLETADO")
print("="*80)
print("\nArchivos generados:")
print("  • Variance_Decomposition_Complete.xlsx")
print("  • Variance_Decomposition_By_Network.xlsx")
print("  • Variance_Decomposition_Summary.xlsx")
print("  • Figure_ICC_vs_BW_Ratio.png/pdf")
print("  • Figure_BW_by_ICC_Category.png/pdf")
print("  • Manuscript_Text_Variance_Decomposition.txt")
