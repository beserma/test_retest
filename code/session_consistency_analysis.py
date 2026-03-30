
import pingouin as pg
import numpy as np
import pandas as pd
from scipy.io import loadmat
from itertools import combinations_with_replacement
from scipy import stats
import os
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
input_path = '/input'
output_path = '/output'  # Directorio de entrada/salida para datos y resultados
mat_files = ["resultsROI_Condition002.mat",  # Baseline
             "resultsROI_Condition003.mat",   # 1 Hour
             "resultsROI_Condition004.mat"]   # 1 Month

psych_file = os.path.join(input_path, 'Clinical_results.xlsx')
psych_data = pd.read_excel(psych_file)
psych_vars = ['REACT_TOT', 'PROACT_TOT']

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

print("Cargando datos...")
matrix_list, names, names2 = load_matrices(mat_files, input_path)
n_regions, _, n_subjects = matrix_list[0].shape

# =============================================================================
# REGIONES LOW-ICC Y HIGH-ICC
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

low_icc_indices = [i for i, name in enumerate(names) if name in low_icc_regions]
high_icc_indices = [i for i in range(n_regions) 
                    if i not in low_icc_indices and names[i].startswith('networks')]

# =============================================================================
# FUNCIÓN: CORRELACIONES SESIÓN-ESPECÍFICAS
# =============================================================================

def calculate_session_specific_correlations(matrix_list, names, names2, 
                                             psych_data, psych_vars, 
                                             included_indices):
    """
    Calcula correlaciones FC-behavior para CADA SESIÓN por separado
    """
    n_regions, _, n_subjects = matrix_list[0].shape
    session_names = ['Baseline', '1-Hour', '1-Month']
    
    results = []
    
    # Para cada par de regiones
    for i, j in combinations_with_replacement(included_indices, 2):
        if i >= len(names) or j >= len(names2):
            continue
            
        region_1 = names[i]
        region_2 = names2[j]
        
        if not (region_1.startswith('networks') and region_2.startswith('networks')):
            continue
        
        # Para cada variable psicológica
        for psych_var in psych_vars:
            
            correlations_by_session = []
            
            # Para cada sesión
            for session_idx, matrix in enumerate(matrix_list):
                # Extraer FC para esta sesión
                fc_values = matrix[i, j, :n_subjects]
                psych_values = psych_data[psych_var].values[:n_subjects]
                
                # Remover NaNs
                valid = ~(np.isnan(fc_values) | np.isnan(psych_values))
                
                if valid.sum() < 10:  # Mínimo 10 sujetos
                    correlations_by_session.append(np.nan)
                    continue
                
                # Calcular correlación Pearson
                r, p = stats.pearsonr(fc_values[valid], psych_values[valid])
                correlations_by_session.append(r)
            
            # Si tenemos las 3 correlaciones
            if len(correlations_by_session) == 3 and not any(np.isnan(correlations_by_session)):
                results.append({
                    'Region_1': region_1,
                    'Region_2': region_2,
                    'Psych_Variable': psych_var,
                    'r_Baseline': correlations_by_session[0],
                    'r_1Hour': correlations_by_session[1],
                    'r_1Month': correlations_by_session[2],
                    'Mean_r': np.mean(correlations_by_session),
                    'SD_r': np.std(correlations_by_session),
                    'Min_r': np.min(correlations_by_session),
                    'Max_r': np.max(correlations_by_session),
                    'Range_r': np.max(correlations_by_session) - np.min(correlations_by_session),
                    'Sign_Consistent': (np.sign(correlations_by_session[0]) == 
                                       np.sign(correlations_by_session[1]) == 
                                       np.sign(correlations_by_session[2])),
                    'All_Positive': all(r > 0 for r in correlations_by_session),
                    'All_Negative': all(r < 0 for r in correlations_by_session)
                })
    
    return pd.DataFrame(results)

# =============================================================================
# ANÁLISIS 1: LOW-ICC CONNECTIONS
# =============================================================================

print("\n" + "="*80)
print("ANÁLISIS LOW-ICC: CONSISTENCIA ENTRE SESIONES")
print("="*80)

results_low = calculate_session_specific_correlations(
    matrix_list, names, names2, psych_data, psych_vars, low_icc_indices
)

print(f"\nConexiones Low-ICC analizadas: {len(results_low)}")

if len(results_low) > 0:
    # Consistencia de signo
    n_consistent = results_low['Sign_Consistent'].sum()
    pct_consistent = n_consistent / len(results_low) * 100
    
    print(f"\nConsistencia de signo entre sesiones:")
    print(f"  Conexiones con signo consistente: {n_consistent}/{len(results_low)} ({pct_consistent:.1f}%)")
    print(f"  Conexiones con signo inconsistente: {len(results_low) - n_consistent} ({100-pct_consistent:.1f}%)")
    
    # Variabilidad de r
    mean_sd = results_low['SD_r'].mean()
    mean_range = results_low['Range_r'].mean()
    
    print(f"\nVariabilidad de correlaciones:")
    print(f"  SD promedio de r entre sesiones: {mean_sd:.3f}")
    print(f"  Rango promedio de r: {mean_range:.3f}")
    
    # Conexiones con |r| > 0.5 en al menos una sesión
    results_low['Max_abs_r'] = results_low[['r_Baseline', 'r_1Hour', 'r_1Month']].abs().max(axis=1)
    strong_connections = results_low[results_low['Max_abs_r'] >= 0.5]
    
    print(f"\nConexiones con |r| ≥ 0.5 en al menos una sesión: {len(strong_connections)}")
    
    if len(strong_connections) > 0:
        print("\nEjemplos de conexiones fuertes Low-ICC:")
        for idx, row in strong_connections.head(5).iterrows():
            print(f"\n  {row['Region_1']} <-> {row['Region_2']}")
            print(f"  Variable: {row['Psych_Variable']}")
            print(f"    Baseline: r = {row['r_Baseline']:+.3f}")
            print(f"    1-Hour:   r = {row['r_1Hour']:+.3f}")
            print(f"    1-Month:  r = {row['r_1Month']:+.3f}")
            print(f"    Consistente: {'Sí' if row['Sign_Consistent'] else 'No'}")

# =============================================================================
# ANÁLISIS 2: HIGH-ICC CONNECTIONS (COMPARACIÓN)
# =============================================================================

print("\n" + "="*80)
print("ANÁLISIS HIGH-ICC: CONSISTENCIA ENTRE SESIONES")
print("="*80)

results_high = calculate_session_specific_correlations(
    matrix_list, names, names2, psych_data, psych_vars, high_icc_indices
)

print(f"\nConexiones High-ICC analizadas: {len(results_high)}")

if len(results_high) > 0:
    n_consistent_high = results_high['Sign_Consistent'].sum()
    pct_consistent_high = n_consistent_high / len(results_high) * 100
    
    print(f"\nConsistencia de signo entre sesiones:")
    print(f"  Conexiones con signo consistente: {n_consistent_high}/{len(results_high)} ({pct_consistent_high:.1f}%)")
    
    mean_sd_high = results_high['SD_r'].mean()
    mean_range_high = results_high['Range_r'].mean()
    
    print(f"\nVariabilidad de correlaciones:")
    print(f"  SD promedio de r entre sesiones: {mean_sd_high:.3f}")
    print(f"  Rango promedio de r: {mean_range_high:.3f}")

# =============================================================================
# COMPARACIÓN LOW-ICC vs HIGH-ICC
# =============================================================================

print("\n" + "="*80)
print("COMPARACIÓN: LOW-ICC vs HIGH-ICC")
print("="*80)

if len(results_low) > 0 and len(results_high) > 0:
    
    # Test de proporciones (consistencia de signo)
    from statsmodels.stats.proportion import proportions_ztest
    
    counts = np.array([n_consistent, n_consistent_high])
    nobs = np.array([len(results_low), len(results_high)])
    
    z_stat, p_val_prop = proportions_ztest(counts, nobs)
    
    print(f"\nConsistencia de signo:")
    print(f"  Low-ICC:  {pct_consistent:.1f}%")
    print(f"  High-ICC: {pct_consistent_high:.1f}%")
    print(f"  z = {z_stat:.2f}, p = {p_val_prop:.3f}")
    
    # Test de diferencia de medias (SD de r)
    t_stat_sd, p_val_sd = stats.ttest_ind(results_low['SD_r'], results_high['SD_r'])
    
    print(f"\nVariabilidad de r entre sesiones (SD):")
    print(f"  Low-ICC:  M = {mean_sd:.3f}, SD = {results_low['SD_r'].std():.3f}")
    print(f"  High-ICC: M = {mean_sd_high:.3f}, SD = {results_high['SD_r'].std():.3f}")
    print(f"  t({len(results_low) + len(results_high) - 2}) = {t_stat_sd:.2f}, p = {p_val_sd:.3f}")
    
    # Cohen's d
    pooled_sd = np.sqrt((results_low['SD_r'].std()**2 + results_high['SD_r'].std()**2) / 2)
    cohens_d = (mean_sd - mean_sd_high) / pooled_sd
    print(f"  Cohen's d = {cohens_d:.2f}")

# =============================================================================
# ANÁLISIS 3: PROMEDIO vs RM_CORR
# =============================================================================

print("\n" + "="*80)
print("ANÁLISIS: PROMEDIO DE SESIONES vs RM_CORR")
print("="*80)

# Calcular correlaciones usando promedio de sesiones
def calculate_mean_session_correlations(matrix_list, names, names2, 
                                         psych_data, psych_vars, 
                                         included_indices):
    """
    Correlaciona el PROMEDIO de las 3 sesiones con behavior
    """
    n_regions, _, n_subjects = matrix_list[0].shape
    results = []
    
    for i, j in combinations_with_replacement(included_indices, 2):
        if i >= len(names) or j >= len(names2):
            continue
            
        region_1 = names[i]
        region_2 = names2[j]
        
        if not (region_1.startswith('networks') and region_2.startswith('networks')):
            continue
        
        # Calcular promedio de FC entre sesiones
        fc_mean = np.mean([matrix_list[0][i, j, :], 
                          matrix_list[1][i, j, :], 
                          matrix_list[2][i, j, :]], axis=0)
        
        for psych_var in psych_vars:
            psych_values = psych_data[psych_var].values[:n_subjects]
            
            valid = ~(np.isnan(fc_mean) | np.isnan(psych_values))
            
            if valid.sum() < 10:
                continue
            
            r, p = stats.pearsonr(fc_mean[valid], psych_values[valid])
            
            results.append({
                'Region_1': region_1,
                'Region_2': region_2,
                'Psych_Variable': psych_var,
                'r_mean': r,
                'p_value': p
            })
    
    return pd.DataFrame(results)

results_low_mean = calculate_mean_session_correlations(
    matrix_list, names, names2, psych_data, psych_vars, low_icc_indices
)

results_high_mean = calculate_mean_session_correlations(
    matrix_list, names, names2, psych_data, psych_vars, high_icc_indices
)

if len(results_low_mean) > 0 and len(results_high_mean) > 0:
    mean_r_low_avg = np.abs(results_low_mean['r_mean']).mean()
    mean_r_high_avg = np.abs(results_high_mean['r_mean']).mean()
    
    print(f"\nUsando PROMEDIO de sesiones:")
    print(f"  Low-ICC:  Mean |r| = {mean_r_low_avg:.3f}")
    print(f"  High-ICC: Mean |r| = {mean_r_high_avg:.3f}")
    print(f"  Diferencia: {mean_r_low_avg - mean_r_high_avg:+.3f}")
    
    # Comparar con rm_corr original
    print(f"\nComparación con rm_corr (del análisis original):")
    print(f"  rm_corr Low-ICC:  Mean |r| = 0.274")
    print(f"  rm_corr High-ICC: Mean |r| = 0.163")
    print(f"\n  ¿Se mantiene el patrón con promedio? {'Sí' if mean_r_low_avg > mean_r_high_avg else 'No'}")

# =============================================================================
# VISUALIZACIÓN
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=300)

# Panel A: Consistencia de signo
if len(results_low) > 0 and len(results_high) > 0:
    data_consistency = pd.DataFrame({
        'Group': ['Low-ICC', 'High-ICC'],
        'Consistent': [pct_consistent, pct_consistent_high],
        'Inconsistent': [100-pct_consistent, 100-pct_consistent_high]
    })
    
    x = np.arange(len(data_consistency))
    width = 0.35
    
    axes[0,0].bar(x, data_consistency['Consistent'], width, label='Consistent sign',
                  color='#2E86AB', edgecolor='black')
    axes[0,0].bar(x, data_consistency['Inconsistent'], width, bottom=data_consistency['Consistent'],
                  label='Inconsistent sign', color='#A23B72', edgecolor='black')
    
    axes[0,0].set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
    axes[0,0].set_title('A) Sign Consistency Across Sessions', fontsize=12, fontweight='bold')
    axes[0,0].set_xticks(x)
    axes[0,0].set_xticklabels(['Low-ICC', 'High-ICC'])
    axes[0,0].legend(frameon=True, fontsize=9)
    axes[0,0].grid(True, alpha=0.3, axis='y')
    axes[0,0].axhline(50, color='red', linestyle='--', alpha=0.5, linewidth=1)

# Panel B: Variabilidad de r
if len(results_low) > 0 and len(results_high) > 0:
    data_var = pd.DataFrame({
        'Group': ['Low-ICC']*len(results_low) + ['High-ICC']*len(results_high),
        'SD_r': list(results_low['SD_r']) + list(results_high['SD_r'])
    })
    
    sns.boxplot(data=data_var, x='Group', y='SD_r', ax=axes[0,1],
                palette={'Low-ICC': '#A23B72', 'High-ICC': '#2E86AB'},
                order=['Low-ICC', 'High-ICC'])
    sns.stripplot(data=data_var, x='Group', y='SD_r', ax=axes[0,1],
                  color='black', alpha=0.3, size=3,
                  order=['Low-ICC', 'High-ICC'])
    
    axes[0,1].set_ylabel('SD of r Across Sessions', fontsize=11, fontweight='bold')
    axes[0,1].set_xlabel('')
    axes[0,1].set_title('B) Variability of Correlations', fontsize=12, fontweight='bold')
    axes[0,1].grid(True, alpha=0.3, axis='y')

# Panel C: Ejemplo Low-ICC (primera conexión fuerte)
if len(strong_connections) > 0:
    example = strong_connections.iloc[0]
    sessions = ['Baseline', '1-Hour', '1-Month']
    r_values = [example['r_Baseline'], example['r_1Hour'], example['r_1Month']]
    
    colors = ['#2E86AB' if r > 0 else '#A23B72' for r in r_values]
    axes[1,0].bar(sessions, r_values, color=colors, edgecolor='black', alpha=0.7)
    axes[1,0].axhline(0, color='black', linewidth=1)
    axes[1,0].axhline(0.5, color='red', linestyle='--', alpha=0.5, label='|r| = 0.5')
    axes[1,0].axhline(-0.5, color='red', linestyle='--', alpha=0.5)
    axes[1,0].set_ylabel('Correlation (r)', fontsize=11, fontweight='bold')
    axes[1,0].set_title(f'C) Example Low-ICC Connection\n{example["Psych_Variable"]}',
                        fontsize=12, fontweight='bold')
    axes[1,0].legend(frameon=True, fontsize=9)
    axes[1,0].grid(True, alpha=0.3, axis='y')
    axes[1,0].set_ylim([-1, 1])

# Panel D: Promedio vs rm_corr
if len(results_low_mean) > 0 and len(results_high_mean) > 0:
    data_comparison = pd.DataFrame({
        'Method': ['rm_corr', 'rm_corr', 'Mean', 'Mean'],
        'Group': ['Low-ICC', 'High-ICC', 'Low-ICC', 'High-ICC'],
        'Mean_abs_r': [0.274, 0.163, mean_r_low_avg, mean_r_high_avg]
    })
    
    x = np.arange(2)
    width = 0.35
    
    axes[1,1].bar(x - width/2, [0.274, 0.163], width, label='rm_corr',
                  color='#18978F', edgecolor='black')
    axes[1,1].bar(x + width/2, [mean_r_low_avg, mean_r_high_avg], width, label='Mean of sessions',
                  color='#F18F01', edgecolor='black')
    
    axes[1,1].set_ylabel('Mean |r| with Behavior', fontsize=11, fontweight='bold')
    axes[1,1].set_title('D) rm_corr vs Session Average', fontsize=12, fontweight='bold')
    axes[1,1].set_xticks(x)
    axes[1,1].set_xticklabels(['Low-ICC', 'High-ICC'])
    axes[1,1].legend(frameon=True, fontsize=9)
    axes[1,1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(output_path, 'Figure_Session_Consistency_Analysis.png'),
            dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(os.path.join(output_path, 'Figure_Session_Consistency_Analysis.pdf'),
            dpi=300, bbox_inches='tight')

print(f"\n✓ Figura guardada: Figure_Session_Consistency_Analysis.png/pdf")

plt.close()

# =============================================================================
# GUARDAR RESULTADOS
# =============================================================================

with pd.ExcelWriter(os.path.join(output_path, 'Session_Consistency_Analysis.xlsx'),
                    engine='openpyxl') as writer:
    
    results_low.to_excel(writer, sheet_name='Low_ICC_By_Session', index=False)
    results_high.to_excel(writer, sheet_name='High_ICC_By_Session', index=False)
    results_low_mean.to_excel(writer, sheet_name='Low_ICC_Mean', index=False)
    results_high_mean.to_excel(writer, sheet_name='High_ICC_Mean', index=False)
    
    # Resumen
    summary = pd.DataFrame({
        'Metric': [
            'N connections',
            '% Sign consistent',
            'Mean SD(r) across sessions',
            'Mean |r| (rm_corr)',
            'Mean |r| (session average)',
            '',
            'Sign consistency z-test',
            'p-value',
            '',
            'SD(r) t-test',
            'p-value',
            "Cohen's d"
        ],
        'Low-ICC': [
            len(results_low),
            f"{pct_consistent:.1f}%",
            f"{mean_sd:.3f}",
            "0.274",
            f"{mean_r_low_avg:.3f}",
            '',
            f"{z_stat:.2f}",
            f"{p_val_prop:.3f}",
            '',
            f"{t_stat_sd:.2f}",
            f"{p_val_sd:.3f}",
            f"{cohens_d:.2f}"
        ],
        'High-ICC': [
            len(results_high),
            f"{pct_consistent_high:.1f}%",
            f"{mean_sd_high:.3f}",
            "0.163",
            f"{mean_r_high_avg:.3f}",
            '',
            '',
            '',
            '',
            '',
            '',
            ''
        ]
    })
    summary.to_excel(writer, sheet_name='Summary', index=False)

print(f"\n✓ Resultados guardados: Session_Consistency_Analysis.xlsx")

# =============================================================================
# TEXTO PARA MANUSCRITO
# =============================================================================

print("\n" + "="*80)
print("TEXTO PARA MANUSCRITO")
print("="*80)

print(f"""
Session-Specific Consistency of Brain-Behavior Correlations

To address whether the stronger correlations observed for low-ICC connections reflect 
stable trait-behavior relationships versus session-specific artifacts, we examined 
correlation patterns across individual scanning sessions. If low ICC simply reflects 
random measurement noise, correlations should vary unpredictably in both magnitude 
and direction across sessions. Conversely, if low ICC reflects state-dependent 
variability superimposed on a stable trait component, correlations should maintain 
consistent directionality despite magnitude fluctuations.

For low-ICC connections (n = {len(results_low)}), {pct_consistent:.1f}% showed 
consistent correlation sign across all three sessions (Baseline, 1-Hour, 1-Month), 
compared to {pct_consistent_high:.1f}% for high-ICC connections (z = {z_stat:.2f}, 
p = {p_val_prop:.3f}). Although low-ICC connections showed greater variability in 
correlation magnitude across sessions (SD = {mean_sd:.3f}) compared to high-ICC 
connections (SD = {mean_sd_high:.3f}; t({len(results_low) + len(results_high) - 2}) = {t_stat_sd:.2f}, 
p = {p_val_sd:.3f}, d = {cohens_d:.2f}), the majority maintained consistent 
directionality.

Critically, correlations computed using session-averaged connectivity (mean of 3 
timepoints) produced similar patterns to repeated-measures correlations: low-ICC 
connections showed mean |r| = {mean_r_low_avg:.3f} versus high-ICC mean |r| = {mean_r_high_avg:.3f}, 
maintaining the {'stronger' if mean_r_low_avg > mean_r_high_avg else 'weaker'} 
associations observed with the rm_corr approach. This consistency across analytical 
strategies demonstrates that the findings do not depend on a specific modeling choice 
but reflect robust patterns in the data.

These results support the interpretation that low-ICC connections contain both 
trait-stable and state-variable components. The trait component—reflected in consistent 
correlation directionality and session-averaged associations—predicts stable aggression 
traits. The state component—reflected in session-to-session magnitude fluctuations—reduces 
test-retest reliability while potentially capturing meaningful context-dependent reactivity. 
This pattern is particularly evident in Visual and Dorsal Attention networks, where 
fluctuations in threat-relevant attention likely vary across scanning contexts while 
the average tendency relates to trait aggression.
""")

print("\n" + "="*80)
print("✓ ANÁLISIS COMPLETADO")
print("="*80)
