import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import kendalltau
from scipy.spatial.distance import euclidean
from sklearn.utils.extmath import randomized_svd
import networkx as nx
import numpy as np
import pandas as pd
from scipy.io import loadmat
import os
from scipy import stats
from statsmodels.stats.multitest import multipletests
from scipy.stats import ttest_rel
from scipy.spatial import procrustes

# Funciones auxiliares
def frobenius_distance(mat1, mat2):
    """Calcula la distancia de Frobenius entre dos matrices, ignorando los NaN."""
    # Asegurarse de que mat1 y mat2 tienen las mismas dimensiones
    assert mat1.shape == mat2.shape, "Las matrices deben tener las mismas dimensiones."
    
    # Crear una máscara que marque los valores válidos (no NaN)
    valid_mask = ~np.isnan(mat1) & ~np.isnan(mat2)
    
    # Usar la máscara para ignorar las posiciones NaN
    diff = (mat1[valid_mask] - mat2[valid_mask]) ** 2
    
    # Calcular la distancia de Frobenius solo sobre los datos válidos
    distance = np.sqrt(np.sum(diff))
    
    return distance

def procrustes_analysis(mat1, mat2):
    """Análisis de Procrustes que ignora las posiciones NaN."""
    # Asegurarse de que las matrices tienen las mismas dimensiones
    assert mat1.shape == mat2.shape, "Las matrices deben tener las mismas dimensiones."
    
    # Crear una máscara que marque los valores válidos (no NaN)
    valid_mask = ~np.isnan(mat1) & ~np.isnan(mat2)
    
    # Filtrar las matrices utilizando la máscara, conservando solo los valores válidos
    mat1_valid = mat1[valid_mask].reshape(-1, 1)  # reshape a una forma adecuada
    mat2_valid = mat2[valid_mask].reshape(-1, 1)  # reshape a una forma adecuada
    
    # Si las matrices no tienen datos válidos, no podemos realizar el análisis
    if mat1_valid.size == 0 or mat2_valid.size == 0:
        raise ValueError("Las matrices no tienen suficientes datos válidos para realizar el análisis de Procrustes.")
    
    # Realizar el análisis de Procrustes sobre los datos válidos
    mtx1, mtx2, disparity = procrustes(mat1_valid, mat2_valid)
    
    return disparity

def graph_metrics(matrix, threshold=0.5):
    """
    Calcula métricas de red usando NetworkX.
    Aplica un umbral para binarizar la matriz antes de construir el grafo.
    
    :param matrix: Matriz de conectividad funcional (promedio entre sujetos).
    :param threshold: Umbral para binarizar la matriz.
    :return: Diccionario con métricas de red.
    """
    # Aplicar umbral para binarizar la matriz
    thresholded_matrix = (matrix > threshold).astype(int)
    
    # Asegurar que la matriz sea simétrica
    sym_matrix = np.maximum(thresholded_matrix, thresholded_matrix.T)
    
    # Crear el grafo
    G = nx.from_numpy_array(sym_matrix)
    
    # Calcular métricas
    metrics = {
        "density": nx.density(G),
        "average_clustering": nx.average_clustering(G),
        "transitivity": nx.transitivity(G),
    }
    return metrics

def kendalls_w(matrix_zscore):
    """
    Calcula Kendall's W utilizando scipy.stats.kendalltau para medir estabilidad global.
    Maneja filas con valores NaN excluyéndolas del análisis.
    :param matrix_zscore: Lista de matrices (z-score) de conectividad funcional.
    :return: Valor de Kendall's W.
    """
    n_sesiones = len(matrix_zscore)
    n_regions = matrix_zscore[0].shape[0]

    # Vectorizar matrices promedio por sujetos
    vectorized = np.zeros((n_regions * n_regions, n_sesiones))
    for idx, matrix in enumerate(matrix_zscore):
        avg_matrix = np.mean(matrix, axis=2)  # Promedio por sujetos
        vectorized[:, idx] = avg_matrix.flatten()

    # Excluir filas con valores NaN
    valid_rows = ~np.isnan(vectorized).any(axis=1)
    if not np.any(valid_rows):
        raise ValueError("Todas las filas contienen valores NaN. No se puede calcular Kendall's W.")
    vectorized = vectorized[valid_rows]

    # Calcular Kendall's tau para cada par de condiciones
    tau_matrix = []
    for i in range(n_sesiones):
        for j in range(i + 1, n_sesiones):
            tau, _ = kendalltau(vectorized[:, i], vectorized[:, j])
            if np.isnan(tau):
                print(f"Advertencia: Kendall's tau es NaN para condiciones {i+1} y {j+1}.")
                tau = 0  # Asigna un valor neutro si no hay variación
            tau_matrix.append(tau)

    # Calcular W como promedio de tau
    kendall_w = np.mean(tau_matrix)
    return kendall_w

# Carga de matrices
def load_matrices(mat_files, path):
    names_key = "names"
    names2_key = "names2"
    functional_connectivity = "Z"
    
    matrix_zscore = []
    matrix_raw = []
    names = None
    names2 = None

    for file in mat_files:
        mat_data = loadmat(os.path.join(path, file))
        if functional_connectivity in mat_data:
            functional_matrix = stats.zscore(mat_data[functional_connectivity], nan_policy='omit')
            matrix_zscore.append(np.array(functional_matrix))
            matrix_raw.append(np.array(mat_data[functional_connectivity]))
        else:
            raise KeyError(f"La clave '{functional_connectivity}' no se encontró en {file}.")
        
        # Cargar nombres solo una vez (asumimos que son consistentes entre condiciones)
        if names is None and names_key in mat_data:
            names = [str(name[0]) for name in mat_data[names_key].flatten()]
        if names2 is None and names2_key in mat_data:
            names2 = [str(name[0]) for name in mat_data[names2_key].flatten()]
    
    return matrix_zscore, matrix_raw, names, names2

# Configuración de archivos y directorio
path = 'Y:\\mnt\\rimp\\PROJECTS\\TEST-RETEST\\Conectividad funcional\\conn_project01\\results\\firstlevel\\SBC_01'
mat_files = ["resultsROI_Condition002.mat",
             "resultsROI_Condition003.mat",
             "resultsROI_Condition004.mat"]

# Cargar las matrices de conectividad funcional y nombres
matrix_zscore, matrix_raw, names, names2 = load_matrices(mat_files, path)

# Verificar consistencia de las dimensiones
n_regions, _, n_subjects = matrix_zscore[0].shape
assert len(names) == n_regions and len(names2) == n_regions, "Dimensiones de nombres y matriz no coinciden."

# 1. Estabilidad Global (Kendall’s W)
kendall_w_value = kendalls_w(matrix_zscore)
print(f"Estabilidad Global (Kendall's W): {kendall_w_value}")

# 2. Métricas de red para cada condición
for idx, matrix in enumerate(matrix_zscore):
    avg_matrix = np.mean(matrix, axis=2)  # Promedio entre sujetos
    metrics = graph_metrics(avg_matrix)
    print(f"Métricas de red para la condición {idx + 1}: {metrics}")

# 3. Similitud de Matrices
# Distancia de Frobenius
distances = {}
for i in range(len(matrix_zscore)):
    for j in range(i + 1, len(matrix_zscore)):
        mean_matrix1 = np.mean(matrix_zscore[i], axis=2)
        mean_matrix2 = np.mean(matrix_zscore[j], axis=2)
        distances[f"Condición {i+1} vs Condición {j+1}"] = frobenius_distance(mean_matrix1, mean_matrix2)

print("Distancias de Frobenius entre condiciones:", distances)

# Disparidad de Procrustes
procrustes_disparities = {}
for i in range(len(matrix_zscore)):
    for j in range(i + 1, len(matrix_zscore)):
        mean_matrix1 = np.mean(matrix_zscore[i], axis=2)
        mean_matrix2 = np.mean(matrix_zscore[j], axis=2)
        procrustes_disparities[f"Condición {i+1} vs Condición {j+1}"] = procrustes_analysis(mean_matrix1, mean_matrix2)

print("Disparidad de Procrustes entre condiciones:", procrustes_disparities)
