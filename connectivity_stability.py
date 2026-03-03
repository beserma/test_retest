import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import kendalltau
from scipy.spatial.distance import euclidean
from scipy.spatial import procrustes
from scipy import stats
import networkx as nx


def frobenius_distance(mat1, mat2):
    """Compute the Frobenius distance between two matrices, ignoring NaNs."""
    if mat1.shape != mat2.shape:
        raise ValueError("Matrices must have the same shape.")

    valid_mask = ~np.isnan(mat1) & ~np.isnan(mat2)
    diff = (mat1[valid_mask] - mat2[valid_mask]) ** 2
    return np.sqrt(np.sum(diff))


def procrustes_analysis(mat1, mat2):
    """Run Procrustes analysis ignoring NaN positions and return disparity."""
    if mat1.shape != mat2.shape:
        raise ValueError("Matrices must have the same shape.")

    valid_mask = ~np.isnan(mat1) & ~np.isnan(mat2)
    mat1_valid = mat1[valid_mask].reshape(-1, 1)
    mat2_valid = mat2[valid_mask].reshape(-1, 1)

    if mat1_valid.size == 0 or mat2_valid.size == 0:
        raise ValueError("Not enough valid data for Procrustes analysis.")

    _, _, disparity = procrustes(mat1_valid, mat2_valid)
    return disparity


def graph_metrics(matrix, threshold=0.5):
    """Compute simple graph metrics after thresholding matrix."""
    thresholded = (matrix > threshold).astype(int)
    sym = np.maximum(thresholded, thresholded.T)
    G = nx.from_numpy_array(sym)
    return {
        "density": nx.density(G),
        "average_clustering": nx.average_clustering(G),
        "transitivity": nx.transitivity(G),
    }


def kendalls_w(matrix_zscore):
    """Compute Kendall's W as the mean Kendall's tau across condition pairs.

    Excludes rows that contain NaNs.
    """
    n_sessions = len(matrix_zscore)
    n_regions = matrix_zscore[0].shape[0]

    vectorized = np.zeros((n_regions * n_regions, n_sessions))
    for idx, mat in enumerate(matrix_zscore):
        vectorized[:, idx] = np.mean(mat, axis=2).flatten()

    valid_rows = ~np.isnan(vectorized).any(axis=1)
    if not np.any(valid_rows):
        raise ValueError("No valid rows available to compute Kendall's W.")
    vectorized = vectorized[valid_rows]

    taus = []
    for i in range(n_sessions):
        for j in range(i + 1, n_sessions):
            tau, _ = kendalltau(vectorized[:, i], vectorized[:, j])
            if np.isnan(tau):
                print(f"Warning: Kendall's tau is NaN for conditions {i+1} and {j+1}.")
                tau = 0.0
            taus.append(tau)

    return float(np.mean(taus))


def load_matrices(mat_files, path):
    """Load matrices from a folder of .mat files and return z-scored versions plus names."""
    matrix_zscore = []
    matrix_raw = []
    names = None
    names2 = None

    for file in mat_files:
        mat = loadmat(os.path.join(path, file))
        if 'Z' not in mat:
            raise KeyError(f"Key 'Z' not found in {file}.")
        Z = np.array(mat['Z'])
        matrix_raw.append(Z)
        matrix_zscore.append(stats.zscore(Z, axis=2, nan_policy='omit'))

        if names is None and 'names' in mat:
            names = [str(n[0]) for n in mat['names'].flatten()]
        if names2 is None and 'names2' in mat:
            names2 = [str(n[0]) for n in mat['names2'].flatten()]

    return matrix_zscore, matrix_raw, names, names2


if __name__ == '__main__':
    # Standardized input/output
    input_path = '/input'
    output_path = '/output'
    mat_files = ["resultsROI_Condition002.mat",
                 "resultsROI_Condition003.mat",
                 "resultsROI_Condition004.mat"]

    matrix_zscore, matrix_raw, names, names2 = load_matrices(mat_files, input_path)

    n_regions, _, n_subjects = matrix_zscore[0].shape
    assert names is not None and len(names) == n_regions, "Names and matrix dimensions do not match."

    # Global stability
    kendall_w_value = kendalls_w(matrix_zscore)
    print(f"Global stability (Kendall's W): {kendall_w_value}")

    # Network metrics per condition
    for idx, mat in enumerate(matrix_zscore):
        avg = np.mean(mat, axis=2)
        metrics = graph_metrics(avg)
        print(f"Network metrics for condition {idx+1}: {metrics}")

    # Matrix similarities: Frobenius and Procrustes
    distances = {}
    procrustes_disparities = {}
    for i in range(len(matrix_zscore)):
        for j in range(i + 1, len(matrix_zscore)):
            m1 = np.mean(matrix_zscore[i], axis=2)
            m2 = np.mean(matrix_zscore[j], axis=2)
            key = f"Condition {i+1} vs Condition {j+1}"
            distances[key] = frobenius_distance(m1, m2)
            procrustes_disparities[key] = procrustes_analysis(m1, m2)

    print("Frobenius distances between conditions:", distances)
    print("Procrustes disparities between conditions:", procrustes_disparities)
