import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm
from statsmodels.tsa.stattools import grangercausalitytests
import numpy as np
import torch
from tqdm import tqdm
from torch_geometric.data import Data
from sklearn.feature_selection import mutual_info_regression
from statsmodels.tsa.stattools import grangercausalitytests
from joblib import Parallel, delayed
import numpy as np
import warnings

def compute_granger_adjacency_from_windows(X_windowed, maxlag=1, threshold=0.05):

    N, num_nodes, window_size = X_windowed.shape
    A = np.zeros((num_nodes, num_nodes))


    node_features = X_windowed.mean(axis=2)  # shape: [N, num_nodes]

    for i in tqdm(range(num_nodes), desc="caculating"):
        for j in range(num_nodes):
            if i == j:
                continue
            try:
                series = np.stack([node_features[:, j], node_features[:, i]], axis=1)
                result = grangercausalitytests(series, maxlag=maxlag, verbose=False)
                p_values = [result[lag][0]['ssr_ftest'][1] for lag in range(1, maxlag + 1)]
                if np.min(p_values) < threshold:
                    A[i, j] = 1
            except:
                continue
    return A


def Tcompute_granger_adjacency_from_windows(X_windowed, maxlag=1, threshold=0.05, method='ssr_ftest', n_jobs=-1):

    warnings.filterwarnings("ignore")

    N, num_nodes, window_size = X_windowed.shape
    node_features = X_windowed.mean(axis=2)  # shape: [N, num_nodes]

    def compute_single_pair(i, j):
        if i == j:
            return (i, j, 0.0)

        series = np.stack([node_features[:, j], node_features[:, i]], axis=1)
        if series.shape[0] <= maxlag + 1:
            return (i, j, 0.0)

        try:
            result = grangercausalitytests(series, maxlag=maxlag, verbose=False)
            p_values = [result[lag][0][method][1] for lag in range(1, maxlag + 1) if method in result[lag][0]]
            if p_values and np.min(p_values) < threshold:
                return (i, j, 1.0)
        except Exception:
            pass
        return (i, j, 0.0)

    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_single_pair)(i, j)
        for i in range(num_nodes)
        for j in range(num_nodes)
    )

    A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for i, j, val in results:
        A[i, j] = val

    return A


def slice_spectra_to_windows(X, window_size=5, stride=5):
    """
    将每条光谱切片为多个窗口，返回 shape = [N, num_nodes, window_size]
    """
    stride = window_size
    if isinstance(X, torch.Tensor):
        X = X.cpu().numpy()
    N, D = X.shape
    num_nodes = (D - window_size) // stride + 1
    sliced = np.zeros((N, num_nodes, window_size))
    for i in range(N):
        for j in range(num_nodes):
            start = j * stride
            sliced[i, j] = X[i, start:start+window_size]
    return sliced  # shape: [N, num_nodes, window_size]

from joblib import Parallel, delayed

def get_pyg_dataset(X, labels, adj_matrix=None, window_size=5, stride=5, maxlag=1, threshold=0.05, method="granger"):
    X_windowed = slice_spectra_to_windows(X, window_size=window_size, stride=stride)
    N, num_nodes, win_size = X_windowed.shape

    if adj_matrix is None:
        if method == "granger":
            adj_matrix = compute_granger_adjacency_from_windows(X_windowed, maxlag=maxlag, threshold=threshold)
        else:
            raise ValueError("Unsupported method for graph construction. Use 'granger' or 'mutual_info'.")

    edge_index_np = np.array(np.nonzero(adj_matrix))  # shape: [2, num_edges]
    edge_index = torch.tensor(edge_index_np, dtype=torch.long)
    edge_weight = torch.tensor(adj_matrix[adj_matrix > 0], dtype=torch.float)


    graph_list = []

    for i in range(N):
        x_i = torch.tensor(X_windowed[i], dtype=torch.float)  # shape: [num_nodes, window_size]
        y_i = torch.tensor(labels[i], dtype=torch.float)
        data = Data(x=x_i, edge_index=edge_index, edge_attr=edge_weight, y=y_i, spec=X[i])
        graph_list.append(data)

    return graph_list, adj_matrix
