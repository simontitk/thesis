import numpy as np
import networkx as nx

class ShortestPathKernel:

    @staticmethod
    def _transition_to_cost_matrix(transition_matrix: np.ndarray) -> np.ndarray:
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        transition_matrix_norm = transition_matrix / row_sums
        cost_matrix = np.where(transition_matrix_norm > 0, -np.log(transition_matrix_norm), np.inf)
        np.fill_diagonal(cost_matrix, 0)  
        return cost_matrix


    @staticmethod
    def _transition_matrix_to_graph(transition_matrix: np.ndarray) -> nx.Graph:
        cost_matrix = ShortestPathKernel._transition_to_cost_matrix(transition_matrix)
        G = nx.DiGraph()

        nodes = range(len(cost_matrix))
        for i in nodes:
            for j in nodes:
                cost = cost_matrix[i][j]
                if not np.isinf(cost):
                    G.add_edge(i, j, weight=cost)
        return G


    @staticmethod
    def get_feature_vector(transition_matrix: np.ndarray) -> dict[tuple[int, int], float]:
        """Return:
        - dict of (u,v) node pairs and the total shortest path length between them, `np.inf` if unreachable"""
        G = ShortestPathKernel._transition_matrix_to_graph(transition_matrix)
        shortest_paths: dict[tuple[int, int], float] = {}
        for i in G:
            for j in G:
                try:
                    length = nx.dijkstra_path_length(G, source=i, target=j, weight='weight')
                    shortest_paths[(i, j)] = float(length)
                except nx.NetworkXNoPath:
                    shortest_paths[(i, j)] = np.inf
        return shortest_paths
    

    @staticmethod
    def get_kernel_value(transition_matrix1: np.ndarray, transition_matrix2: np.ndarray, kernel, normalize: bool=False) -> float:
        shortest_paths1 = ShortestPathKernel.get_feature_vector(transition_matrix1)
        shortest_paths2 = ShortestPathKernel.get_feature_vector(transition_matrix2)
        K = 0.0
        shared_paths = shortest_paths1.keys() & shortest_paths2.keys()
        for path in shared_paths:
           similarity = kernel(shortest_paths1[path], shortest_paths2[path])
           K += similarity
        if normalize:
            return K / len(shared_paths)
        return K

    
               # if normalize:
           #    similarity = similarity / (kernel(shortest_paths1[path], shortest_paths1[path]) * kernel(shortest_paths2[path], shortest_paths2[path]))

    @staticmethod
    def inverse_abs_diff_kernel(x: float, y:float) -> float:
        if np.isinf(x) and np.isinf(y):
            return 1.0   
        elif np.isinf(x) or np.isinf(y):
            return 0.0   
        else:
            return 1 / (1 + np.abs(x - y))