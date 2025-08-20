from kneed import KneeLocator
import seaborn as sns
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.cluster import KMeans
from typedefs import AOICenters
from scipy import linalg

class GraphUtils:

    @staticmethod
    def cluster(k_clusters: int, df: pd.DataFrame) -> tuple[pd.DataFrame, float, AOICenters]:
        """Input:
        - k_clusters: number of clusters
        - df: dataframe with columns named 'x' and 'y' for time-ordered gaze coordinates
        \nReturn:
        - new df with column 'AOI' and 'cluster_center' contianing the cluster number and its center
        - inertia"""

        X = df[["x", "y"]]
        kmeans = KMeans(n_clusters=k_clusters, random_state=0)
        kmeans.fit(X)
        df_copy = df.copy()
        df_copy["AOI"] = kmeans.labels_
        df_copy["center_x"] = df_copy.groupby("AOI")["x"].transform("mean")
        df_copy["center_y"] = df_copy.groupby("AOI")["y"].transform("mean")

        cluster_centers: AOICenters = df_copy.groupby("AOI")[["x", "y"]].mean().apply(lambda row: (int(row.x), int(row.y)), axis=1).to_dict()
        
        return df_copy, kmeans.inertia_, cluster_centers


    @staticmethod
    def clustering(min_clusters: int, max_clusters: int, df: pd.DataFrame) -> tuple[dict[int, pd.DataFrame], dict[int, float], dict[int, AOICenters]]:
        """Input:
        - min_clusters: inclusive
        - max_clusters: inclusive
        - df: dataframe with columns named 'x' and 'y' for time-ordered gaze coordinates
        \nReturn:
        - dict(k, fitted df), and dict(k, inertias)"""

        dfs: dict[int, pd.DataFrame] = {}
        inertias: dict[int, float] = {}
        cluster_centers: dict[int, AOICenters] = {}

        cluster_numbers = range(min_clusters, max_clusters+1)
        for k in cluster_numbers:
            df_copy, inertia, center_dict = GraphUtils.cluster(k, df)
            dfs[k] = df_copy
            inertias[k] = inertia
            cluster_centers[k] = center_dict

        return dfs, inertias, cluster_centers
    

    @staticmethod
    def optimal_clustering(min_clusters: int, max_clusters: int, df: pd.DataFrame):
        """Input:
        - min_clusters: inclusive
        - max_clusters: inclusive
        - df: dataframe with columns named 'x' and 'y' for time-ordered gaze coordinates
        \nReturn:
        - fitted df with new 'AOI' and 'cluster_center' column, inertia for optimal k, optimal k"""
        dfs, inertias, cluster_centers = GraphUtils.clustering(min_clusters, max_clusters, df)
        k_values, inertia_values = zip(*inertias.items())
        knee_locator = KneeLocator(k_values, inertia_values, curve='convex', direction='decreasing')
        k: int = knee_locator.knee
        return dfs[k], inertias[k], cluster_centers[k], k
    

    @staticmethod
    def cluster_to_aois(trial: pd.DataFrame, aois: AOICenters) -> pd.DataFrame:
        trial_copy = trial.copy()
        aoi_labels = np.array(list(aois.keys()))
        aoi_coords = np.array(list(aois.values()))
        points = trial[["x", "y"]].to_numpy()
        distances = np.sum((points[:, None, :] - aoi_coords[None, :, :]) ** 2, axis=2) 
        nearest_indices = np.argmin(distances, axis=1)
        trial_copy["AOI"] = aoi_labels[nearest_indices]      
        return trial_copy


    @staticmethod
    def transition_matrix(trial: pd.DataFrame, normalize: bool = False, aois: AOICenters = None) -> np.ndarray:
        """Input:
        - trial: dataframe with x, y, t, AOI columns
        - normalize: divide each transition count by the total number of transitions from that AOI
        \nReturn:
        - AOI x AOI transition matrix"""

        aoi_count = len(aois) if aois else trial["AOI"].nunique()
        transition_matrix = np.zeros(shape=(aoi_count, aoi_count), dtype=int)
        for i in range(len(trial.index)-1):
            from_aoi = trial.loc[i, "AOI"]
            to_aoi = trial.loc[i+1, "AOI"]
            transition_matrix[from_aoi][to_aoi] += 1
        if normalize:
            return GraphUtils._normalize(transition_matrix)
        
        return transition_matrix
    

    @staticmethod
    def _normalize(transition_matrix: np.ndarray) -> np.ndarray:
        rows_sum = np.sum(transition_matrix, axis=1)
        transition_matrix_norm = transition_matrix / rows_sum
        return np.nan_to_num(transition_matrix_norm, nan=0.0)
    

    @staticmethod
    def transition_matrix_to_graph(transition_matrix: np.ndarray) -> nx.Graph:
        G = nx.DiGraph()
        nodes = range(len(transition_matrix))
        G.add_nodes_from(nodes)
        for i in nodes:
            for j in nodes:
                weight = transition_matrix[i][j]
                if weight != 0:
                    G.add_edge(i, j, weight=weight)
        return G


    @staticmethod
    def transition_entropy(transition_matrix: np.ndarray) -> float:
        M = GraphUtils._normalize(transition_matrix)
        aois = len(M)
        for aoi in range(aois):
            if M[aoi].sum() == 0:
                M[aoi] = 1 / aois 
        pi = GraphUtils._stationary_probabilites(M)

        H_transition = 0
        for i in range(aois):
            for j in range(aois):   
                if M[i][j] > 0:
                    H_transition += -1 * pi[i] * M[i][j] * np.log2(M[i][j])
        return H_transition

    
    @staticmethod
    def stationairy_entropy(transition_matrix: np.ndarray) -> float:
        M = GraphUtils._normalize(transition_matrix)
        aois = len(M)
        for aoi in range(aois):
            if M[aoi].sum() == 0:
                M[aoi] = 1 / aois 
        pi = GraphUtils._stationary_probabilites(M)
        print(pi)
        H_stationary = -np.sum([p*np.log2(p) for p in pi if p > 0])
        return H_stationary
    
    
    @staticmethod
    def _stationary_probabilites(transition_matrix: np.ndarray) -> float:
        eigen_vals, eigen_vecs = linalg.eig(transition_matrix.T)  
        idx = np.argmin(np.abs(eigen_vals - 1))
        pi = np.real(eigen_vecs[:, idx])
        pi /= pi.sum()  
        return pi
