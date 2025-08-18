import numpy as np
import networkx as nx


class LocalMetrics:

    @staticmethod
    def degree_centrality(G: nx.Graph):
        centralities =  nx.degree_centrality(G)
        return np.array([centralities[k] for k in sorted(centralities.keys())])


    @staticmethod
    def closeness_centrality(G: nx.Graph):
        centralities =  nx.closeness_centrality(G)
        return np.array([centralities[k] for k in sorted(centralities.keys())])


    @staticmethod
    def betweenness_centrality(G: nx.Graph):
        centralities =  nx.betweenness_centrality(G)
        return np.array([centralities[k] for k in sorted(centralities.keys())])
        
    
    @staticmethod
    def reach_centrality(G: nx.Graph):
        reach = []
        for node in sorted(G.nodes):
            reachable = nx.descendants(G, node)  
            reach.append(len(reachable) / (len(G.nodes) - 1))
        return np.array(reach)