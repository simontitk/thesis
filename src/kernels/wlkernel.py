from collections import Counter
from typing import Any

import networkx as nx

from kernels.hasher import Hasher


class WLKernel:

    def __init__(self, hasher: Hasher, max_iter: int = 100):
        self.hasher = hasher
        self.max_iter = max_iter


    def refine(self, G: nx.Graph, labels: dict[Any, int]) -> dict[Any, int]:
        """Single relabeling step that collects each node's neighbor labels and hashes this multiset into a new one."""
        new_labels = {}
        for node in G.nodes:
            neighbor_labels = tuple(
                sorted([labels[neighbor] for neighbor in G.neighbors(node)])
            )
            new_labels[node] = self.hasher.hash((labels[node], neighbor_labels))

        return new_labels


    def color(self, G: nx.Graph):
        """Performs repeated refinement steps until the number of different labels stabilizes (or reaching an iteration limit)."""
        iter_count = 0
        labels: dict[Any, int] = {node: 0 for node in G.nodes}
        label_steps = [
            labels,
        ]
        while True:
            iter_count += 1
            new_labels = self.refine(G, labels)
            if (
                len(set(labels.values())) == len(set(new_labels.values()))
                or iter_count == self.max_iter
            ):
                return label_steps
            else:
                label_steps.append(new_labels)
                labels = new_labels


    def get_feature_vector(self, G: nx.Graph, refinement_steps: int) -> list[Counter[Any, int]]:
        labels: dict[Any, int] = {node: 0 for node in G.nodes}
        feature_vector = [Counter(labels.values())]
        for _ in range(refinement_steps):
            new_labels = self.refine(G, labels)
            if len(set(labels.values())) != len(set(new_labels.values())):
                labels = new_labels
            feature_vector.append(Counter(labels.values()))
        return feature_vector



    @staticmethod
    def get_kernel_value(
        feature_vector1: list[Counter[int]], feature_vector2: list[Counter[int]], normalize: bool = True
    ) -> float:
        """Computes and sums the dot products between the feature vectors for each iteration step."""
        feature_vectors = zip(feature_vector1, feature_vector2)
        kernel_value = 0.0
        for label_pair in feature_vectors:
            labels1, labels2 = label_pair
            shared_labels = labels1.keys() & labels2.keys()
            labels_dot_product = sum(labels1[l] * labels2[l] for l in shared_labels)
            if normalize:
                try:
                    labels_dot_product = labels_dot_product / (
                        sum(value**2 for value in labels1.values()) ** 0.5
                        * sum(value**2 for value in labels2.values()) ** 0.5
                    )
                except:
                    labels_dot_product = 0
            kernel_value += labels_dot_product

        if normalize:
            return kernel_value / len(feature_vector1)
        return kernel_value
