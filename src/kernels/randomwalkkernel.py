import numpy as np


class RandomWalkKernel:
    def weighted_random_walk_kernel(
        transition_matrix1: np.ndarray,
        transition_matrix2: np.ndarray,
        decay: float = 0.9,
        normalize: bool = False,
    ) -> float:
        rho_1 = max(abs((np.linalg.eigvals(transition_matrix1))))
        rho_2 = max(abs((np.linalg.eigvals(transition_matrix2))))
        decay_max = 1 / (rho_1 * rho_2)
        decay_safe = decay * decay_max

        n, m = len(transition_matrix1), len(transition_matrix2)
        identity_matrix = np.eye(n * m)
        product_matrix = np.kron(transition_matrix1, transition_matrix2)
        ones_vector = np.ones((n * m, 1))
        inverted_matrix = np.linalg.inv(identity_matrix - decay_safe * product_matrix)
        kernel_value = (ones_vector.T @ inverted_matrix @ ones_vector).item()
        if normalize:
            self_similarity_1 = RandomWalkKernel.weighted_random_walk_kernel(
                transition_matrix1, transition_matrix1, decay=decay, normalize=False
            )
            self_similarity_2 = RandomWalkKernel.weighted_random_walk_kernel(
                transition_matrix2, transition_matrix2, decay=decay, normalize=False
            )
            return kernel_value / np.sqrt(self_similarity_1 * self_similarity_2)
        else:
            return kernel_value
