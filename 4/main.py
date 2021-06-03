import numpy as np
from sklearn.datasets import load_digits
from scipy import linalg
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import squareform
from scipy.spatial.distance import pdist
import math

MACHINE_EPSILON = np.finfo(np.double).eps
n_components = 2
perplexity = 30

class TSNE:
    def fit_transform(self, input_vecs):
        self.n_samples = input_vecs.shape[0]
        distances = pairwise_distances(input_vecs, metric='euclidean', squared=True)
        P = self.__calc_P(distances=distances)
        flat_dim_start_point = 1e-4 * np.random.mtrand._rand.randn(self.n_samples, n_components).astype(np.float32)
        return self.__run_tsne(P, flat_dim_start_point)

    def __calc_P(self, distances):
        distances.astype(np.float32)
        conditional_P = self.__compute_p_by_perplexity(distances)
        P = conditional_P + conditional_P.T
        P = np.maximum(squareform(P) / np.maximum(np.sum(P), MACHINE_EPSILON), MACHINE_EPSILON)
        return P

    def __compute_p_by_perplexity(self, distances : np.ndarray):
        n_steps : int = 100
        n_samples : int = distances.shape[0]
        n_neighbors : int = distances.shape[1]
        using_neighbors : int = n_neighbors < n_samples
        beta = 0.0
        beta_min = 0.0
        beta_max = 0.0
        beta_sum = 0.0
        perplexity_log : float = math.log(perplexity)
        entropy_diff = 0.0

        sum_Pi = 0.0
        sum_disti_Pi = 0.0
        i = j = 0

        P : np.ndarray = np.zeros((n_samples, n_neighbors), dtype=np.float64)

        for i in range(n_samples):
            beta_min = -np.inf
            beta_max = np.inf
            beta = 1.0

            for l in range(n_steps):
                sum_Pi = 0.0
                for j in range(n_neighbors):
                    if j != i or using_neighbors:
                        P[i, j] = math.exp(-distances[i, j] * beta)
                        sum_Pi += P[i, j]

                if sum_Pi == 0.0:
                    sum_Pi = MACHINE_EPSILON
                sum_disti_Pi = 0.0

                for j in range(n_neighbors):
                    P[i, j] /= sum_Pi
                    sum_disti_Pi += distances[i, j] * P[i, j]

                entropy_diff = math.log(sum_Pi) + beta * sum_disti_Pi - perplexity_log

                if math.fabs(entropy_diff) <= 1e-5:
                    break

                if entropy_diff > 0.0:
                    beta_min = beta
                    if beta_max == np.inf:
                        beta *= 2.0
                    else:
                        beta = (beta + beta_max) / 2.0
                else:
                    beta_max = beta
                    if beta_min == -np.inf:
                        beta /= 2.0
                    else:
                        beta = (beta + beta_min) / 2.0

            beta_sum += beta
        return P

    def __run_tsne(self, P, input_space):
        input_vec = input_space.ravel()
        input_vec = self.__optimize_by_kl_divergence(input_vec, P)
        input_space = input_vec.reshape(self.n_samples, n_components)
        return input_space

    def __count_kl_divergence(self, vectors, P):
        vectors_in_low_dim = vectors.reshape(self.n_samples, n_components)
        qij_prob = (pdist(vectors_in_low_dim, "sqeuclidean") + 1.0) ** (-1.0)
        Q = np.maximum(qij_prob / (2.0 * np.sum(qij_prob)), MACHINE_EPSILON)

        kl_divergence = 2.0 * \
            np.dot(P, np.log(np.maximum(P, MACHINE_EPSILON) / Q))

        # Gradient: dC/dY
        dCdY = np.ndarray((self.n_samples, n_components), dtype=vectors.dtype)
        PQd = squareform((P - Q) * qij_prob)
        for i in range(self.n_samples):
            dCdY[i] = np.dot(np.ravel(PQd[i], order='K'),
                             vectors_in_low_dim[i] - vectors_in_low_dim)
        dCdY = dCdY.ravel()
        dCdY *= 4.0
        return kl_divergence, dCdY

    def __optimize_by_kl_divergence(self, input_vec, P):
        n_iter = 1000
        n_iter_without_progress = 300
        momentum = 0.8
        learning_rate = 200.0
        min_gain = 0.0
        min_grad_norm = 1e-7

        vec_to_optimize = input_vec.copy()
        optimization_step = np.zeros_like(vec_to_optimize)
        gains = np.ones_like(vec_to_optimize)
        best_error = error = np.finfo(np.float).max
        best_iter = 0

        for i in range(n_iter):
            error, gradients = self.__count_kl_divergence(vec_to_optimize, P)
            grad_norm = linalg.norm(gradients)
            inc = optimization_step * gradients < 0.0
            dec = np.invert(inc)
            gains[inc] += 0.3
            gains[dec] *= 0.7
            np.clip(gains, min_gain, np.inf, out=gains)
            gradients *= gains
            optimization_step = momentum * optimization_step - learning_rate * gradients
            vec_to_optimize += optimization_step

            if error < best_error:
                best_error = error
                best_iter = i
            elif i - best_iter > n_iter_without_progress:
                break

            if grad_norm <= min_grad_norm:
                break

        return vec_to_optimize

X, y = load_digits(return_X_y=True)
tsne_obj = TSNE()
X_flattened = tsne_obj.fit_transform(X)
palette = sns.color_palette("bright", 10)
sns_plot = sns.scatterplot(X_flattened[:,0], X_flattened[:,1], hue=y, legend='full', palette=palette)
sns_plot.figure.savefig("output.png")