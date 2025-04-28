import numpy as np

from sklearn.base import (
    BaseEstimator,
    ClassNamePrefixFeaturesOutMixin,
    ClusterMixin,
    TransformerMixin,
    _fit_context,
)

from sklearn.utils.validation import (
    check_random_state,
    check_is_fitted,
    validate_data,
)

from numbers import Integral, Real
from sklearn.utils._param_validation import Interval, StrOptions


class SOM(
    ClassNamePrefixFeaturesOutMixin, TransformerMixin, ClusterMixin, BaseEstimator
):
    _parameter_constraints: dict = {
        "lattice_rows": [Interval(Integral, 1, None, closed="left")],
        "lattice_columns": [Interval(Integral, 1, None, closed="left")],
        "neighbourhood_radius": [Interval(Integral, 1, None, closed="left")],
        "initial_learning_rate": [Interval(Real, 0, None, closed="left")],
        "max_iters": [Interval(Integral, 1, None, closed="left")],
        "learning_rate_type": [StrOptions({"exponential", "inverse_time", "cosine", "step", "polynomial"})],
        "lr_decay_rate": [Interval(Real, 0, None, closed="left")],
        "lr_decay_factor": [Interval(Real, 0, None, closed="left")],
        "lr_step_size": [Interval(Integral, 1, None, closed="left")],
        "lr_power": [Interval(Real, 0, None, closed="left")],
        "random_state": ["random_state"],
        "verbose": ["boolean"],
        "use_tqdm": ["boolean"],
        "lattice_type": [StrOptions({"square", "hexagonal"})],
        "distance_metric": [StrOptions({"euclidean", "manhattan"})],
    }



    def __init__(
        self,
        *,
        lattice_rows=10,
        lattice_columns=10,
        initial_learning_rate=1,
        neighbourhood_radius=None,
        max_iters=300,
        learning_rate_type="exponential",
        lr_decay_rate=1e-3,
        lr_decay_factor=0.5,
        lr_step_size=100,
        lr_power=2.0,
        random_state=None,
        verbose=False,
        use_tqdm=False,
        lattice_type="square",
        distance_metric="euclidean",
    ):
        self.lattice_rows = lattice_rows
        self.lattice_columns = lattice_columns
        self.initial_learning_rate = initial_learning_rate
        self.neighbourhood_radius = neighbourhood_radius or max(lattice_rows, lattice_columns) // 2
        self.max_iters = max_iters
        self.random_state = random_state
        self.learning_rate_type = learning_rate_type
        self.lr_decay_rate = lr_decay_rate
        self.lr_decay_factor = lr_decay_factor
        self.lr_step_size = lr_step_size
        self.lr_power = lr_power
        self.verbose = verbose
        self.use_tqdm = use_tqdm
        self.lattice_type = lattice_type
        self.distance_metric = distance_metric

    @property
    def grid_shape(self):
        return (self.lattice_rows, self.lattice_columns)

    def _get_learning_rate(self, itr):
        if self.learning_rate_type == "exponential":
            return self.initial_learning_rate * np.exp(-(itr + 1) / self.max_iters)
        elif self.learning_rate_type == "inverse_time":
            return self.initial_learning_rate / (1 + self.lr_decay_rate * itr)
        elif self.learning_rate_type == "cosine":
            return self.initial_learning_rate * 0.5 * (1 + np.cos(np.pi * itr / self.max_iters))
        elif self.learning_rate_type == "step":
            return self.initial_learning_rate * (self.lr_decay_factor ** (itr // self.lr_step_size))
        elif self.learning_rate_type == "polynomial":
            return self.initial_learning_rate * (1 - itr / self.max_iters) ** self.lr_power
        else:
            raise ValueError(f"Unsupported learning rate type: {self.learning_rate_type}")

    def _compute_distance(self, A, B):
        if self.distance_metric == "euclidean":
            return np.sum((A - B) ** 2, axis=-1)
        elif self.distance_metric == "manhattan":
            return np.sum(np.abs(A - B), axis=-1)
        else:
            raise ValueError(f"Unsupported distance metric: {self.distance_metric}")

    def _neuron_distance(self, grid, bmu_index):
        if self.lattice_type == "square":
            return np.sum((grid - np.array(bmu_index)) ** 2, axis=-1)
        elif self.lattice_type == "hexagonal":
            delta = grid - np.array(bmu_index)
            return np.max(np.abs(delta), axis=-1)
        else:
            raise ValueError(f"Unsupported lattice type: {self.lattice_type}")

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        X = validate_data(
            self,
            X,
            accept_sparse="csr",
            dtype=[np.float64, np.float32],
            order="C",
            accept_large_sparse=False,
        )

        random_state = check_random_state(self.random_state)
        n_samples, n_features = X.shape

        lattice_weights = random_state.rand(self.lattice_rows, self.lattice_columns, n_features)
        best_inertia, best_winner_neurons, best_weights = None, None, None
        inertia_history = []

        grid = np.stack(
            np.meshgrid(np.arange(self.lattice_rows), np.arange(self.lattice_columns), indexing="ij"),
            axis=-1
        )

        if self.use_tqdm:
            from tqdm import trange
            loop = trange(self.max_iters)
        else:
            loop = range(self.max_iters)

        for itr in loop:
            learning_rate = self._get_learning_rate(itr)
            neighbour_hood_factor = self.neighbourhood_radius * np.exp(-(itr + 1) / self.max_iters)

            sample = X[random_state.randint(0, n_samples)]
            diff = lattice_weights - sample.reshape(1, 1, -1)
            dists = self._compute_distance(diff, np.zeros_like(diff))

            bmu_index = np.unravel_index(np.argmin(dists), (self.lattice_rows, self.lattice_columns))

            dists_to_bmu = self._neuron_distance(grid, bmu_index)
            denom = 2 * (neighbour_hood_factor ** 2 + 1e-8)
            neighborhood_influence = np.exp(-dists_to_bmu / denom)

            error = sample - lattice_weights
            lattice_weights += learning_rate * neighborhood_influence[..., np.newaxis] * error

            flat_weights = lattice_weights.reshape(-1, n_features)
            dists = self._compute_distance(X[:, np.newaxis, :], flat_weights[np.newaxis, :, :])

            best_indices = np.argmin(dists, axis=1)
            rows = best_indices // self.lattice_columns
            cols = best_indices % self.lattice_columns

            inertia = np.sum(np.min(dists, axis=1))
            winner_neurons = np.stack((rows, cols), axis=1)

            inertia_history.append(inertia)

            if best_inertia is None or inertia < best_inertia:
                best_inertia = inertia
                best_winner_neurons = winner_neurons
                best_weights = lattice_weights

            if self.verbose and ((itr + 1) % 100) == 0:
                print(
                    f"Iter: {itr+1}: inertia: {inertia:.2f} | Learning Rate: {learning_rate:.3f} | Neighbourhood factor: {neighbour_hood_factor:.3f}"
                )

        self.best_winner_neurons_ = np.array(best_winner_neurons)

        cluster_labels = best_winner_neurons[:, 0] * self.lattice_columns + best_winner_neurons[:, 1]
        self.labels_ = np.array(cluster_labels)
        self.inertia_ = best_inertia
        self.inertia_history_ = np.array(inertia_history)
        self.weights_ = best_weights

        distinct_clusters = len(cluster_labels)
        self.clusters_ = distinct_clusters

        if self.verbose:
            print(f"Number of Unique Clusters: {distinct_clusters}")

        return self

    def predict(self, X, return_inertia=False):
        check_is_fitted(self)

        X = validate_data(
            self,
            X,
            accept_sparse="csr",
            dtype=[np.float64, np.float32],
            order="C",
            accept_large_sparse=False,
        )

        n_features = X.shape[1]
        flat_weights = self.weights_.reshape(-1, n_features)
        dists = self._compute_distance(X[:, np.newaxis, :], flat_weights[np.newaxis, :, :])

        bmu_flat_idx = np.argmin(dists, axis=1)
        rows = bmu_flat_idx // self.lattice_columns
        cols = bmu_flat_idx % self.lattice_columns
        winner_neurons = np.stack((rows, cols), axis=1)

        min_distances = np.min(dists, axis=1)
        inertia = np.sum(min_distances)

        if return_inertia:
            return winner_neurons, inertia
        else:
            return winner_neurons

    def plot_inertia(self):
        """Plot Inertia over Training Epochs."""
        check_is_fitted(self, attributes=["inertia_history_"])
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 4))
        plt.plot(self.inertia_history_)
        plt.title("SOM Inertia Over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Inertia")
        plt.grid(True)
        plt.show()

    def plot_clusters(self, X, y=None):
        """Plot 2D PCA projection of data and SOM grid."""
        check_is_fitted(self, attributes=["weights_"])
        from sklearn.decomposition import PCA
        import matplotlib.pyplot as plt

        X = validate_data(
            self,
            X,
            accept_sparse="csr",
            dtype=[np.float64, np.float32],
            order="C",
            accept_large_sparse=False,
        )

        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)
        grid_coords_2d = pca.transform(self.weights_.reshape(-1, X.shape[1]))

        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y if y is not None else "gray", cmap='viridis', edgecolor='k', alpha=0.6)
        plt.scatter(grid_coords_2d[:, 0], grid_coords_2d[:, 1], c='red', marker='X')
        plt.title("SOM Clusters in PCA Projection")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        if y is not None:
            plt.colorbar(scatter, label='Labels')
        plt.grid(True)
        plt.show()
        
    def plot_umatrix(self):
        """Plot U-Matrix (distance map) of the SOM."""
        check_is_fitted(self, attributes=["weights_"])
        import matplotlib.pyplot as plt

        rows, cols, features = self.weights_.shape
        umatrix = np.zeros((rows, cols))

        # Compute average distance to neighboring neurons
        for i in range(rows):
            for j in range(cols):
                neighbors = []
                if i > 0:
                    neighbors.append(self.weights_[i - 1, j])
                if i < rows - 1:
                    neighbors.append(self.weights_[i + 1, j])
                if j > 0:
                    neighbors.append(self.weights_[i, j - 1])
                if j < cols - 1:
                    neighbors.append(self.weights_[i, j + 1])

                if neighbors:
                    dists = [np.linalg.norm(self.weights_[i, j] - neighbor) for neighbor in neighbors]
                    umatrix[i, j] = np.mean(dists)

        plt.figure(figsize=(8, 6))
        plt.imshow(umatrix, cmap="bone", interpolation="nearest")
        plt.title("SOM U-Matrix (Cluster Boundary Visualization)")
        plt.colorbar(label="Average Distance to Neighbors")
        plt.xlabel("Grid X")
        plt.ylabel("Grid Y")
        plt.show()

