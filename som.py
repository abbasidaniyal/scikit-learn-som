import numpy as np

from sklearn.base import (
    BaseEstimator,
    ClassNamePrefixFeaturesOutMixin,
    ClusterMixin,
    TransformerMixin,
    _fit_context,
)


from sklearn.utils.validation import (
    _check_sample_weight,
    _is_arraylike_not_scalar,
    check_random_state,
    check_is_fitted,
    validate_data,
)

from numbers import Integral, Real
from sklearn.utils._param_validation import Interval, StrOptions, validate_params


class SOM(
    ClassNamePrefixFeaturesOutMixin, TransformerMixin, ClusterMixin, BaseEstimator
):

    _parameter_constraints: dict = {
        "lattice_rows": [Interval(Integral, 1, None, closed="left")],
        "lattice_columns": [Interval(Integral, 1, None, closed="left")],
        "neighbourhood_radius": [Interval(Integral, 1, None, closed="left")],
        "initial_learning_rate": [Interval(Real, 0, None, closed="left")],
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "verbose": ["verbose"],
        "random_state": ["random_state"],
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
    ):
        self.lattice_rows = lattice_rows
        self.lattice_columns = lattice_columns

        self.initial_learning_rate = initial_learning_rate

        if neighbourhood_radius == None:
            neighbourhood_radius = max(self.lattice_columns, self.lattice_rows) // 2

        self.neighbourhood_radius = neighbourhood_radius

        self.max_iters = max_iters
        self.random_state = random_state
        self.learning_rate_type = learning_rate_type
        self.lr_decay_rate = lr_decay_rate
        self.lr_decay_factor = lr_decay_factor
        self.lr_step_size = lr_step_size
        self.lr_power = lr_power
        self.verbose = verbose

    @property
    def grid_shape(self):
        return (self.lattice_rows, self.lattice_columns)

    def _get_learning_rate(self, itr):
        if self.learning_rate_type == "exponential":
            learning_rate = self.initial_learning_rate * np.exp(
                -(itr + 1) / self.max_iters
            )

        elif self.learning_rate_type == "inverse_time":
            learning_rate = self.initial_learning_rate / (1 + self.lr_decay_rate * itr)

        elif self.learning_rate_type == "cosine":
            learning_rate = (
                self.initial_learning_rate
                * 0.5
                * (1 + np.cos(np.pi * itr / self.max_iters))
            )

        elif self.learning_rate_type == "step":
            learning_rate = self.initial_learning_rate * (
                self.lr_decay_factor ** (itr // self.lr_step_size)
            )

        elif self.learning_rate_type == "polynomial":
            learning_rate = (
                self.initial_learning_rate * (1 - itr / self.max_iters) ** self.lr_power
            )
        return learning_rate

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

        lattice_weights = random_state.rand(
            self.lattice_rows, self.lattice_columns, n_features
        )

        best_inertia, best_winner_neurons, best_weights = None, None, None
        inertia_history = []

        for itr in range(self.max_iters):
            # learning_rate = self.initial_learning_rate * np.exp(
            #     -(itr + 1) / self.max_iters
            # )
            learning_rate = self._get_learning_rate(itr)

            neighbour_hood_factor = self.neighbourhood_radius * np.exp(
                -(itr + 1) / self.max_iters
            )

            input_vector = X[random_state.random_integers(0, n_samples-1)]
            diff = lattice_weights - input_vector.reshape(1, 1, -1)
            dist = np.linalg.norm(diff, axis=2)

            # Finding BMU
            bmu_index = np.unravel_index(
                np.argmin(dist), (self.lattice_rows, self.lattice_rows)
            )

            # Calculating distance of all neurons to BMU
            for row_idx in range(self.lattice_rows):
                for column_idx in range(self.lattice_columns):
                    neuron_position = np.array([row_idx, column_idx])
                    dist_to_bmu = np.linalg.norm(neuron_position - bmu_index) ** 2
                    # Adjusting weights for relevant neurons

                    neighbour_hood_value = np.exp(
                        -dist_to_bmu
                        / (2 * neighbour_hood_factor * neighbour_hood_factor)
                    )
                    error = input_vector - lattice_weights[row_idx, column_idx]

                    # if self.verbose:
                    #     print(f"Update Equation: {learning_rate} * {neighbour_hood_value} * {error} = {learning_rate * neighbour_hood_value * error}")
                    #     print(f"Original: {lattice_weights[row_idx, column_idx]}")
                    lattice_weights[row_idx, column_idx] += (
                        learning_rate * neighbour_hood_value * error
                    )
                    # if self.verbose:
                    #     print(f"Updated: {lattice_weights[row_idx, column_idx]}")

            
            # inertia = np.linalg.norm(X - lattice_weights)
            flat_lattice = lattice_weights.reshape(-1, n_features)  # (n_rows * n_cols, features)

            dists = np.linalg.norm(X[:, np.newaxis, :] - flat_lattice[np.newaxis, :, :], axis=2)  # (num_points, num_units)

            best_indices = np.argmin(dists, axis=1)  # (num_points,)
            rows = best_indices // self.lattice_columns
            cols = best_indices % self.lattice_columns

            inertia = np.sum(np.min(dists, axis=1) ** 2)
            winner_neurons = list(zip(rows, cols))



            inertia_history.append(inertia)

            if best_inertia is None or inertia < best_inertia:
                best_inertia = inertia
                best_winner_neurons = winner_neurons
                best_weights = lattice_weights

            if self.verbose and ((itr+1) % 100) == 0:
                print(
                    f"Iter: {itr+1}: inertia: {inertia:.2f} | Learning Rate: {learning_rate:.3f} | Neighbourhood factor: {neighbour_hood_factor:.3f}"
                )

        self.best_winner_neurons_ = np.array(best_winner_neurons)

        # Map each unique coord to a label
        coord_to_label = {
            (i, j): i * self.lattice_rows + j
            for i in range(self.lattice_rows)
            for j in range(self.lattice_columns)
        }

        # Convert each coord to its label
        cluster_labels = [coord_to_label[coord] for coord in best_winner_neurons]
        self.coord_label_map_ = coord_to_label
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

        n_samples, n_features = X.shape

        winner_neurons = []
        inertia = 0

        for sample_no in range(n_samples):
            input_vector = X[sample_no]
            diff = self.weights_ - input_vector.reshape(1, 1, -1)
            dist = np.linalg.norm(diff, axis=2) ** 2

            # Finding BMU
            bmu_index = np.unravel_index(
                np.argmin(dist), (self.lattice_rows, self.lattice_rows)
            )

            winner_neurons.append(bmu_index)
            inertia += np.sum(np.linalg.norm(diff[bmu_index]))

        if return_inertia:
            return np.array(winner_neurons), inertia
        else:
            return np.array(winner_neurons)
