import numpy as np
import pytest
from som.som import SOM


@pytest.fixture
def dummy_data():
    rng = np.random.RandomState(42)
    X = rng.rand(100, 4)  # 100 samples, 4 features
    return X

def test_som_fit_predict(dummy_data):
    som = SOM(lattice_rows=5, lattice_columns=5, random_state=42, verbose=False)
    som.fit(dummy_data)
    preds = som.predict(dummy_data)
    assert preds.shape == (dummy_data.shape[0], 2)

def test_som_inertia_plot(dummy_data):
    som = SOM(lattice_rows=5, lattice_columns=5, random_state=42)
    som.fit(dummy_data)
    som.plot_inertia()  # Should run without error

def test_som_clusters_plot(dummy_data):
    som = SOM(lattice_rows=5, lattice_columns=5, random_state=42)
    som.fit(dummy_data)
    som.plot_clusters(dummy_data)

def test_som_umatrix_plot(dummy_data):
    som = SOM(lattice_rows=5, lattice_columns=5, random_state=42)
    som.fit(dummy_data)
    som.plot_umatrix()

def test_invalid_learning_rate_type(dummy_data):
    with pytest.raises(ValueError):
        som = SOM(learning_rate_type="invalid")
        som.fit(dummy_data)

def test_invalid_distance_metric(dummy_data):
    som = SOM(distance_metric="invalid")
    with pytest.raises(ValueError):
        som._compute_distance(np.zeros((2, 4)), np.zeros((2, 4)))

def test_invalid_lattice_type(dummy_data):
    som = SOM(lattice_type="invalid")
    with pytest.raises(ValueError):
        som._neuron_distance(np.zeros((5, 5, 2)), (2, 2))


@pytest.mark.parametrize("lr_type", ["exponential", "inverse_time", "cosine", "step", "polynomial"])
def test_get_learning_rate_all_types(lr_type):
    som = SOM(learning_rate_type=lr_type, random_state=42)
    rate = som._get_learning_rate(10)
    assert isinstance(rate, float)
    assert rate >= 0

@pytest.mark.parametrize("distance_metric", ["euclidean", "manhattan"])
def test_compute_distance_all_types(distance_metric):
    som = SOM(distance_metric=distance_metric, random_state=42)
    A = np.array([[1.0, 2.0, 3.0]])
    B = np.array([[4.0, 5.0, 6.0]])
    dist = som._compute_distance(A, B)
    assert isinstance(dist, np.ndarray)
    assert dist.shape == (1, )

@pytest.mark.parametrize("lattice_type", ["square", "hexagonal"])
def test_neuron_distance_all_types(lattice_type):
    som = SOM(lattice_type=lattice_type, random_state=42)
    grid = np.stack(np.meshgrid(np.arange(3), np.arange(3), indexing="ij"), axis=-1)
    bmu_index = (1, 1)
    dist = som._neuron_distance(grid, bmu_index)
    assert isinstance(dist, np.ndarray)
    assert dist.shape == (3, 3)
