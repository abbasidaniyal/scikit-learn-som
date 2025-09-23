# scikit-learn-som

A **pluggable Self-Organizing Map (SOM)** implementation for [scikit-learn](https://scikit-learn.org), designed to integrate seamlessly with the scikit-learn API.

This package provides an efficient, NumPy-optimized implementation of the SOM algorithm with support for multiple distance metrics, learning rate schedules, and lattice structures.

---

## Features

*  **Scikit-learn compatible**: follows the `fit`, `transform`, `predict` API.
*  **Cluster & transformer mixin**: usable for clustering, dimensionality reduction, or feature engineering.
*  **Square lattice support** for grid-based maps.
*  **Hexagonal lattice support** for topologically-aware maps.
*  **Configurable distance metrics**: **L2 (Euclidean)** and **L1 (Manhattan)**.
*  **Flexible learning rate schedules**: exponential, inverse-time, cosine, step, and polynomial decay.
*  **Optimized NumPy implementation** for efficient training on large datasets.
*  **Customizable training hyperparameters** with scikit-learn style validation.

---

## Installation

```bash
pip install scikit-learn-som
```

---

## Quick Start

```python
from som import SOM
import numpy as np

# Example data
X = np.random.rand(500, 10)

# Initialize SOM
som = SOM(
    lattice_rows=20,
    lattice_columns=20,
    lattice_type="hexagonal",         # "square" or "hexagonal"
    distance_metric="euclidean",      # "euclidean" (L2) or "manhattan" (L1)
    max_iters=500,
    initial_learning_rate=0.5,
    learning_rate_type="cosine",
    use_tqdm=True,
)

# Train SOM
som.fit(X)

# Assign each sample to its BMU (Best Matching Unit)
labels = som.predict(X)

# Transform data into SOM feature space
embedding = som.transform(X)
```

---

## Parameters

* **`lattice_rows`** *(int, default=10)*: number of rows in the SOM lattice.
* **`lattice_columns`** *(int, default=10)*: number of columns in the SOM lattice.
* **`neighbourhood_radius`** *(int, optional)*: initial neighborhood radius. If `None`, defaults to `max(lattice_rows, lattice_columns) / 2`.
* **`initial_learning_rate`** *(float, default=1.0)*: initial learning rate.
* **`max_iters`** *(int, default=300)*: maximum number of training iterations.
* **`learning_rate_type`** *({"exponential", "inverse\_time", "cosine", "step", "polynomial"}, default="exponential")*: learning rate decay strategy.
* **`lr_decay_rate`** *(float, default=1e-3)*: decay rate used in certain schedules.
* **`lr_decay_factor`** *(float, default=0.5)*: multiplicative decay factor for step schedule.
* **`lr_step_size`** *(int, default=100)*: number of iterations before each learning rate step (for step schedule).
* **`lr_power`** *(float, default=2.0)*: power used in polynomial decay.
* **`random_state`** *(int, RandomState instance, or None, default=None)*: seed for reproducibility.
* **`verbose`** *(bool, default=False)*: if `True`, print progress during training.
* **`use_tqdm`** *(bool, default=False)*: if `True`, use `tqdm` progress bar for training loop.
* **`lattice_type`** *({"square", "hexagonal"}, default="square")*: type of SOM lattice structure.
* **`distance_metric`** *({"euclidean", "manhattan"}, default="euclidean")*: distance metric for BMU search.

---

## API

* **`fit(X, y=None)`**
  Train the SOM on input data `X`.

* **`predict(X)`**
  Return the indices of the Best Matching Unit (BMU) for each sample.

* **`transform(X)`**
  Map input data into the SOM embedding space.

* **`fit_predict(X, y=None)`**
  Fit the SOM and return BMU assignments for `X`.

---

## License

This project is licensed under the **MIT License**.

