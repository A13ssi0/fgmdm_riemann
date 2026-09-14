# fgmdm_riemann

A small wrapper around `pyriemann.classification.FgMDM` for EEG/BCI workflows that use covariance matrices from one or more frequency bands.

The package trains one FgMDM model per band, returns class predictions and probabilities, supports low-confidence rejection, exposes centroid distances, and includes helper methods for band merging and centroid updates.

## Installation

Clone the repository:

```bash
git clone https://github.com/A13ssi0/fgmdm_riemann.git
cd fgmdm_riemann
```

Install in editable mode:

```bash
python -m pip install -e .
```

Or install it from a parent project that includes this repository as a submodule:

```bash
python -m pip install -e path/to/fgmdm_riemann
```

## Dependencies

The package declares the main dependencies in `setup.py`:

```bash
python -m pip install numpy scipy joblib pyriemann scikit-learn pandas tqdm rich
```

## Basic Usage

```python
import numpy as np

from fgmdm_riemann.fgmdm_riemann import FgMDM

classes = [769, 770]

# Shape: bands x samples x channels x channels
covariances = np.random.randn(2, 100, 8, 8)
covariances = covariances @ np.swapaxes(covariances, -1, -2)

labels = np.array([769] * 50 + [770] * 50)

model = FgMDM(njobs=-1)
model.train(covariances, labels, classes, rejectionTh=0.51)

predicted_classes = model.predict(covariances)
predicted_probabilities = model.predict_probabilities(covariances)
```

Input covariance matrices should be symmetric positive definite. In EEG workflows, these matrices are usually computed from filtered signal windows before being passed to `FgMDM`.

## Data Shapes

Most methods expect covariance data shaped as:

```text
bands x samples/windows x channels x channels
```

Labels are expected as:

```text
samples/windows
```

The `classes` argument defines the class labels and their order, for example:

```python
classes = [769, 770]
```

Predictions are returned as:

```text
bands x samples/windows
```

Predicted probabilities are returned as:

```text
bands x samples/windows x classes
```

## Main API

### `FgMDM.train()`

Train one FgMDM model for each frequency band.

```python
model.train(
    data=covariances,
    labels=labels,
    classes=[769, 770],
    idx_train=[],
    idx_val=[],
    rejectionTh=0.51
)
```

Arguments:

- `data`: covariance matrices shaped as bands by samples by channels by channels.
- `labels`: class labels for each sample/window.
- `classes`: list or array of class labels.
- `idx_train`: optional boolean/index vector for training samples.
- `idx_val`: optional boolean/index vector for validation samples.
- `rejectionTh`: optional probability threshold for low-confidence rejection.

During training, the class prints per-band confusion matrices and accuracies.

### `FgMDM.predict()`

Return predicted class labels.

```python
pred = model.predict(covariances, rejectionTh=0.6)
```

If `rejectionTh` is provided, samples whose maximum class probability is below the threshold are returned as `np.nan`.

### `FgMDM.predict_probabilities()`

Return class probabilities.

```python
prob = model.predict_probabilities(covariances, rejectionTh=0.6)
```

If `rejectionTh` is provided, low-confidence samples are set to `np.nan`.

### `FgMDM.get_confusion_matrix()`

Compute one confusion matrix per band.

```python
cf = model.get_confusion_matrix(pred, labels)
```

Rejected samples are ignored. If all samples are rejected for a band, the returned confusion matrix for that band is filled with zeros.

### `FgMDM.get_accuracies()`

Compute one accuracy value per band.

```python
accuracies = model.get_accuracies(pred, labels)
```

Rejected samples are ignored. If all samples are rejected for a band, the accuracy for that band is `np.nan`.

### `FgMDM.get_distances_fromCentroids()`

Return Riemannian distances from each sample to each class centroid.

```python
distances = model.get_distances_fromCentroids(covariances)
```

### `FgMDM.get_centroids()`

Return class centroids and their tangent-space representation.

```python
centroids, tangent_centroids = model.get_centroids()
```

### `FgMDM.merge_bands()`

Merge per-band probabilities using `model.merge_grid`.

```python
merged_probabilities = model.merge_bands(predicted_probabilities)
```

`merge_grid` is currently populated from validation accuracies by `get_merge_grid()`.

### `FgMDM.update()`

Update class centroids from a batch of labeled covariance matrices.

```python
model.update(update_batch, alpha=0.98)
```

The `update_batch` object is expected to provide a `get_batch()` method returning covariance matrices and labels.

## Notes

- This package is a lightweight research utility, not a general-purpose classifier framework.
- The implementation uses `pyriemann.classification.FgMDM` internally.
- The current implementation assumes that class labels in `labels` match the values passed in `classes`.
- Low-confidence rejection is represented with `np.nan`.
- `print_confusion_matrix()` uses Rich to display colored terminal tables.

## License

This project is released under the MIT License. See `LICENSE` for details.
