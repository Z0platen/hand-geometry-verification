"""Supervised regularized LDA for the raw Gabor palm descriptor."""

from numbers import Integral, Real

import numpy as np

from palmprint import PALM_FEATURE_SIZE


METRIC_KIND = "regularized_lda_v1"
MAX_COMPONENTS = 64
SHRINKAGE = 0.1


def _finite_array(values, name):
    try:
        array = np.asarray(values)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must contain finite numeric values.") from error
    if array.dtype.kind not in "fiu":
        raise ValueError(f"{name} must contain finite numeric values.")
    array = array.astype(float, copy=False)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite numeric values.")
    return array


def _raw_descriptors(values):
    descriptors = _finite_array(values, "Palm descriptors")
    if descriptors.ndim not in (1, 2) or descriptors.shape[-1] != PALM_FEATURE_SIZE or not descriptors.size:
        raise ValueError(f"Expected one or more raw {PALM_FEATURE_SIZE}-dimensional palm descriptors.")
    if np.any(descriptors < 0) or np.any(descriptors > 1 + 1e-5):
        raise ValueError("Palm descriptors must be non-negative and unit length.")
    norms = np.linalg.norm(descriptors, axis=-1, keepdims=True)
    if not np.allclose(norms, 1.0, rtol=1e-5, atol=1e-8):
        raise ValueError("Palm descriptors must be non-negative and unit length.")
    return descriptors / norms


def _metric_arrays(metric):
    if not isinstance(metric, dict) or metric.get("kind") != METRIC_KIND:
        raise ValueError("Expected a regularized LDA palm metric.")
    input_size = metric.get("input_size")
    output_size = metric.get("output_size")
    sample_count = metric.get("training_samples")
    class_count = metric.get("training_classes")
    shrinkage = metric.get("shrinkage")
    if (type(input_size) is not int or input_size != PALM_FEATURE_SIZE
            or type(sample_count) is not int or type(class_count) is not int
            or not 2 <= class_count <= sample_count
            or type(output_size) is not int or not 1 <= output_size <= min(input_size, class_count - 1)
            or type(shrinkage) not in (int, float) or not 0 < shrinkage <= 1):
        raise ValueError("Invalid dimensions or training settings for the learned palm metric.")
    mean = _finite_array(metric.get("mean"), "Palm metric mean")
    projection = _finite_array(metric.get("projection"), "Palm metric projection")
    if mean.shape != (input_size,) or np.any(mean < 0) or not 0 < np.linalg.norm(mean) <= 1 + 1e-8:
        raise ValueError("Invalid learned palm metric mean.")
    if projection.shape != (input_size, output_size) or not np.any(projection, axis=0).all():
        raise ValueError("Invalid learned palm metric projection.")
    return mean, projection


def validate_metric(metric):
    """Validate a serialized metric while preserving its JSON-ready representation."""
    _metric_arrays(metric)
    return metric


def fit_metric(features, labels, *, max_components=MAX_COMPONENTS, shrinkage=SHRINKAGE):
    """Fit a regularized projection with up to max_components discriminative axes."""
    if (isinstance(max_components, bool) or not isinstance(max_components, Integral)
            or not 1 <= max_components <= PALM_FEATURE_SIZE):
        raise ValueError(f"max_components must be an integer between 1 and {PALM_FEATURE_SIZE}.")
    if (isinstance(shrinkage, bool) or not isinstance(shrinkage, Real)
            or not 0 < shrinkage <= 1 or float(shrinkage) == 0):
        raise ValueError("shrinkage must be a finite number greater than 0 and at most 1.")
    shrinkage = float(shrinkage)

    samples = _raw_descriptors(features)
    labels = list(labels)
    if samples.ndim != 2 or len(labels) != len(samples):
        raise ValueError("Each enrollment descriptor requires a corresponding label.")
    if any(not isinstance(label, str) or not label for label in labels):
        raise ValueError("Enrollment labels must be nonempty subject IDs.")
    classes, membership, counts = np.unique(labels, return_inverse=True, return_counts=True)
    if len(classes) < 2:
        raise ValueError("Training the palm metric requires at least two enrollment identities.")

    mean = samples.mean(axis=0)
    centers = np.zeros((len(classes), PALM_FEATURE_SIZE))
    np.add.at(centers, membership, samples)
    centers /= counts[:, None]
    residuals = samples - centers[membership]
    within = residuals.T @ residuals / len(samples)
    offsets = (centers - mean) * np.sqrt(counts[:, None] / len(samples))
    between = offsets.T @ offsets
    scale = np.trace(within) / PALM_FEATURE_SIZE
    if scale == 0:
        # Repeated or single-shot enrollment can still distinguish class centers.
        scale = np.trace(between) / PALM_FEATURE_SIZE
    if scale <= 0:
        raise ValueError("Training descriptors have no variation across identities.")

    regularized = (1 - shrinkage) * within + shrinkage * scale * np.eye(PALM_FEATURE_SIZE)
    eigenvalues, eigenvectors = np.linalg.eigh(regularized)
    if eigenvalues[0] <= 0:
        raise ValueError("shrinkage is too small for a numerically stable palm metric.")
    whitener = eigenvectors / np.sqrt(eigenvalues)
    scatter = whitener.T @ between @ whitener
    values, vectors = np.linalg.eigh((scatter + scatter.T) / 2)
    order = np.argsort(values)[::-1]
    significant = order[values[order] > max(0.0, values[-1]) * PALM_FEATURE_SIZE * np.finfo(float).eps]
    components = significant[:min(max_components, len(classes) - 1)]
    if not components.size:
        raise ValueError("Training descriptors have no discriminative directions.")

    projection = whitener @ vectors[:, components]
    # Give each axis a consistent sign for serialization.
    largest = np.argmax(np.abs(projection), axis=0)
    projection *= np.sign(projection[largest, np.arange(len(components))])
    return validate_metric({
        "kind": METRIC_KIND,
        "input_size": PALM_FEATURE_SIZE,
        "output_size": len(components),
        "mean": mean.tolist(),
        "projection": projection.tolist(),
        "training_samples": len(samples),
        "training_classes": len(classes),
        "shrinkage": shrinkage,
    })


def project_features(raw, metric):
    """Center, project, and normalize one raw descriptor or a matrix of descriptors."""
    samples = _raw_descriptors(raw)
    mean, projection = _metric_arrays(metric)
    return _project_features_with_arrays(samples, mean, projection)


def _project_features_with_arrays(samples, mean, projection):
    """Project already-normalized descriptors with already-validated arrays."""
    with np.errstate(over="ignore", invalid="ignore"):
        projected = np.einsum("...i,ij->...j", samples - mean, projection)
        norms = np.linalg.norm(projected, axis=-1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms == 0):
        raise ValueError("The learned palm projection must have a finite, nonzero norm.")
    return projected / norms


class _PreparedPalmProjector:
    """Validated projection arrays for repeated probes against one fixed metric."""

    def __init__(self, metric):
        self._mean, self._projection = _metric_arrays(metric)

    def __call__(self, raw):
        return _project_features_with_arrays(_raw_descriptors(raw), self._mean, self._projection)
