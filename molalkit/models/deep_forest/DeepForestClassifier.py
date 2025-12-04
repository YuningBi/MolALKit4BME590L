#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Use standalone Deep Forest implementation (no C extensions required)
from .deepforest_src import CascadeForestClassifier
from molalkit.models.base import BaseSklearnModel


class DFClassifier(BaseSklearnModel):
    """
    Deep Forest Classifier for MolALKit.

    This class wraps the CascadeForestClassifier from the deepforest package
    to be compatible with MolALKit's active learning framework.

    Deep Forest has internal state (layers_, binners_, counters) that makes
    refitting problematic. To handle this, we create a new internal model
    instance each time fit is called.

    Parameters
    ----------
    n_estimators : int, default=4
        The number of trees in each random forest or extra trees in each cascade layer.
        This is different from standard Random Forest - in Deep Forest, this is per layer.

    max_layers : int, default=10
        The maximum number of cascade layers. The model will stop adding layers if
        performance on validation set doesn't improve.

    **kwargs : dict
        Additional parameters passed to CascadeForestClassifier, including:
        - max_depth : int, maximum depth of trees (default: None, unlimited)
        - min_samples_split : int (default: 2)
        - min_samples_leaf : int (default: 1)
        - n_jobs : int, number of parallel jobs (default: -1 for all cores)
        - random_state : int, random seed for reproducibility

    Notes
    -----
    - Deep Forest doesn't have oob_score parameter like Random Forest
    - Uncertainty is estimated using variance of predicted probabilities
    - The model cascades multiple layers of forests for hierarchical learning
    - Each fit() call creates a fresh model instance to avoid state conflicts

    References
    ----------
    Zhou, Z. H., & Feng, J. (2019). Deep forest. National science review, 6(1), 74-86.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the Deep Forest wrapper.

        Parameters are stored and used to create fresh model instances on each fit.
        """
        self._init_args = args
        self._init_kwargs = kwargs
        self._model = None

    def _create_new_model(self):
        """Create a fresh Deep Forest model instance."""
        return CascadeForestClassifier(*self._init_args, **self._init_kwargs)

    def fit(self, X, y, sample_weight=None):
        """
        Fit a fresh Deep Forest model.

        This method creates a new internal model instance each time to avoid
        state conflicts from Deep Forest's internal counters and containers.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        sample_weight : array-like of shape (n_samples,), optional
            Sample weights.

        Returns
        -------
        self : DFClassifier
            Fitted model.
        """
        # Create a brand new model instance to avoid refitting issues
        self._model = self._create_new_model()
        self._model.fit(X, y, sample_weight)
        return self

    def predict_proba(self, X):
        """
        Predict class probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Class probabilities.
        """
        if self._model is None:
            raise RuntimeError("Model must be fitted before calling predict_proba")
        return self._model.predict_proba(X)

    def predict(self, X):
        """
        Predict class labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        if self._model is None:
            raise RuntimeError("Model must be fitted before calling predict")
        return self._model.predict(X)

    def fit_molalkit(self, train_data, iteration: int = 0):
        """
        Fit the Deep Forest model using MolALKit data format.

        Parameters
        ----------
        train_data : Dataset
            MolALKit dataset object with X (features) and y (labels) attributes.
        iteration : int, default=0
            Current iteration of active learning (not used by Deep Forest,
            but kept for API consistency).

        Returns
        -------
        self : DFClassifier
            Fitted model.
        """
        return self.fit_molalkit_(train_data, self)

    def predict_uncertainty(self, pred_data):
        """
        Predict uncertainty for samples in pred_data.

        For Deep Forest, uncertainty is computed based on the variance of
        predicted class probabilities. Higher variance indicates higher uncertainty.

        This is the same method used by Random Forest in MolALKit.

        Parameters
        ----------
        pred_data : Dataset
            MolALKit dataset object with X (features) attribute.

        Returns
        -------
        uncertainty : np.ndarray
            Uncertainty scores for each sample. Higher values indicate more uncertainty.
            Scaled to [0, 1] range where:
            - 0.0 = completely certain (probability at [0, 1] or [1, 0])
            - 1.0 = maximum uncertainty (probability at [0.5, 0.5])
        """
        return self.predict_uncertainty_c(pred_data, self)

    def predict_value(self, pred_data):
        """
        Predict class probabilities for positive class.

        For binary classification in MolALKit, this returns the probability
        of the positive class (class 1).

        Parameters
        ----------
        pred_data : Dataset
            MolALKit dataset object with X (features) attribute.

        Returns
        -------
        probabilities : np.ndarray
            Probability of positive class for each sample, shape (n_samples,).
        """
        X = pred_data.X
        proba = self.predict_proba(X)

        # DeepForest may return shape (n_samples, 1) for binary classification
        # when only one class is present during training
        if proba.shape[1] == 1:
            # Assuming single column is probability of class 1
            return proba[:, 0]
        else:
            # Standard binary classification with 2 columns
            return proba[:, 1]
