"""A wrapper on the base estimator for the naming consistency.

Modified to use only sklearn backend (no custom C extensions).
"""


__all__ = ["Estimator"]

import numpy as np
# Only use sklearn backend - no custom C extensions
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    RandomForestRegressor,
    ExtraTreesRegressor,
)


def make_classifier_estimator(
    name,
    criterion,
    n_trees=100,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    backend="sklearn",  # Force sklearn backend
    n_jobs=None,
    random_state=None,
):
    # RandomForestClassifier
    if name == "rf":
        estimator = RandomForestClassifier(
            criterion=criterion,
            n_estimators=n_trees,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            bootstrap=True,
            oob_score=True,
            n_jobs=n_jobs,
            random_state=random_state,
        )
    # ExtraTreesClassifier
    elif name == "erf":
        estimator = ExtraTreesClassifier(
            criterion=criterion,
            n_estimators=n_trees,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            bootstrap=True,
            oob_score=True,
            n_jobs=n_jobs,
            random_state=random_state,
        )
    else:
        msg = "Unknown type of estimator, which should be one of {{rf, erf}}."
        raise NotImplementedError(msg)

    return estimator


def make_regressor_estimator(
    name,
    criterion,
    n_trees=100,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    backend="sklearn",  # Force sklearn backend
    n_jobs=None,
    random_state=None,
):
    # Map classifier criteria to regressor criteria
    # Classifier uses 'gini'/'entropy', Regressor uses 'squared_error'/'absolute_error'
    criterion_mapping = {
        'gini': 'squared_error',
        'entropy': 'absolute_error',
        'mse': 'squared_error',  # old sklearn name
        'mae': 'absolute_error',  # old sklearn name
    }
    regressor_criterion = criterion_mapping.get(criterion, criterion)

    # RandomForestRegressor
    if name == "rf":
        estimator = RandomForestRegressor(
            criterion=regressor_criterion,
            n_estimators=n_trees,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            bootstrap=True,
            oob_score=True,
            n_jobs=n_jobs,
            random_state=random_state,
        )
    # ExtraTreesRegressor
    elif name == "erf":
        estimator = ExtraTreesRegressor(
            criterion=regressor_criterion,
            n_estimators=n_trees,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            bootstrap=True,
            oob_score=True,
            n_jobs=n_jobs,
            random_state=random_state,
        )
    else:
        msg = "Unknown type of estimator, which should be one of {{rf, erf}}."
        raise NotImplementedError(msg)

    return estimator


class Estimator(object):
    def __init__(
        self,
        name,
        criterion,
        n_trees=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        backend="custom",
        n_jobs=None,
        random_state=None,
        is_classifier=True,
    ):

        self.backend = backend
        self.is_classifier = is_classifier
        if self.is_classifier:
            self.estimator_ = make_classifier_estimator(
                name,
                criterion,
                n_trees,
                max_depth,
                min_samples_split,
                min_samples_leaf,
                backend,
                n_jobs,
                random_state,
            )
        else:
            self.estimator_ = make_regressor_estimator(
                name,
                criterion,
                n_trees,
                max_depth,
                min_samples_split,
                min_samples_leaf,
                backend,
                n_jobs,
                random_state,
            )

    @property
    def oob_decision_function_(self):
        # Scikit-Learn uses `oob_prediction_` for ForestRegressor
        if self.backend == "sklearn" and not self.is_classifier:
            oob_prediction = self.estimator_.oob_prediction_
            if len(oob_prediction.shape) == 1:
                oob_prediction = np.expand_dims(oob_prediction, 1)
            return oob_prediction
        return self.estimator_.oob_decision_function_

    @property
    def feature_importances_(self):
        """Return the impurity-based feature importances from the estimator."""

        return self.estimator_.feature_importances_

    def fit_transform(self, X, y, sample_weight=None):
        self.estimator_.fit(X, y, sample_weight)
        return self.oob_decision_function_

    def transform(self, X):
        """Preserved for the naming consistency."""
        return self.predict(X)

    def predict(self, X):
        if self.is_classifier:
            return self.estimator_.predict_proba(X)
        pred = self.estimator_.predict(X)
        if len(pred.shape) == 1:
            pred = np.expand_dims(pred, 1)
        return pred
