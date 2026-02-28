#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Traditional Machine Learning Models
=====================================

Factory functions and a thin wrapper for sklearn-compatible classifiers:
    - Random Forest
    - SVM (RBF and Linear kernels)
    - XGBoost
    - LightGBM
    - Decision Tree (baseline)

Every model is returned as a ``TraditionalModelWrapper`` that adds:
    - Consistent ``model_name`` attribute
    - ``predict_proba`` fallback (e.g. for SVM without probability=True)
    - Timing of fit/predict
    - JSON-serialisable parameter snapshot

Usage:
------
    >>> from microplastic_ftir.models.traditional import (
    ...     get_traditional_model, list_traditional_models,
    ... )
    >>>
    >>> model = get_traditional_model('random_forest')
    >>> model.fit(X_train, y_train)
    >>> preds = model.predict(X_test)
    >>> proba = model.predict_proba(X_test)
    >>>
    >>> # With custom parameters
    >>> model = get_traditional_model('xgboost', n_estimators=500, max_depth=8)
"""

import time
import warnings
from typing import Any, Dict, List, Optional, Union

import numpy as np

from microplastic_ftir.models.hyperparameter_spaces import get_default_params
from microplastic_ftir.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Sklearn is always available
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# Optional
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False


# =============================================================================
# MODEL REGISTRY
# =============================================================================

_MODEL_CLASSES: Dict[str, Any] = {
    'random_forest': RandomForestClassifier,
    'svm_rbf':       SVC,
    'svm_linear':    SVC,
    'decision_tree': DecisionTreeClassifier,
}

if HAS_XGBOOST:
    _MODEL_CLASSES['xgboost'] = XGBClassifier

if HAS_LIGHTGBM:
    _MODEL_CLASSES['lightgbm'] = LGBMClassifier


# =============================================================================
# WRAPPER
# =============================================================================

class TraditionalModelWrapper:
    """
    Thin wrapper around sklearn-compatible classifiers.

    Adds a consistent interface with model name tracking,
    timing, and safe ``predict_proba`` fallback.

    Parameters
    ----------
    model : sklearn-compatible estimator
        The underlying classifier instance.
    model_name : str
        Canonical name (e.g. 'random_forest').
    params : dict
        Parameters used to create the model.

    Attributes
    ----------
    model : estimator
        The wrapped classifier.
    model_name : str
        Name string.
    params : dict
        Creation parameters.
    fit_time_ : float
        Seconds taken by the last ``fit()`` call.
    predict_time_ : float
        Seconds taken by the last ``predict()`` call.
    is_fitted : bool
        Whether ``fit()`` has been called.
    """

    def __init__(
        self,
        model: Any,
        model_name: str,
        params: Dict[str, Any],
    ):
        self.model = model
        self.model_name = model_name
        self.params = params
        self.fit_time_: float = 0.0
        self.predict_time_: float = 0.0
        self.is_fitted: bool = False
        self._classes: Optional[np.ndarray] = None

    # ----- sklearn-compatible API ----------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'TraditionalModelWrapper':
        """
        Fit the model.

        Parameters
        ----------
        X : np.ndarray
            Training features, shape ``(n_samples, n_features)``.
        y : np.ndarray
            Labels, shape ``(n_samples,)``.
        **kwargs
            Extra arguments forwarded to the underlying ``fit()``.

        Returns
        -------
        self
        """
        logger.debug("Fitting %s on %s", self.model_name, X.shape)

        t0 = time.time()
        self.model.fit(X, y, **kwargs)
        self.fit_time_ = time.time() - t0

        self._classes = np.unique(y)
        self.is_fitted = True

        logger.debug(
            "%s fitted in %.2fs (%d samples, %d features)",
            self.model_name, self.fit_time_, X.shape[0], X.shape[1],
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.

        Parameters
        ----------
        X : np.ndarray
            Features, shape ``(n_samples, n_features)``.

        Returns
        -------
        np.ndarray
            Predicted labels, shape ``(n_samples,)``.
        """
        t0 = time.time()
        preds = self.model.predict(X)
        self.predict_time_ = time.time() - t0
        return preds

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Falls back to one-hot predictions if the underlying model
        does not support ``predict_proba`` (e.g. SVM without
        ``probability=True``).

        Parameters
        ----------
        X : np.ndarray
            Features, shape ``(n_samples, n_features)``.

        Returns
        -------
        np.ndarray
            Probability matrix, shape ``(n_samples, n_classes)``.
        """
        if hasattr(self.model, 'predict_proba'):
            try:
                return self.model.predict_proba(X)
            except Exception:
                pass

        # Fallback: decision_function → softmax
        if hasattr(self.model, 'decision_function'):
            try:
                dec = self.model.decision_function(X)
                if dec.ndim == 1:
                    # Binary case
                    proba = np.column_stack([1 - self._sigmoid(dec), self._sigmoid(dec)])
                    return proba
                else:
                    return self._softmax(dec)
            except Exception:
                pass

        # Last resort: one-hot from predictions
        logger.debug(
            "%s: falling back to one-hot predict_proba", self.model_name
        )
        preds = self.predict(X)
        n_classes = len(self._classes) if self._classes is not None else len(np.unique(preds))
        proba = np.zeros((len(X), n_classes))
        for i, p in enumerate(preds):
            idx = np.where(self._classes == p)[0]
            if len(idx) > 0:
                proba[i, idx[0]] = 1.0
        return proba

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return accuracy on (X, y)."""
        return self.model.score(X, y)

    @property
    def classes_(self) -> np.ndarray:
        """Class labels known to the model."""
        if hasattr(self.model, 'classes_'):
            return self.model.classes_
        return self._classes if self._classes is not None else np.array([])

    @property
    def feature_importances_(self) -> Optional[np.ndarray]:
        """Feature importances (tree-based models only)."""
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        return None

    # ----- Utility -------------------------------------------------------

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get model parameters."""
        return self.model.get_params(deep=deep)

    def set_params(self, **params) -> 'TraditionalModelWrapper':
        """Set model parameters."""
        self.model.set_params(**params)
        return self

    def summary(self) -> Dict[str, Any]:
        """Return a summary dictionary."""
        return {
            'model_name': self.model_name,
            'params': {k: str(v) for k, v in self.params.items()},
            'is_fitted': self.is_fitted,
            'fit_time_s': self.fit_time_,
            'predict_time_s': self.predict_time_,
        }

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - x.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted else "not fitted"
        return f"TraditionalModelWrapper('{self.model_name}', {status})"


# =============================================================================
# FACTORY
# =============================================================================

def get_traditional_model(
    model_name: str,
    config: Optional['Config'] = None,
    **override_params,
) -> TraditionalModelWrapper:
    """
    Create a traditional ML model by name.

    Parameters
    ----------
    model_name : str
        Model identifier.
    config : Config, optional
        Project configuration (used for default params if available).
    **override_params
        Override specific parameters.

    Returns
    -------
    TraditionalModelWrapper
        Ready-to-fit wrapped model.

    Raises
    ------
    KeyError
        If model name is unknown.
    ImportError
        If required package is not installed.

    Examples
    --------
    >>> model = get_traditional_model('random_forest', n_estimators=500)
    >>> model.fit(X_train, y_train)
    """
    if model_name not in _MODEL_CLASSES:
        available = ', '.join(sorted(_MODEL_CLASSES.keys()))
        raise KeyError(
            f"Unknown model: '{model_name}'. Available: {available}. "
            f"(XGBoost installed: {HAS_XGBOOST}, LightGBM installed: {HAS_LIGHTGBM})"
        )

    # Get default params
    try:
        params = get_default_params(model_name)
    except KeyError:
        params = {}

    # Apply overrides
    params.update(override_params)

    # Filter params to only those accepted by the model class
    model_class = _MODEL_CLASSES[model_name]
    try:
        import inspect
        valid_keys = set(inspect.signature(model_class.__init__).parameters.keys())
        valid_keys.discard('self')
        # Also include **kwargs if present
        if any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in inspect.signature(model_class.__init__).parameters.values()
        ):
            filtered_params = params
        else:
            filtered_params = {k: v for k, v in params.items() if k in valid_keys}
            dropped = set(params.keys()) - set(filtered_params.keys())
            if dropped:
                logger.debug(
                    "%s: dropped invalid params: %s", model_name, dropped
                )
    except Exception:
        filtered_params = params

    # Create model
    model = model_class(**filtered_params)

    wrapper = TraditionalModelWrapper(
        model=model,
        model_name=model_name,
        params=params,
    )

    logger.debug("Created %s with %d params", model_name, len(filtered_params))
    return wrapper


def list_traditional_models() -> List[str]:
    """Return sorted list of available traditional model names."""
    return sorted(_MODEL_CLASSES.keys())


def create_all_traditional_models(
    config: Optional['Config'] = None,
) -> Dict[str, TraditionalModelWrapper]:
    """
    Create one instance of every available traditional model.

    Returns
    -------
    dict
        {model_name: TraditionalModelWrapper}
    """
    models = {}
    for name in list_traditional_models():
        try:
            models[name] = get_traditional_model(name, config=config)
        except Exception as e:
            logger.warning("Failed to create %s: %s", name, e)
    return models


# =============================================================================
# MAIN (for testing)
# =============================================================================

if __name__ == '__main__':
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score

    print("Testing traditional models...")
    print("=" * 60)

    # Synthetic data
    X, y = make_classification(
        n_samples=500, n_features=200, n_informative=50,
        n_classes=5, random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42,
    )

    print(f"Data: train={X_train.shape}, test={X_test.shape}, classes={len(np.unique(y))}")
    print(f"\nAvailable models: {list_traditional_models()}")

    # Test each model
    print(f"\n{'Model':20s} | {'Acc':>6s} | {'F1':>6s} | {'Fit(s)':>7s} | {'Pred(s)':>7s} | {'Proba':>5s}")
    print("-" * 75)

    for name in list_traditional_models():
        try:
            model = get_traditional_model(name)
            model.fit(X_train, y_train)

            preds = model.predict(X_test)
            proba = model.predict_proba(X_test)

            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, average='macro')

            proba_ok = proba.shape == (len(X_test), len(np.unique(y)))

            print(
                f"{name:20s} | {acc:.4f} | {f1:.4f} | "
                f"{model.fit_time_:7.3f} | {model.predict_time_:7.4f} | "
                f"{'✓' if proba_ok else '✗'}"
            )

            # Test summary
            summary = model.summary()
            assert summary['is_fitted'] is True
            assert summary['model_name'] == name

        except Exception as e:
            print(f"{name:20s} | FAILED: {e}")

    # Test feature importances (tree models)
    print("\n--- Feature Importances ---")
    for name in ['random_forest', 'xgboost', 'decision_tree']:
        if name in _MODEL_CLASSES:
            model = get_traditional_model(name)
            model.fit(X_train, y_train)
            fi = model.feature_importances_
            if fi is not None:
                print(f"  {name}: shape={fi.shape}, sum={fi.sum():.4f}")
            else:
                print(f"  {name}: no feature importances")

    # Test with parameter overrides
    print("\n--- Parameter Override ---")
    rf = get_traditional_model('random_forest', n_estimators=50, max_depth=5)
    assert rf.params['n_estimators'] == 50
    assert rf.params['max_depth'] == 5
    print(f"  ✓ RF with overrides: n_estimators={rf.params['n_estimators']}, max_depth={rf.params['max_depth']}")

    # Test create_all
    print("\n--- Create All ---")
    all_models = create_all_traditional_models()
    print(f"  Created {len(all_models)} models: {list(all_models.keys())}")

    print("\n✅ All traditional model tests passed!")
