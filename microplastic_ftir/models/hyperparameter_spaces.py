#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Hyperparameter Search Spaces for Optuna Tuning
================================================

Defines the Optuna ``suggest_*`` search space for every model in the
project.  Each space is a callable that takes an ``optuna.Trial``
object and returns a parameter dictionary ready to instantiate the
corresponding model.

This separation means:
    - Model code (traditional.py / deep_learning.py) stays clean
    - Tuning code (tuning/) only needs the trial + space function
    - Spaces are easy to extend or override per experiment

Supported Models:
    Traditional — random_forest, svm_rbf, svm_linear, xgboost,
                  lightgbm, decision_tree
    Deep Learning — cnn_1d, se_cnn

Usage:
------
    >>> import optuna
    >>> from microplastic_ftir.models.hyperparameter_spaces import get_search_space
    >>>
    >>> space_fn = get_search_space('random_forest')
    >>>
    >>> def objective(trial):
    ...     params = space_fn(trial)
    ...     model = RandomForestClassifier(**params)
    ...     ...
"""

from typing import Any, Callable, Dict, List, Optional

from microplastic_ftir.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Type alias for a space function
SpaceFunction = Callable  # (optuna.Trial) -> Dict[str, Any]


# =============================================================================
# TRADITIONAL MODEL SPACES
# =============================================================================

def _space_random_forest(trial) -> Dict[str, Any]:
    """Optuna search space for Random Forest."""
    return {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500, step=50),
        'max_depth': trial.suggest_categorical('max_depth', [None, 10, 20, 30, 50]),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'class_weight': trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample', None]),
        'n_jobs': -1,
        'random_state': 42,
    }


def _space_svm_rbf(trial) -> Dict[str, Any]:
    """Optuna search space for SVM with RBF kernel."""
    return {
        'C': trial.suggest_float('C', 1e-2, 1e3, log=True),
        'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
        'kernel': 'rbf',
        'class_weight': trial.suggest_categorical('class_weight', ['balanced', None]),
        'probability': True,
        'random_state': 42,
        'cache_size': 1000,
    }


def _space_svm_linear(trial) -> Dict[str, Any]:
    """Optuna search space for Linear SVM."""
    return {
        'C': trial.suggest_float('C', 1e-3, 1e2, log=True),
        'kernel': 'linear',
        'class_weight': trial.suggest_categorical('class_weight', ['balanced', None]),
        'probability': True,
        'random_state': 42,
        'cache_size': 1000,
    }


def _space_xgboost(trial) -> Dict[str, Any]:
    """Optuna search space for XGBoost."""
    return {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500, step=50),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('xgb_gamma', 0.0, 5.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'random_state': 42,
        'n_jobs': -1,
        'eval_metric': 'mlogloss',
        'use_label_encoder': False,
    }


def _space_lightgbm(trial) -> Dict[str, Any]:
    """Optuna search space for LightGBM."""
    return {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500, step=50),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 15, 127),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'reg_alpha': trial.suggest_float('lgb_reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('lgb_reg_lambda', 1e-8, 10.0, log=True),
        'class_weight': trial.suggest_categorical('class_weight', ['balanced', None]),
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1,
    }


def _space_decision_tree(trial) -> Dict[str, Any]:
    """Optuna search space for Decision Tree (baseline)."""
    return {
        'max_depth': trial.suggest_categorical('max_depth', [None, 5, 10, 20, 30]),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'class_weight': trial.suggest_categorical('class_weight', ['balanced', None]),
        'random_state': 42,
    }


# =============================================================================
# DEEP LEARNING SPACES
# =============================================================================

def _space_cnn_1d(trial) -> Dict[str, Any]:
    """Optuna search space for 1D-CNN."""
    n_conv_layers = trial.suggest_int('n_conv_layers', 2, 4)

    filters = []
    kernel_sizes = []
    for i in range(n_conv_layers):
        filters.append(trial.suggest_categorical(
            f'filters_{i}', [16, 32, 64, 128, 256],
        ))
        kernel_sizes.append(trial.suggest_categorical(
            f'kernel_size_{i}', [3, 5, 7, 9, 11],
        ))

    return {
        'n_conv_layers': n_conv_layers,
        'filters': filters,
        'kernel_sizes': kernel_sizes,
        'dropout': trial.suggest_float('dropout', 0.1, 0.5),
        'fc_hidden': trial.suggest_categorical('fc_hidden', [64, 128, 256, 512]),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
        'optimizer': trial.suggest_categorical('optimizer', ['adam', 'adamw']),
    }


def _space_se_cnn(trial) -> Dict[str, Any]:
    """Optuna search space for Spectral-SE-CNN."""
    n_conv_layers = trial.suggest_int('n_conv_layers', 2, 4)

    filters = []
    kernel_sizes = []
    for i in range(n_conv_layers):
        filters.append(trial.suggest_categorical(
            f'filters_{i}', [16, 32, 64, 128, 256],
        ))
        kernel_sizes.append(trial.suggest_categorical(
            f'kernel_size_{i}', [3, 5, 7, 9, 11],
        ))

    return {
        'n_conv_layers': n_conv_layers,
        'filters': filters,
        'kernel_sizes': kernel_sizes,
        'se_reduction': trial.suggest_categorical('se_reduction', [4, 8, 16]),
        'dropout': trial.suggest_float('dropout', 0.1, 0.5),
        'fc_hidden': trial.suggest_categorical('fc_hidden', [64, 128, 256, 512]),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
        'optimizer': trial.suggest_categorical('optimizer', ['adam', 'adamw']),
        'scheduler': trial.suggest_categorical('scheduler', [
            'reduce_on_plateau', 'cosine', 'none',
        ]),
    }


# =============================================================================
# REGISTRY
# =============================================================================

_SEARCH_SPACES: Dict[str, SpaceFunction] = {
    'random_forest':   _space_random_forest,
    'svm_rbf':         _space_svm_rbf,
    'svm_linear':      _space_svm_linear,
    'xgboost':         _space_xgboost,
    'lightgbm':        _space_lightgbm,
    'decision_tree':   _space_decision_tree,
    'cnn_1d':          _space_cnn_1d,
    'se_cnn':          _space_se_cnn,
}


def get_search_space(model_name: str) -> SpaceFunction:
    """
    Get the Optuna search space function for a model.

    Parameters
    ----------
    model_name : str
        Model identifier.

    Returns
    -------
    callable
        Function ``(optuna.Trial) -> dict``.

    Raises
    ------
    KeyError
        If model name is unknown.
    """
    if model_name not in _SEARCH_SPACES:
        available = ', '.join(sorted(_SEARCH_SPACES.keys()))
        raise KeyError(
            f"Unknown model: '{model_name}'. Available: {available}"
        )
    return _SEARCH_SPACES[model_name]


def get_all_search_spaces() -> Dict[str, SpaceFunction]:
    """Return all registered search spaces."""
    return dict(_SEARCH_SPACES)


def list_tunable_models() -> List[str]:
    """Return sorted list of all tunable model names."""
    return sorted(_SEARCH_SPACES.keys())


def get_default_params(model_name: str) -> Dict[str, Any]:
    """
    Get sensible default parameters (no Optuna trial needed).

    These are middle-of-the-road values suitable for quick experiments
    or as starting points before tuning.

    Parameters
    ----------
    model_name : str
        Model identifier.

    Returns
    -------
    dict
        Default parameter dictionary.
    """
    defaults = {
        'random_forest': {
            'n_estimators': 200, 'max_depth': None,
            'min_samples_split': 2, 'min_samples_leaf': 1,
            'max_features': 'sqrt', 'class_weight': 'balanced',
            'n_jobs': -1, 'random_state': 42,
        },
        'svm_rbf': {
            'C': 10.0, 'gamma': 'scale', 'kernel': 'rbf',
            'class_weight': 'balanced', 'probability': True,
            'random_state': 42, 'cache_size': 1000,
        },
        'svm_linear': {
            'C': 1.0, 'kernel': 'linear',
            'class_weight': 'balanced', 'probability': True,
            'random_state': 42, 'cache_size': 1000,
        },
        'xgboost': {
            'n_estimators': 200, 'max_depth': 6,
            'learning_rate': 0.1, 'subsample': 0.8,
            'colsample_bytree': 0.8, 'min_child_weight': 1,
            'random_state': 42, 'n_jobs': -1,
            'eval_metric': 'mlogloss', 'use_label_encoder': False,
        },
        'lightgbm': {
            'n_estimators': 200, 'max_depth': -1,
            'learning_rate': 0.1, 'num_leaves': 31,
            'subsample': 0.8, 'colsample_bytree': 0.8,
            'class_weight': 'balanced',
            'random_state': 42, 'n_jobs': -1, 'verbose': -1,
        },
        'decision_tree': {
            'max_depth': None, 'min_samples_split': 2,
            'min_samples_leaf': 1, 'max_features': 'sqrt',
            'class_weight': 'balanced', 'random_state': 42,
        },
        'cnn_1d': {
            'n_conv_layers': 3, 'filters': [32, 64, 128],
            'kernel_sizes': [7, 5, 3], 'dropout': 0.3,
            'fc_hidden': 128, 'learning_rate': 0.001,
            'weight_decay': 1e-4, 'batch_size': 32,
            'optimizer': 'adam',
        },
        'se_cnn': {
            'n_conv_layers': 3, 'filters': [32, 64, 128],
            'kernel_sizes': [7, 5, 3], 'se_reduction': 16,
            'dropout': 0.3, 'fc_hidden': 128,
            'learning_rate': 0.001, 'weight_decay': 1e-4,
            'batch_size': 32, 'optimizer': 'adam',
            'scheduler': 'reduce_on_plateau',
        },
    }

    if model_name not in defaults:
        raise KeyError(f"No defaults for model: '{model_name}'")
    return defaults[model_name]


# =============================================================================
# MAIN (for testing)
# =============================================================================

if __name__ == '__main__':
    print("Testing hyperparameter_spaces...")
    print("=" * 60)

    # Test all models have spaces and defaults
    for name in list_tunable_models():
        space_fn = get_search_space(name)
        default = get_default_params(name)
        print(f"  ✓ {name:20s} | space_fn={space_fn.__name__:25s} | "
              f"defaults={len(default)} params")

    # Test with mock trial
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def _test_objective(trial):
            results = {}
            for name in list_tunable_models():
                space_fn = get_search_space(name)
                params = space_fn(trial)
                results[name] = len(params)
            return 0.0

        study = optuna.create_study()
        study.optimize(_test_objective, n_trials=1)
        trial = study.best_trial
        print(f"\n  ✓ Optuna integration: {len(trial.params)} params sampled")

    except ImportError:
        print("\n  ⚠ Optuna not available — skipping integration test")

    print(f"\nTunable models: {list_tunable_models()}")

    print("\n✅ hyperparameter_spaces tests passed!")
