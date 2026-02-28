#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Preprocessing Package for Microplastic FTIR Classification
============================================================

Systematic spectral preprocessing with ablation study support.

Modules:
    - techniques: Individual preprocessing operations (baseline correction,
      smoothing, derivatives, normalization)
    - pipeline: Composable preprocessing pipelines
    - ablation_study: Exhaustive evaluation of preprocessing combinations
      with statistical testing (Hypothesis H2)

Design Philosophy:
    Preprocessing is treated as a *first-class experimental variable*,
    not a fixed step. The ablation study systematically evaluates every
    combination and uses statistical tests to determine whether
    preprocessing choice matters more than model selection.

Usage:
------
    >>> from microplastic_ftir.preprocessing import (
    ...     PreprocessingTechniques, PreprocessingPipeline, AblationStudy,
    ... )
    >>>
    >>> # Apply a single technique
    >>> tech = PreprocessingTechniques(config)
    >>> corrected = tech.baseline_als(spectrum)
    >>>
    >>> # Build and apply a pipeline
    >>> pipeline = PreprocessingPipeline(
    ...     steps=['baseline_polynomial', 'savgol_11', 'snv'],
    ...     config=config,
    ... )
    >>> X_processed = pipeline.fit_transform(X)
    >>>
    >>> # Run full ablation study
    >>> study = AblationStudy(config)
    >>> results = study.run(X_train, y_train, X_val, y_val)
"""

from microplastic_ftir.preprocessing.techniques import PreprocessingTechniques
from microplastic_ftir.preprocessing.pipeline import (
    PreprocessingPipeline,
    build_pipeline,
    list_available_pipelines,
)
from microplastic_ftir.preprocessing.ablation_study import AblationStudy

__all__ = [
    "PreprocessingTechniques",
    "PreprocessingPipeline",
    "build_pipeline",
    "list_available_pipelines",
    "AblationStudy",
]