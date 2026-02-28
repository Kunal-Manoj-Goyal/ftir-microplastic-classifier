#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Preprocessing Pipeline for FTIR Spectral Data
===============================================

Composable, serialisable preprocessing pipelines built from named
technique steps.  A pipeline is simply an ordered list of technique
names that are applied sequentially.

Features:
    - Scikit-learn-compatible ``fit`` / ``transform`` / ``fit_transform`` API
    - JSON-serialisable pipeline specification
    - Automatic generation of all valid pipeline combinations for
      the ablation study
    - Registry of pre-defined "sensible default" pipelines

Usage:
------
    >>> from microplastic_ftir.preprocessing.pipeline import (
    ...     PreprocessingPipeline, build_pipeline,
    ... )
    >>>
    >>> # Build from step names
    >>> pipe = PreprocessingPipeline(
    ...     steps=['baseline_als', 'savgol_11', 'snv'],
    ...     config=config,
    ... )
    >>> X_out = pipe.fit_transform(X)
    >>>
    >>> # Short-hand
    >>> pipe = build_pipeline('baseline_als+savgol_11+snv', config)
    >>> X_out = pipe.fit_transform(X)
"""

import json
import itertools
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field

import numpy as np

from microplastic_ftir.preprocessing.techniques import PreprocessingTechniques
from microplastic_ftir.utils.logging_utils import get_logger

logger = get_logger(__name__)


# =============================================================================
# PIPELINE
# =============================================================================

class PreprocessingPipeline:
    """
    Sequential preprocessing pipeline.

    Parameters
    ----------
    steps : list of str
        Ordered technique names (see ``PreprocessingTechniques.list_techniques()``).
        'none' steps are silently skipped.
    config : Config, optional
        Project configuration.
    name : str, optional
        Human-readable pipeline name.  Auto-generated if None.

    Attributes
    ----------
    steps : list of str
        Active (non-'none') step names.
    name : str
        Pipeline display name.
    techniques : PreprocessingTechniques
        Technique provider.
    is_fitted : bool
        Always True (these transforms are stateless, but the flag
        exists for sklearn compatibility).

    Examples
    --------
    >>> pipe = PreprocessingPipeline(['baseline_als', 'savgol_11', 'snv'])
    >>> X_processed = pipe.fit_transform(X)
    """

    def __init__(
        self,
        steps: List[str],
        config: Optional['Config'] = None,
        name: Optional[str] = None,
    ):
        # Filter out 'none' steps
        self.steps = [s.strip() for s in steps if s.strip().lower() != 'none']
        self.techniques = PreprocessingTechniques(config)
        self.name = name or self._auto_name()
        self.is_fitted = False

        # Validate step names
        available = set(self.techniques.list_techniques())
        for step in self.steps:
            if step not in available:
                raise KeyError(
                    f"Unknown preprocessing step: '{step}'. "
                    f"Available: {sorted(available)}"
                )

    def _auto_name(self) -> str:
        """Generate a name from step names."""
        if not self.steps:
            return 'raw'
        return '+'.join(self.steps)

    # ----- sklearn-compatible API ------------------------------------

    def fit(self, X: np.ndarray, y: Any = None) -> 'PreprocessingPipeline':
        """
        Fit the pipeline.

        All techniques are stateless, so this is a no-op
        (exists for API compatibility).
        """
        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply the pipeline to spectral data.

        Parameters
        ----------
        X : np.ndarray
            Spectral data, shape ``(n_samples, n_features)`` or ``(n_features,)``.

        Returns
        -------
        np.ndarray
            Processed spectra (same shape as input).
        """
        result = X.copy()

        for step_name in self.steps:
            func = self.techniques.get_technique(step_name)
            result = func(result)

            # Safety: replace any NaN/inf introduced by the step
            if not np.all(np.isfinite(result)):
                n_bad = (~np.isfinite(result)).sum()
                logger.debug(
                    "Pipeline step '%s' produced %d non-finite values — replacing with 0",
                    step_name, n_bad,
                )
                result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

        return result

    def fit_transform(self, X: np.ndarray, y: Any = None) -> np.ndarray:
        """Fit and transform in one call."""
        return self.fit(X, y).transform(X)

    # ----- Serialisation -----------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialise pipeline specification."""
        return {
            'name': self.name,
            'steps': self.steps,
        }

    @classmethod
    def from_dict(
        cls,
        spec: Dict[str, Any],
        config: Optional['Config'] = None,
    ) -> 'PreprocessingPipeline':
        """Reconstruct pipeline from dict."""
        return cls(
            steps=spec['steps'],
            config=config,
            name=spec.get('name'),
        )

    def to_json(self) -> str:
        """Serialise to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(
        cls,
        json_str: str,
        config: Optional['Config'] = None,
    ) -> 'PreprocessingPipeline':
        """Reconstruct from JSON string."""
        return cls.from_dict(json.loads(json_str), config=config)

    # ----- Display ------------------------------------------------------

    def __repr__(self) -> str:
        return f"PreprocessingPipeline(name='{self.name}', steps={self.steps})"

    def __str__(self) -> str:
        if not self.steps:
            return "Pipeline: [raw — no preprocessing]"
        step_str = " → ".join(self.steps)
        return f"Pipeline: {step_str}"

    def __len__(self) -> int:
        return len(self.steps)

    def __eq__(self, other) -> bool:
        if not isinstance(other, PreprocessingPipeline):
            return False
        return self.steps == other.steps

    def __hash__(self):
        return hash(tuple(self.steps))


# =============================================================================
# PIPELINE BUILDER
# =============================================================================

def build_pipeline(
    spec: str,
    config: Optional['Config'] = None,
) -> PreprocessingPipeline:
    """
    Build a pipeline from a ``+``-separated string specification.

    Parameters
    ----------
    spec : str
        Pipeline specification, e.g. ``'baseline_als+savgol_11+snv'``.
        Use ``'raw'`` or ``'none'`` for no preprocessing.
    config : Config, optional
        Project configuration.

    Returns
    -------
    PreprocessingPipeline

    Examples
    --------
    >>> pipe = build_pipeline('baseline_als+savgol_11+snv')
    >>> pipe = build_pipeline('raw')  # no preprocessing
    """
    spec = spec.strip()
    if spec.lower() in ('raw', 'none', ''):
        return PreprocessingPipeline(steps=[], config=config, name='raw')

    steps = [s.strip() for s in spec.split('+')]
    return PreprocessingPipeline(steps=steps, config=config)


# =============================================================================
# PIPELINE GENERATION FOR ABLATION STUDY
# =============================================================================

def generate_all_pipelines(
    config: Optional['Config'] = None,
    max_depth: Optional[int] = None,
    include_raw: bool = True,
) -> List[PreprocessingPipeline]:
    """
    Generate all valid preprocessing pipeline combinations
    for the ablation study.

    The pipeline structure is::

        [baseline] → [smoothing] → [derivative] → [normalization]

    Each category can be 'none' (skipped) or one of its techniques.
    This gives ``|B| × |S| × |D| × |N|`` combinations.

    A ``max_depth`` limit can be applied to exclude pipelines
    with too many active steps.

    Parameters
    ----------
    config : Config, optional
        Configuration providing technique lists.
    max_depth : int, optional
        Maximum number of active (non-none) steps.
        Default from ``config.preprocessing.max_pipeline_depth`` or 4.
    include_raw : bool
        Include the raw (no preprocessing) pipeline.

    Returns
    -------
    list of PreprocessingPipeline
        All valid combinations.
    """
    if config is not None:
        pp = config.preprocessing
        baseline_opts = list(pp.baseline_methods)
        smooth_opts = list(pp.smoothing_methods)
        deriv_opts = list(pp.derivative_methods)
        norm_opts = list(pp.normalization_methods)
        if max_depth is None:
            max_depth = pp.max_pipeline_depth
    else:
        baseline_opts = ['none', 'baseline_polynomial', 'baseline_rubberband', 'baseline_als']
        smooth_opts = ['none', 'savgol_5', 'savgol_11', 'savgol_21']
        deriv_opts = ['none', 'first_derivative', 'second_derivative']
        norm_opts = ['none', 'snv', 'vector', 'minmax', 'area']
        if max_depth is None:
            max_depth = 4

    pipelines: List[PreprocessingPipeline] = []

    # Raw pipeline
    if include_raw:
        pipelines.append(
            PreprocessingPipeline(steps=[], config=config, name='raw')
        )

    for b, s, d, n in itertools.product(baseline_opts, smooth_opts, deriv_opts, norm_opts):
        steps = [b, s, d, n]
        active = [st for st in steps if st.lower() != 'none']

        # Skip if empty (that's the raw pipeline)
        if not active:
            continue

        # Skip if over max depth
        if len(active) > max_depth:
            continue

        # Skip contradictory combos: derivative includes implicit smoothing,
        # so having derivative + very heavy smoothing is redundant but not invalid.
        # We include them all and let the ablation study sort out performance.

        try:
            pipe = PreprocessingPipeline(steps=active, config=config)
            pipelines.append(pipe)
        except KeyError:
            # Invalid step name — skip
            continue

    # Deduplicate (same steps but generated via different 'none' combos)
    seen = set()
    unique = []
    for p in pipelines:
        key = tuple(p.steps)
        if key not in seen:
            seen.add(key)
            unique.append(p)

    logger.info(
        "Generated %d unique pipelines (from %d combos, max_depth=%d)",
        len(unique),
        len(baseline_opts) * len(smooth_opts) * len(deriv_opts) * len(norm_opts),
        max_depth,
    )

    return unique


# =============================================================================
# PREDEFINED PIPELINES
# =============================================================================

_PREDEFINED_PIPELINES: Dict[str, List[str]] = {
    'raw':                          [],
    'snv_only':                     ['snv'],
    'minmax_only':                  ['minmax'],
    'als_snv':                      ['baseline_als', 'snv'],
    'poly_snv':                     ['baseline_polynomial', 'snv'],
    'als_savgol11_snv':             ['baseline_als', 'savgol_11', 'snv'],
    'als_savgol11_minmax':          ['baseline_als', 'savgol_11', 'minmax'],
    'als_d1_snv':                   ['baseline_als', 'first_derivative', 'snv'],
    'als_d1_vector':                ['baseline_als', 'first_derivative', 'vector'],
    'als_d2_snv':                   ['baseline_als', 'second_derivative', 'snv'],
    'poly_savgol11_snv':            ['baseline_polynomial', 'savgol_11', 'snv'],
    'rubberband_savgol11_snv':      ['baseline_rubberband', 'savgol_11', 'snv'],
    'd1_only':                      ['first_derivative'],
    'd2_only':                      ['second_derivative'],
    'savgol11_d1_snv':              ['savgol_11', 'first_derivative', 'snv'],
    'savgol21_d2_vector':           ['savgol_21', 'second_derivative', 'vector'],
    'als_savgol11_d1_snv':          ['baseline_als', 'savgol_11', 'first_derivative', 'snv'],
}


def list_available_pipelines() -> List[str]:
    """Return names of all predefined pipelines."""
    return sorted(_PREDEFINED_PIPELINES.keys())


def get_predefined_pipeline(
    name: str,
    config: Optional['Config'] = None,
) -> PreprocessingPipeline:
    """
    Get a predefined pipeline by name.

    Parameters
    ----------
    name : str
        Pipeline name (see ``list_available_pipelines()``).
    config : Config, optional
        Project configuration.

    Returns
    -------
    PreprocessingPipeline
    """
    if name not in _PREDEFINED_PIPELINES:
        available = ', '.join(sorted(_PREDEFINED_PIPELINES.keys()))
        raise KeyError(f"Unknown pipeline: '{name}'. Available: {available}")
    return PreprocessingPipeline(
        steps=_PREDEFINED_PIPELINES[name],
        config=config,
        name=name,
    )


# =============================================================================
# MAIN (for testing)
# =============================================================================

if __name__ == '__main__':
    print("Testing PreprocessingPipeline...")
    print("=" * 60)

    # Synthetic data
    np.random.seed(42)
    n_samples, n_features = 50, 1800
    X = np.random.randn(n_samples, n_features) * 0.1 + 0.5

    # Test basic pipeline
    print("\n--- Basic Pipeline ---")
    pipe = PreprocessingPipeline(['baseline_als', 'savgol_11', 'snv'])
    print(f"  {pipe}")
    print(f"  repr: {repr(pipe)}")
    X_out = pipe.fit_transform(X)
    assert X_out.shape == X.shape, "Shape changed"
    assert np.all(np.isfinite(X_out)), "Non-finite output"
    print(f"  ✓ shape={X_out.shape}, all finite")

    # Test from string
    print("\n--- Build from string ---")
    pipe2 = build_pipeline('baseline_polynomial+savgol_21+minmax')
    print(f"  {pipe2}")
    X_out2 = pipe2.fit_transform(X)
    assert X_out2.shape == X.shape
    print(f"  ✓ shape={X_out2.shape}")

    # Test raw
    pipe_raw = build_pipeline('raw')
    X_raw = pipe_raw.fit_transform(X)
    assert np.allclose(X_raw, X), "Raw pipeline should be identity"
    print(f"  ✓ raw pipeline is identity")

    # Test serialisation round-trip
    print("\n--- Serialisation ---")
    spec = pipe.to_dict()
    pipe_restored = PreprocessingPipeline.from_dict(spec)
    assert pipe_restored.steps == pipe.steps
    X_restored = pipe_restored.transform(X)
    assert np.allclose(X_out, X_restored), "Serialisation changed output"
    print(f"  ✓ dict round-trip")

    json_str = pipe.to_json()
    pipe_json = PreprocessingPipeline.from_json(json_str)
    assert pipe_json.steps == pipe.steps
    print(f"  ✓ JSON round-trip: {json_str}")

    # Test predefined pipelines
    print("\n--- Predefined Pipelines ---")
    for name in list_available_pipelines():
        p = get_predefined_pipeline(name)
        result = p.fit_transform(X)
        assert result.shape == X.shape
        assert np.all(np.isfinite(result))
        print(f"  ✓ {name:30s} | steps={len(p)}")

    # Test pipeline generation
    print("\n--- Pipeline Generation ---")
    all_pipes = generate_all_pipelines(max_depth=3)
    print(f"  Generated {len(all_pipes)} pipelines (max_depth=3)")

    all_pipes_4 = generate_all_pipelines(max_depth=4)
    print(f"  Generated {len(all_pipes_4)} pipelines (max_depth=4)")

    # Verify no duplicates
    step_tuples = [tuple(p.steps) for p in all_pipes]
    assert len(step_tuples) == len(set(step_tuples)), "Duplicate pipelines!"
    print(f"  ✓ No duplicates")

    # Show a few examples
    print("\n  Examples:")
    for p in all_pipes[:10]:
        print(f"    {p.name}")
    print(f"    ... ({len(all_pipes) - 10} more)")

    print("\n✅ All PreprocessingPipeline tests passed!")