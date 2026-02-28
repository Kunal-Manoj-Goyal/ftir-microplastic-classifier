#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Preprocessing Techniques for FTIR Spectral Data
=================================================

Individual, stateless preprocessing operations that can be composed
into pipelines.  Every function operates on either:
    - a single 1-D spectrum (np.ndarray of shape ``(n_wavenumbers,)``)
    - a 2-D spectral matrix (``(n_samples, n_wavenumbers)``)

Each function follows the signature::

    result = func(X, **kwargs)

where ``X`` may be 1-D or 2-D and the function handles both transparently.

Technique Categories:
    1. Baseline Correction  — polynomial, rubberband, ALS
    2. Smoothing            — Savitzky-Golay
    3. Derivatives          — 1st and 2nd order (via Savitzky-Golay)
    4. Normalization        — SNV, vector (L2), min-max, area

All techniques are collected in the ``PreprocessingTechniques`` class
which reads default parameters from ``Config.preprocessing``.

Usage:
------
    >>> from microplastic_ftir.preprocessing.techniques import PreprocessingTechniques
    >>> tech = PreprocessingTechniques(config)
    >>>
    >>> X_bl = tech.baseline_polynomial(X)
    >>> X_sm = tech.savgol_smooth(X, window_length=11)
    >>> X_d1 = tech.first_derivative(X)
    >>> X_n  = tech.snv(X)
"""

import warnings
from typing import Optional, Tuple, Union

import numpy as np
from scipy.signal import savgol_filter
from scipy import sparse
from scipy.sparse.linalg import spsolve

from microplastic_ftir.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Optional: BaselineRemoval package
try:
    from BaselineRemoval import BaselineRemoval
    HAS_BASELINE_REMOVAL = True
except ImportError:
    HAS_BASELINE_REMOVAL = False


# =============================================================================
# HELPER
# =============================================================================

def _ensure_2d(X: np.ndarray) -> Tuple[np.ndarray, bool]:
    """
    Ensure X is 2-D.  Returns (X_2d, was_1d).

    If input is 1-D, it is reshaped to ``(1, n_features)`` and
    ``was_1d=True`` so the caller can squeeze back if needed.
    """
    if X.ndim == 1:
        return X.reshape(1, -1), True
    return X, False


def _maybe_squeeze(X: np.ndarray, was_1d: bool) -> np.ndarray:
    """Squeeze back to 1-D if the original input was 1-D."""
    if was_1d:
        return X.squeeze(axis=0)
    return X


# =============================================================================
# PREPROCESSING TECHNIQUES
# =============================================================================

class PreprocessingTechniques:
    """
    Collection of spectral preprocessing operations.

    All methods accept 1-D or 2-D arrays and return the same
    dimensionality.

    Parameters
    ----------
    config : Config, optional
        Project configuration.  If None, uses sensible defaults.
    """

    def __init__(self, config: Optional['Config'] = None):
        self.config = config

        # Pull defaults from config or use hardcoded values
        if config is not None:
            pp = config.preprocessing
            self.poly_degree = pp.polynomial_degree
            self.als_lam = pp.als_lam
            self.als_p = pp.als_p
            self.als_niter = pp.als_n_iter
            self.savgol_polyorder = pp.savgol_polyorder
            self.deriv_window = pp.derivative_window
            self.deriv_polyorder = pp.derivative_polyorder
            self.minmax_range = pp.minmax_range
        else:
            self.poly_degree = 2
            self.als_lam = 1e6
            self.als_p = 0.01
            self.als_niter = 10
            self.savgol_polyorder = 2
            self.deriv_window = 11
            self.deriv_polyorder = 2
            self.minmax_range = (0.0, 1.0)

    # =====================================================================
    # 1.  BASELINE CORRECTION
    # =====================================================================

    def baseline_polynomial(
        self,
        X: np.ndarray,
        degree: Optional[int] = None,
    ) -> np.ndarray:
        """
        Polynomial baseline correction.

        Fits a polynomial of the specified degree to the spectrum
        and subtracts it.

        Parameters
        ----------
        X : np.ndarray
            1-D or 2-D spectral data.
        degree : int, optional
            Polynomial degree (default from config).

        Returns
        -------
        np.ndarray
            Baseline-corrected spectra.
        """
        degree = degree or self.poly_degree
        X, was_1d = _ensure_2d(X)
        result = np.empty_like(X)
        n_points = X.shape[1]
        x = np.arange(n_points, dtype=np.float64)

        for i in range(X.shape[0]):
            spectrum = X[i]
            valid = np.isfinite(spectrum)
            if valid.sum() < degree + 1:
                result[i] = spectrum
                continue

            coeffs = np.polyfit(x[valid], spectrum[valid], degree)
            baseline = np.polyval(coeffs, x)
            result[i] = spectrum - baseline

        return _maybe_squeeze(result, was_1d)

    def baseline_rubberband(self, X: np.ndarray) -> np.ndarray:
        """
        Rubberband (convex-hull) baseline correction.

        Computes the lower convex hull of the spectrum and subtracts it.

        Parameters
        ----------
        X : np.ndarray
            1-D or 2-D spectral data.

        Returns
        -------
        np.ndarray
            Baseline-corrected spectra.
        """
        from scipy.spatial import ConvexHull

        X, was_1d = _ensure_2d(X)
        result = np.empty_like(X)
        n_points = X.shape[1]
        x = np.arange(n_points, dtype=np.float64)

        for i in range(X.shape[0]):
            spectrum = X[i].copy()

            try:
                # Build 2-D points for convex hull
                points = np.column_stack([x, spectrum])
                hull = ConvexHull(points)

                # Extract lower hull vertices (sorted by x)
                hull_vertices = sorted(set(hull.vertices))
                hull_x = x[hull_vertices]
                hull_y = spectrum[hull_vertices]

                # Keep only the lower envelope:
                # vertices where the spectrum values are relatively low
                # Strategy: linear interpolation of hull vertices
                from scipy.interpolate import interp1d
                baseline_func = interp1d(
                    hull_x, hull_y,
                    kind='linear',
                    bounds_error=False,
                    fill_value=(hull_y[0], hull_y[-1]),
                )
                baseline = baseline_func(x)

                # Take the minimum of spectrum and baseline at each point
                # to get the lower envelope
                baseline = np.minimum(baseline, spectrum)
                result[i] = spectrum - baseline

            except Exception:
                # Fallback: simple polynomial baseline
                coeffs = np.polyfit(x, spectrum, 2)
                baseline = np.polyval(coeffs, x)
                result[i] = spectrum - baseline

        return _maybe_squeeze(result, was_1d)

    def baseline_als(
        self,
        X: np.ndarray,
        lam: Optional[float] = None,
        p: Optional[float] = None,
        n_iter: Optional[int] = None,
    ) -> np.ndarray:
        """
        Asymmetric Least Squares (ALS) baseline correction.

        Also known as the Whittaker smoother with asymmetric weights.
        Excellent for FTIR baseline removal.

        Parameters
        ----------
        X : np.ndarray
            1-D or 2-D spectral data.
        lam : float, optional
            Smoothness parameter (10^2 to 10^9). Larger = smoother baseline.
        p : float, optional
            Asymmetry parameter (0.001 to 0.1). Smaller = baseline below data.
        n_iter : int, optional
            Number of reweighted iterations.

        Returns
        -------
        np.ndarray
            Baseline-corrected spectra.

        References
        ----------
        Eilers, P.H.C., Boelens, H.F.M. (2005).
        "Baseline Correction with Asymmetric Least Squares Smoothing."
        """
        lam = lam if lam is not None else self.als_lam
        p = p if p is not None else self.als_p
        n_iter = n_iter if n_iter is not None else self.als_niter

        X, was_1d = _ensure_2d(X)
        result = np.empty_like(X)

        for i in range(X.shape[0]):
            result[i] = X[i] - self._als_baseline(X[i], lam, p, n_iter)

        return _maybe_squeeze(result, was_1d)

    @staticmethod
    def _als_baseline(
        y: np.ndarray,
        lam: float,
        p: float,
        n_iter: int,
    ) -> np.ndarray:
        """Compute ALS baseline for a single spectrum."""
        L = len(y)
        # Second-difference matrix
        D = sparse.diags([1, -2, 1], [0, 1, 2], shape=(L - 2, L))
        D = D.tocsc()
        w = np.ones(L)

        for _ in range(n_iter):
            W = sparse.spdiags(w, 0, L, L)
            Z = W + lam * D.T.dot(D)
            z = spsolve(Z, w * y)
            w = p * (y > z) + (1 - p) * (y <= z)

        return z

    # =====================================================================
    # 2.  SMOOTHING
    # =====================================================================

    def savgol_smooth(
        self,
        X: np.ndarray,
        window_length: int = 11,
        polyorder: Optional[int] = None,
    ) -> np.ndarray:
        """
        Savitzky-Golay smoothing filter.

        Parameters
        ----------
        X : np.ndarray
            1-D or 2-D spectral data.
        window_length : int
            Length of the filter window (must be odd and > polyorder).
        polyorder : int, optional
            Polynomial order (default from config).

        Returns
        -------
        np.ndarray
            Smoothed spectra.
        """
        polyorder = polyorder if polyorder is not None else self.savgol_polyorder

        # Ensure window_length is valid
        X, was_1d = _ensure_2d(X)
        n_points = X.shape[1]

        if window_length >= n_points:
            window_length = n_points - 1 if n_points % 2 == 0 else n_points - 2
        if window_length < polyorder + 2:
            window_length = polyorder + 2
        if window_length % 2 == 0:
            window_length += 1

        result = np.empty_like(X)
        for i in range(X.shape[0]):
            result[i] = savgol_filter(
                X[i], window_length=window_length,
                polyorder=polyorder, deriv=0,
            )

        return _maybe_squeeze(result, was_1d)

    # =====================================================================
    # 3.  DERIVATIVES
    # =====================================================================

    def first_derivative(
        self,
        X: np.ndarray,
        window_length: Optional[int] = None,
        polyorder: Optional[int] = None,
    ) -> np.ndarray:
        """
        First derivative via Savitzky-Golay differentiation.

        Parameters
        ----------
        X : np.ndarray
            1-D or 2-D spectral data.
        window_length : int, optional
            Filter window length.
        polyorder : int, optional
            Polynomial order.

        Returns
        -------
        np.ndarray
            First-derivative spectra.
        """
        return self._derivative(X, order=1,
                                window_length=window_length,
                                polyorder=polyorder)

    def second_derivative(
        self,
        X: np.ndarray,
        window_length: Optional[int] = None,
        polyorder: Optional[int] = None,
    ) -> np.ndarray:
        """
        Second derivative via Savitzky-Golay differentiation.

        Parameters
        ----------
        X : np.ndarray
            1-D or 2-D spectral data.
        window_length : int, optional
            Filter window length.
        polyorder : int, optional
            Polynomial order.

        Returns
        -------
        np.ndarray
            Second-derivative spectra.
        """
        return self._derivative(X, order=2,
                                window_length=window_length,
                                polyorder=polyorder)

    def _derivative(
        self,
        X: np.ndarray,
        order: int = 1,
        window_length: Optional[int] = None,
        polyorder: Optional[int] = None,
    ) -> np.ndarray:
        """Compute nth derivative via Savitzky-Golay."""
        window_length = window_length or self.deriv_window
        polyorder = polyorder or self.deriv_polyorder

        # polyorder must be >= deriv order
        if polyorder < order:
            polyorder = order

        X, was_1d = _ensure_2d(X)
        n_points = X.shape[1]

        # Adjust window
        if window_length >= n_points:
            window_length = n_points - 1 if n_points % 2 == 0 else n_points - 2
        if window_length < polyorder + 2:
            window_length = polyorder + 2
        if window_length % 2 == 0:
            window_length += 1

        result = np.empty_like(X)
        for i in range(X.shape[0]):
            result[i] = savgol_filter(
                X[i],
                window_length=window_length,
                polyorder=polyorder,
                deriv=order,
            )

        return _maybe_squeeze(result, was_1d)

    # =====================================================================
    # 4.  NORMALIZATION
    # =====================================================================

    def snv(self, X: np.ndarray) -> np.ndarray:
        """
        Standard Normal Variate (SNV) normalization.

        Each spectrum is centred to zero mean and scaled to unit
        standard deviation::

            X_snv[i] = (X[i] - mean(X[i])) / std(X[i])

        Removes multiplicative scatter effects common in
        diffuse-reflectance and ATR-FTIR.

        Parameters
        ----------
        X : np.ndarray
            1-D or 2-D spectral data.

        Returns
        -------
        np.ndarray
            SNV-normalized spectra.
        """
        X, was_1d = _ensure_2d(X)
        result = np.empty_like(X)

        for i in range(X.shape[0]):
            spectrum = X[i]
            mean = np.mean(spectrum)
            std = np.std(spectrum)
            if std < 1e-12:
                result[i] = spectrum - mean
            else:
                result[i] = (spectrum - mean) / std

        return _maybe_squeeze(result, was_1d)

    def vector_normalize(self, X: np.ndarray) -> np.ndarray:
        """
        Vector (L2) normalization.

        Each spectrum is divided by its L2 norm::

            X_norm[i] = X[i] / ||X[i]||_2

        Parameters
        ----------
        X : np.ndarray
            1-D or 2-D spectral data.

        Returns
        -------
        np.ndarray
            L2-normalized spectra.
        """
        X, was_1d = _ensure_2d(X)
        result = np.empty_like(X)

        for i in range(X.shape[0]):
            norm = np.linalg.norm(X[i])
            if norm < 1e-12:
                result[i] = X[i]
            else:
                result[i] = X[i] / norm

        return _maybe_squeeze(result, was_1d)

    def minmax_normalize(
        self,
        X: np.ndarray,
        feature_range: Optional[Tuple[float, float]] = None,
    ) -> np.ndarray:
        """
        Min-Max normalization (per spectrum).

        Scales each spectrum independently to ``[min_val, max_val]``.

        Parameters
        ----------
        X : np.ndarray
            1-D or 2-D spectral data.
        feature_range : tuple of (float, float), optional
            Target range.  Default from config.

        Returns
        -------
        np.ndarray
            Min-max normalized spectra.
        """
        if feature_range is None:
            feature_range = self.minmax_range
        min_val, max_val = feature_range

        X, was_1d = _ensure_2d(X)
        result = np.empty_like(X)

        for i in range(X.shape[0]):
            s_min = np.min(X[i])
            s_max = np.max(X[i])
            denom = s_max - s_min

            if denom < 1e-12:
                result[i] = np.full_like(X[i], min_val)
            else:
                result[i] = (X[i] - s_min) / denom * (max_val - min_val) + min_val

        return _maybe_squeeze(result, was_1d)

    def area_normalize(self, X: np.ndarray) -> np.ndarray:
        """
        Area normalization.

        Divides each spectrum by the area under the curve
        (trapezoidal integration), so the integral equals 1.

        Parameters
        ----------
        X : np.ndarray
            1-D or 2-D spectral data.

        Returns
        -------
        np.ndarray
            Area-normalized spectra.
        """
        X, was_1d = _ensure_2d(X)
        result = np.empty_like(X)

        for i in range(X.shape[0]):
            area = np.trapz(np.abs(X[i]))
            if area < 1e-12:
                result[i] = X[i]
            else:
                result[i] = X[i] / area

        return _maybe_squeeze(result, was_1d)

    # =====================================================================
    # REGISTRY — maps string names to callables
    # =====================================================================

    def get_technique(self, name: str):
        """
        Get a preprocessing function by its string name.

        Parameters
        ----------
        name : str
            Technique name as used in pipeline specifications.

        Returns
        -------
        callable
            Function ``(X) -> X_processed``.

        Raises
        ------
        KeyError
            If technique name is unknown.
        """
        registry = self._build_registry()
        if name not in registry:
            available = ', '.join(sorted(registry.keys()))
            raise KeyError(
                f"Unknown technique: '{name}'. Available: {available}"
            )
        return registry[name]

    def _build_registry(self):
        """Build name → callable mapping."""
        return {
            # Baseline
            'none':                 lambda X: X,
            'baseline_polynomial':  self.baseline_polynomial,
            'baseline_rubberband':  self.baseline_rubberband,
            'baseline_als':         self.baseline_als,

            # Smoothing
            'savgol_5':   lambda X: self.savgol_smooth(X, window_length=5),
            'savgol_11':  lambda X: self.savgol_smooth(X, window_length=11),
            'savgol_21':  lambda X: self.savgol_smooth(X, window_length=21),

            # Derivatives
            'first_derivative':  self.first_derivative,
            'second_derivative': self.second_derivative,

            # Normalization
            'snv':        self.snv,
            'vector':     self.vector_normalize,
            'minmax':     self.minmax_normalize,
            'area':       self.area_normalize,
        }

    def list_techniques(self):
        """Return sorted list of available technique names."""
        return sorted(self._build_registry().keys())


# =============================================================================
# MAIN (for testing)
# =============================================================================

if __name__ == '__main__':
    print("Testing PreprocessingTechniques...")
    print("=" * 60)

    tech = PreprocessingTechniques()

    # Synthetic spectrum with baseline drift and noise
    np.random.seed(42)
    n = 1800
    x = np.linspace(0, 1, n)
    baseline = 0.5 + 0.3 * x + 0.2 * x**2
    peaks = (
        0.8 * np.exp(-((x - 0.3) ** 2) / 0.001) +
        0.5 * np.exp(-((x - 0.6) ** 2) / 0.002) +
        0.3 * np.exp(-((x - 0.8) ** 2) / 0.0005)
    )
    noise = np.random.randn(n) * 0.02
    spectrum = baseline + peaks + noise

    # Test 1-D
    print("\n--- 1-D Tests ---")
    for name in tech.list_techniques():
        func = tech.get_technique(name)
        result = func(spectrum)
        assert result.shape == spectrum.shape, f"{name}: shape mismatch"
        assert np.all(np.isfinite(result)), f"{name}: non-finite values"
        print(f"  ✓ {name:25s} | range=[{result.min():.4f}, {result.max():.4f}]")

    # Test 2-D
    print("\n--- 2-D Tests ---")
    X = np.vstack([spectrum + np.random.randn(n) * 0.01 for _ in range(20)])
    print(f"  Input shape: {X.shape}")

    for name in ['baseline_als', 'savgol_11', 'first_derivative', 'snv', 'minmax']:
        func = tech.get_technique(name)
        result = func(X)
        assert result.shape == X.shape, f"{name}: 2-D shape mismatch"
        assert np.all(np.isfinite(result)), f"{name}: 2-D non-finite"
        print(f"  ✓ {name:25s} | shape={result.shape}")

    # Verify SNV properties
    snv_result = tech.snv(X)
    means = np.mean(snv_result, axis=1)
    stds = np.std(snv_result, axis=1)
    assert np.allclose(means, 0, atol=1e-10), "SNV means not zero"
    assert np.allclose(stds, 1, atol=1e-10), "SNV stds not one"
    print(f"\n  ✓ SNV: mean≈0 ({means.mean():.2e}), std≈1 ({stds.mean():.6f})")

    # Verify vector norm
    vec_result = tech.vector_normalize(X)
    norms = np.linalg.norm(vec_result, axis=1)
    assert np.allclose(norms, 1, atol=1e-10), "Vector norms not 1"
    print(f"  ✓ Vector: ||x||₂≈1 ({norms.mean():.6f})")

    # Verify minmax
    mm_result = tech.minmax_normalize(X)
    assert np.allclose(mm_result.min(axis=1), 0, atol=1e-10), "MinMax min not 0"
    assert np.allclose(mm_result.max(axis=1), 1, atol=1e-10), "MinMax max not 1"
    print(f"  ✓ MinMax: min≈0, max≈1")

    print(f"\nAvailable techniques ({len(tech.list_techniques())}):")
    print(f"  {tech.list_techniques()}")

    print("\n✅ All PreprocessingTechniques tests passed!")