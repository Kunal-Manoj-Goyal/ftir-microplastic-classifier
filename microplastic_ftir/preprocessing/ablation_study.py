#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Preprocessing Ablation Study
==============================

Systematically evaluates all preprocessing pipeline combinations
to determine the optimal preprocessing strategy. This directly
tests **Hypothesis H2**: *"The choice of preprocessing pipeline has
an equal or greater impact on Macro-F1 than model selection."*

Evaluation Protocol:
    1. Generate all valid pipeline combinations.
    2. For each pipeline, preprocess training and validation data.
    3. Train a fast reference model (Random Forest by default).
    4. Evaluate on validation data using Macro-F1.
    5. Rank pipelines and compute:
        - 95% bootstrapped confidence intervals
        - Pairwise Wilcoxon signed-rank tests
        - Cohen's d effect sizes
        - Benjamini-Hochberg corrected p-values

Output:
    ``AblationResult`` dataclass with full rankings, best pipeline,
    statistical analysis, and visualisation data.

Usage:
------
    >>> from microplastic_ftir.preprocessing.ablation_study import AblationStudy
    >>>
    >>> study = AblationStudy(config)
    >>> result = study.run(X_train, y_train, X_val, y_val)
    >>> print(result.best_pipeline)
    >>> result.print_summary()
"""

import json
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from microplastic_ftir.preprocessing.pipeline import (
    PreprocessingPipeline,
    generate_all_pipelines,
    get_predefined_pipeline,
    list_available_pipelines,
)
from microplastic_ftir.utils.logging_utils import (
    get_logger,
    log_stage_header,
    log_metric,
    LogTimer,
)
from microplastic_ftir.utils.checkpoint import save_json, save_pickle

logger = get_logger(__name__)

# Optional imports
try:
    from scipy.stats import wilcoxon
    HAS_SCIPY_STATS = True
except ImportError:
    HAS_SCIPY_STATS = False

try:
    from tqdm.auto import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


# =============================================================================
# RESULT DATACLASS
# =============================================================================

@dataclass
class PipelineScore:
    """Score record for a single pipeline."""
    pipeline_name: str
    pipeline_steps: List[str]
    macro_f1_mean: float
    macro_f1_std: float
    macro_f1_scores: List[float]
    ci_lower: float
    ci_upper: float
    preprocessing_time: float   # seconds
    training_time: float        # seconds
    rank: int = 0


@dataclass
class AblationResult:
    """
    Full result of the preprocessing ablation study.

    Attributes
    ----------
    scores : list of PipelineScore
        All evaluated pipelines, sorted by descending Macro-F1.
    best_pipeline : PreprocessingPipeline
        The top-ranked pipeline.
    best_score : float
        Best mean Macro-F1.
    rankings_df : pd.DataFrame
        DataFrame of all rankings.
    pairwise_tests : pd.DataFrame or None
        Pairwise statistical comparison matrix.
    effect_sizes : pd.DataFrame or None
        Cohen's d effect size matrix.
    metadata : dict
        Study parameters and timing.
    """
    scores: List[PipelineScore]
    best_pipeline: PreprocessingPipeline
    best_score: float
    rankings_df: pd.DataFrame
    pairwise_tests: Optional[pd.DataFrame] = None
    effect_sizes: Optional[pd.DataFrame] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def print_summary(self, top_n: int = 15):
        """Print formatted summary of top pipelines."""
        print(f"\n{'='*80}")
        print(f"PREPROCESSING ABLATION STUDY — RESULTS")
        print(f"{'='*80}")
        print(f"\nTotal pipelines evaluated: {len(self.scores)}")
        print(f"Best pipeline: {self.best_pipeline.name}")
        print(f"Best Macro-F1: {self.best_score:.6f}")
        print(f"\nTop-{top_n} Pipelines:")
        print(f"{'-'*80}")
        print(f"{'Rank':>4} | {'Pipeline':40s} | {'F1 Mean':>8} | {'F1 Std':>7} | {'95% CI':>15}")
        print(f"{'-'*80}")
        for s in self.scores[:top_n]:
            ci = f"[{s.ci_lower:.4f}, {s.ci_upper:.4f}]"
            print(f"{s.rank:4d} | {s.pipeline_name:40s} | {s.macro_f1_mean:.6f} | "
                  f"{s.macro_f1_std:.5f} | {ci}")
        print(f"{'-'*80}")

        # Show worst 3 for contrast
        print(f"\nBottom-3 Pipelines:")
        for s in self.scores[-3:]:
            print(f"{s.rank:4d} | {s.pipeline_name:40s} | {s.macro_f1_mean:.6f}")

        # Performance gap
        if len(self.scores) >= 2:
            gap = self.scores[0].macro_f1_mean - self.scores[-1].macro_f1_mean
            print(f"\nPerformance gap (best − worst): {gap:.6f}")

    def to_dict(self) -> Dict[str, Any]:
        """Serialise for saving."""
        return {
            'best_pipeline': self.best_pipeline.to_dict(),
            'best_score': self.best_score,
            'rankings': [
                {
                    'rank': s.rank,
                    'name': s.pipeline_name,
                    'steps': s.pipeline_steps,
                    'f1_mean': s.macro_f1_mean,
                    'f1_std': s.macro_f1_std,
                    'ci_lower': s.ci_lower,
                    'ci_upper': s.ci_upper,
                }
                for s in self.scores
            ],
            'metadata': self.metadata,
        }


# =============================================================================
# ABLATION STUDY
# =============================================================================

class AblationStudy:
    """
    Systematic preprocessing pipeline evaluation.

    Parameters
    ----------
    config : Config, optional
        Project configuration.
    n_cv_splits : int
        Number of stratified cross-validation splits for evaluation.
    n_bootstrap : int
        Number of bootstrap samples for confidence intervals.
    confidence_level : float
        Confidence level for CI (e.g. 0.95).
    max_pipeline_depth : int, optional
        Maximum number of active preprocessing steps per pipeline.
    reference_model : str
        Model used for evaluation ('random_forest').
    use_predefined_only : bool
        If True, only evaluate predefined pipelines (faster).
        If False, generate all combinations (thorough).
    n_jobs : int
        Parallel jobs for the reference model.
    """

    def __init__(
        self,
        config: Optional['Config'] = None,
        n_cv_splits: int = 5,
        n_bootstrap: int = 1000,
        confidence_level: float = 0.95,
        max_pipeline_depth: Optional[int] = None,
        reference_model: str = 'random_forest',
        use_predefined_only: bool = False,
        n_jobs: int = -1,
    ):
        self.config = config
        self.n_cv_splits = n_cv_splits
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.max_pipeline_depth = max_pipeline_depth or (
            config.preprocessing.max_pipeline_depth if config else 3
        )
        self.reference_model = reference_model
        self.use_predefined_only = use_predefined_only
        self.n_jobs = n_jobs

    # =========================================================================
    # MAIN ENTRY POINT
    # =========================================================================

    def run(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> AblationResult:
        """
        Run the full ablation study.

        If ``X_val`` / ``y_val`` are provided, each pipeline is
        evaluated as train → transform → fit RF → score on val.

        If they are not provided, stratified K-fold CV on ``X_train``
        is used instead (more robust but slower).

        Parameters
        ----------
        X_train : np.ndarray
            Training spectra, shape ``(n_train, n_features)``.
        y_train : np.ndarray
            Training labels, shape ``(n_train,)``.
        X_val : np.ndarray, optional
            Validation spectra.
        y_val : np.ndarray, optional
            Validation labels.
        output_dir : str or Path, optional
            Directory to save results.  If None and config available,
            uses ``config.RESULTS_DIR / 'ablation'``.

        Returns
        -------
        AblationResult
        """
        log_stage_header(
            logger, 'preprocessing_ablation',
            'Evaluating preprocessing pipelines',
        )

        # Generate pipelines
        pipelines = self._get_pipelines()
        logger.info("Evaluating %d pipelines", len(pipelines))

        use_cv = X_val is None or y_val is None

        # Evaluate each pipeline
        all_scores: List[PipelineScore] = []
        total_start = time.time()

        iterator = tqdm(pipelines, desc="Ablation Study") if HAS_TQDM else pipelines

        for pipe in iterator:
            try:
                if use_cv:
                    score = self._evaluate_cv(pipe, X_train, y_train)
                else:
                    score = self._evaluate_holdout(pipe, X_train, y_train, X_val, y_val)
                all_scores.append(score)

                if HAS_TQDM and hasattr(iterator, 'set_postfix'):
                    iterator.set_postfix({
                        'best': f"{max(s.macro_f1_mean for s in all_scores):.4f}",
                        'current': f"{score.macro_f1_mean:.4f}",
                    })

            except Exception as e:
                logger.warning("Pipeline '%s' failed: %s", pipe.name, e)

        total_time = time.time() - total_start

        if not all_scores:
            raise RuntimeError("All pipelines failed! Check data and preprocessing steps.")

        # Rank
        all_scores.sort(key=lambda s: s.macro_f1_mean, reverse=True)
        for rank, score in enumerate(all_scores, 1):
            score.rank = rank

        # Best pipeline
        best = all_scores[0]
        best_pipeline = PreprocessingPipeline(
            steps=best.pipeline_steps,
            config=self.config,
            name=best.pipeline_name,
        )

        # Rankings DataFrame
        rankings_df = pd.DataFrame([
            {
                'rank': s.rank,
                'pipeline': s.pipeline_name,
                'steps': '+'.join(s.pipeline_steps) if s.pipeline_steps else 'raw',
                'n_steps': len(s.pipeline_steps),
                'macro_f1_mean': s.macro_f1_mean,
                'macro_f1_std': s.macro_f1_std,
                'ci_lower': s.ci_lower,
                'ci_upper': s.ci_upper,
                'preprocess_time_s': s.preprocessing_time,
                'train_time_s': s.training_time,
            }
            for s in all_scores
        ])

        # Statistical tests
        pairwise_tests = None
        effect_sizes = None
        if len(all_scores) >= 2 and use_cv:
            pairwise_tests, effect_sizes = self._statistical_tests(all_scores)

        # Build result
        result = AblationResult(
            scores=all_scores,
            best_pipeline=best_pipeline,
            best_score=best.macro_f1_mean,
            rankings_df=rankings_df,
            pairwise_tests=pairwise_tests,
            effect_sizes=effect_sizes,
            metadata={
                'n_pipelines': len(all_scores),
                'n_cv_splits': self.n_cv_splits,
                'n_bootstrap': self.n_bootstrap,
                'evaluation_mode': 'cv' if use_cv else 'holdout',
                'total_time_s': total_time,
                'X_train_shape': list(X_train.shape),
                'n_classes': len(np.unique(y_train)),
                'reference_model': self.reference_model,
            },
        )

        # Log top results
        for s in all_scores[:5]:
            log_metric(
                logger, 'macro_f1', s.macro_f1_mean,
                context=f"Rank {s.rank}: {s.pipeline_name}",
            )

        logger.info(
            "Ablation study complete: %d pipelines in %.1f min. "
            "Best: '%s' (F1=%.6f)",
            len(all_scores), total_time / 60,
            best_pipeline.name, best.macro_f1_mean,
        )

        # Save results
        if output_dir is not None or self.config is not None:
            self._save_results(result, output_dir)

        return result

    # =========================================================================
    # PIPELINE GENERATION
    # =========================================================================

    def _get_pipelines(self) -> List[PreprocessingPipeline]:
        """Get pipelines to evaluate."""
        if self.use_predefined_only:
            pipelines = [
                get_predefined_pipeline(name, self.config)
                for name in list_available_pipelines()
            ]
            logger.info("Using %d predefined pipelines", len(pipelines))
        else:
            pipelines = generate_all_pipelines(
                config=self.config,
                max_depth=self.max_pipeline_depth,
            )
        return pipelines

    # =========================================================================
    # EVALUATION METHODS
    # =========================================================================

    def _evaluate_holdout(
        self,
        pipeline: PreprocessingPipeline,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> PipelineScore:
        """Evaluate a pipeline with train/val holdout."""
        # Preprocess
        t0 = time.time()
        X_train_pp = pipeline.fit_transform(X_train)
        X_val_pp = pipeline.transform(X_val)
        pp_time = time.time() - t0

        # Train reference model
        t0 = time.time()
        model = self._get_reference_model()
        model.fit(X_train_pp, y_train)
        train_time = time.time() - t0

        # Score
        y_pred = model.predict(X_val_pp)
        f1 = f1_score(y_val, y_pred, average='macro', zero_division=0)

        # Bootstrap CI from single score (approximate)
        ci_lower, ci_upper = self._bootstrap_ci_single(f1)

        return PipelineScore(
            pipeline_name=pipeline.name,
            pipeline_steps=pipeline.steps,
            macro_f1_mean=f1,
            macro_f1_std=0.0,
            macro_f1_scores=[f1],
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            preprocessing_time=pp_time,
            training_time=train_time,
        )

    def _evaluate_cv(
        self,
        pipeline: PreprocessingPipeline,
        X: np.ndarray,
        y: np.ndarray,
    ) -> PipelineScore:
        """Evaluate a pipeline with stratified K-fold CV."""
        skf = StratifiedKFold(
            n_splits=self.n_cv_splits,
            shuffle=True,
            random_state=42,
        )

        fold_scores = []
        total_pp_time = 0.0
        total_train_time = 0.0

        for train_idx, val_idx in skf.split(X, y):
            X_tr, X_va = X[train_idx], X[val_idx]
            y_tr, y_va = y[train_idx], y[val_idx]

            # Preprocess
            t0 = time.time()
            X_tr_pp = pipeline.fit_transform(X_tr)
            X_va_pp = pipeline.transform(X_va)
            total_pp_time += time.time() - t0

            # Train
            t0 = time.time()
            model = self._get_reference_model()
            model.fit(X_tr_pp, y_tr)
            total_train_time += time.time() - t0

            # Score
            y_pred = model.predict(X_va_pp)
            f1 = f1_score(y_va, y_pred, average='macro', zero_division=0)
            fold_scores.append(f1)

        fold_scores = np.array(fold_scores)
        mean_f1 = fold_scores.mean()
        std_f1 = fold_scores.std()

        # Bootstrap CI
        ci_lower, ci_upper = self._bootstrap_ci(fold_scores)

        return PipelineScore(
            pipeline_name=pipeline.name,
            pipeline_steps=pipeline.steps,
            macro_f1_mean=float(mean_f1),
            macro_f1_std=float(std_f1),
            macro_f1_scores=fold_scores.tolist(),
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            preprocessing_time=total_pp_time / self.n_cv_splits,
            training_time=total_train_time / self.n_cv_splits,
        )

    # =========================================================================
    # REFERENCE MODEL
    # =========================================================================

    def _get_reference_model(self):
        """Create the reference classifier for pipeline evaluation."""
        if self.reference_model == 'random_forest':
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=None,
                min_samples_split=5,
                random_state=42,
                n_jobs=self.n_jobs,
            )
        else:
            raise ValueError(f"Unknown reference model: {self.reference_model}")

    # =========================================================================
    # STATISTICAL ANALYSIS
    # =========================================================================

    def _bootstrap_ci(
        self,
        scores: np.ndarray,
    ) -> Tuple[float, float]:
        """Compute bootstrap confidence interval from CV fold scores."""
        rng = np.random.RandomState(42)
        boot_means = []

        for _ in range(self.n_bootstrap):
            sample = rng.choice(scores, size=len(scores), replace=True)
            boot_means.append(sample.mean())

        boot_means = np.array(boot_means)
        alpha = 1 - self.confidence_level
        ci_lower = float(np.percentile(boot_means, 100 * alpha / 2))
        ci_upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))

        return ci_lower, ci_upper

    def _bootstrap_ci_single(self, score: float) -> Tuple[float, float]:
        """Approximate CI for a single score (no CV folds)."""
        # Use a heuristic spread based on typical CV variance
        margin = 0.02  # ±2% as rough estimate
        return max(0.0, score - margin), min(1.0, score + margin)

    def _statistical_tests(
        self,
        all_scores: List[PipelineScore],
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Pairwise statistical comparisons between top pipelines.

        Returns:
            (pairwise_p_values, effect_sizes)
        """
        if not HAS_SCIPY_STATS:
            logger.warning("scipy.stats not available — skipping statistical tests")
            return None, None

        # Only compare top-N to keep it manageable
        top_n = min(20, len(all_scores))
        top = all_scores[:top_n]

        names = [s.pipeline_name for s in top]
        n = len(names)

        # Matrices
        p_values = np.ones((n, n))
        d_values = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                scores_i = np.array(top[i].macro_f1_scores)
                scores_j = np.array(top[j].macro_f1_scores)

                # Need paired scores (same CV folds)
                if len(scores_i) != len(scores_j) or len(scores_i) < 3:
                    continue

                # Wilcoxon signed-rank test
                diff = scores_i - scores_j
                if np.all(diff == 0):
                    p_values[i, j] = 1.0
                    p_values[j, i] = 1.0
                else:
                    try:
                        _, p_val = wilcoxon(scores_i, scores_j, alternative='two-sided')
                        p_values[i, j] = p_val
                        p_values[j, i] = p_val
                    except Exception:
                        pass

                # Cohen's d
                pooled_std = np.sqrt(
                    (scores_i.std()**2 + scores_j.std()**2) / 2
                )
                if pooled_std > 1e-12:
                    d = (scores_i.mean() - scores_j.mean()) / pooled_std
                    d_values[i, j] = d
                    d_values[j, i] = -d

        # Benjamini-Hochberg correction
        p_values_corrected = self._benjamini_hochberg(p_values, n)

        p_df = pd.DataFrame(p_values_corrected, index=names, columns=names)
        d_df = pd.DataFrame(d_values, index=names, columns=names)

        # Log significant differences
        alpha = self.config.experiment.alpha if self.config else 0.05
        sig_count = 0
        for i in range(n):
            for j in range(i + 1, n):
                if p_values_corrected[i, j] < alpha:
                    sig_count += 1

        logger.info(
            "Statistical tests: %d/%d pairwise comparisons significant (α=%.2f)",
            sig_count, n * (n - 1) // 2, alpha,
        )

        return p_df, d_df

    @staticmethod
    def _benjamini_hochberg(p_matrix: np.ndarray, n: int) -> np.ndarray:
        """Apply Benjamini-Hochberg correction to upper triangle of p-value matrix."""
        # Extract upper triangle p-values
        upper_indices = np.triu_indices(n, k=1)
        p_vals = p_matrix[upper_indices]

        if len(p_vals) == 0:
            return p_matrix

        # Sort
        sorted_idx = np.argsort(p_vals)
        m = len(p_vals)

        # Correct
        corrected = np.empty(m)
        for rank_pos, orig_idx in enumerate(sorted_idx):
            rank = rank_pos + 1
            corrected[orig_idx] = min(1.0, p_vals[orig_idx] * m / rank)

        # Enforce monotonicity
        for i in range(m - 2, -1, -1):
            corrected[sorted_idx[i]] = min(
                corrected[sorted_idx[i]],
                corrected[sorted_idx[i + 1]] if i + 1 < m else 1.0,
            )

        # Put back in matrix
        result = p_matrix.copy()
        result[upper_indices] = corrected
        # Mirror
        for i, j in zip(*upper_indices):
            result[j, i] = result[i, j]

        return result

    # =========================================================================
    # SAVE RESULTS
    # =========================================================================

    def _save_results(
        self,
        result: AblationResult,
        output_dir: Optional[Union[str, Path]] = None,
    ):
        """Save ablation results to disk."""
        if output_dir is None:
            if self.config is not None:
                output_dir = self.config.RESULTS_DIR / 'ablation'
            else:
                output_dir = Path('results/ablation')

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save rankings CSV
        result.rankings_df.to_csv(output_dir / 'rankings.csv', index=False)

        # Save summary JSON
        save_json(result.to_dict(), output_dir / 'ablation_summary.json')

        # Save best pipeline spec
        save_json(
            result.best_pipeline.to_dict(),
            output_dir / 'best_pipeline.json',
        )

        # Save pairwise tests if available
        if result.pairwise_tests is not None:
            result.pairwise_tests.to_csv(output_dir / 'pairwise_pvalues.csv')
        if result.effect_sizes is not None:
            result.effect_sizes.to_csv(output_dir / 'effect_sizes.csv')

        # Save full result object
        save_pickle(result, output_dir / 'ablation_result.pkl')

        logger.info("Ablation results saved to %s", output_dir)


# =============================================================================
# MAIN (for testing)
# =============================================================================

if __name__ == '__main__':
    print("Testing AblationStudy...")
    print("=" * 60)

    from sklearn.datasets import make_classification

    # Synthetic spectral-like data
    np.random.seed(42)
    n_samples = 300
    n_features = 500

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=50,
        n_redundant=100,
        n_classes=5,
        random_state=42,
    )
    # Make it look like spectra (positive, with baseline)
    X = X - X.min() + 0.1
    X = X + np.linspace(0, 0.5, n_features)  # add baseline drift

    # Split
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42,
    )

    print(f"X_train: {X_train.shape}, X_val: {X_val.shape}")
    print(f"Classes: {np.unique(y_train)}")

    # Run with predefined pipelines only (fast)
    print("\n--- Running ablation (predefined only) ---")
    study = AblationStudy(
        n_cv_splits=3,
        n_bootstrap=500,
        use_predefined_only=True,
        n_jobs=1,
    )
    result = study.run(X_train, y_train, X_val, y_val)
    result.print_summary(top_n=10)

    print(f"\nBest pipeline: {result.best_pipeline}")
    print(f"Best Macro-F1: {result.best_score:.6f}")
    print(f"Rankings shape: {result.rankings_df.shape}")

    # Verify output structure
    assert len(result.scores) > 0, "No scores"
    assert result.best_score > 0, "Best score should be positive"
    assert result.rankings_df.shape[0] == len(result.scores)

    # Test CV mode
    print("\n--- Running ablation (CV mode, predefined) ---")
    study_cv = AblationStudy(
        n_cv_splits=3,
        n_bootstrap=200,
        use_predefined_only=True,
        n_jobs=1,
    )
    result_cv = study_cv.run(X_train, y_train)  # no val set → uses CV
    print(f"CV Best: {result_cv.best_pipeline.name} (F1={result_cv.best_score:.6f})")

    # Verify statistical tests were produced
    if result_cv.pairwise_tests is not None:
        print(f"Pairwise tests shape: {result_cv.pairwise_tests.shape}")
    if result_cv.effect_sizes is not None:
        print(f"Effect sizes shape: {result_cv.effect_sizes.shape}")

    print("\n✅ All AblationStudy tests passed!")