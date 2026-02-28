#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visualization Utilities for Microplastic FTIR Classification
=============================================================

Centralized plotting utilities providing:
    - Consistent plot styling across the project
    - Spectral plotting (single, overlay, comparison)
    - Classification result visualization (confusion matrices, ROC curves)
    - Preprocessing comparison plots
    - Attention / SHAP heatmaps
    - Multi-environment figure saving (Kaggle, Colab, local)

Design Decisions:
    - All functions accept an optional ``ax`` parameter so they can be
      composed into multi-panel figures by the caller.
    - ``save_figure()`` handles directory creation and multiple formats.
    - A unified color palette ensures visual consistency.
    - Plotly is used *only* when explicitly requested (not a hard dep).

Usage:
------
    >>> from microplastic_ftir.utils.visualization import (
    ...     set_plot_style, plot_spectrum, plot_confusion_matrix, save_figure,
    ... )
    >>>
    >>> set_plot_style()  # call once at startup
    >>> fig, ax = plt.subplots()
    >>> plot_spectrum(wavenumbers, absorbance, ax=ax, label='PE')
    >>> save_figure(fig, 'spectra/pe_example', config=config)

Author: Your Name
Date: 2024
"""

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

# Core plotting (always available in our env)
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

from microplastic_ftir.utils.logging_utils import get_logger

logger = get_logger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# ---- Colour Palette --------------------------------------------------------
# 12-class qualitative palette designed for colour-blind accessibility.
# First 7 colours correspond to common polymer classes.
POLYMER_COLORS: Dict[str, str] = {
    'PE':      '#1f77b4',   # blue
    'PP':      '#ff7f0e',   # orange
    'PS':      '#2ca02c',   # green
    'PET':     '#d62728',   # red
    'PVC':     '#9467bd',   # purple
    'PA':      '#8c564b',   # brown
    'PMMA':    '#e377c2',   # pink
    'ABS':     '#7f7f7f',   # grey
    'PC':      '#bcbd22',   # olive
    'PU':      '#17becf',   # teal
    'Nylon':   '#aec7e8',   # light blue
    'Other':   '#c7c7c7',   # light grey
}

# Sequential palette for heatmaps
HEATMAP_CMAP = 'RdYlBu_r'

# Diverging palette for SHAP / attention
DIVERGING_CMAP = 'coolwarm'

# ---- Figure Defaults -------------------------------------------------------
DEFAULT_FIG_WIDTH = 10
DEFAULT_FIG_HEIGHT = 6
DEFAULT_DPI = 150
SAVE_DPI = 300
FONT_SIZE_TITLE = 14
FONT_SIZE_LABEL = 12
FONT_SIZE_TICK = 10
FONT_SIZE_LEGEND = 10

# ---- FTIR Spectral Regions of Interest -------------------------------------
FTIR_REGIONS: Dict[str, Tuple[float, float]] = {
    'O-H / N-H stretch':     (3200, 3600),
    'C-H stretch':           (2800, 3000),
    'C=O stretch':           (1680, 1780),
    'C=C stretch':           (1580, 1680),
    'C-H bend':              (1350, 1480),
    'C-O stretch':           (1000, 1300),
    'Fingerprint':           (600,  1500),
}


# =============================================================================
# STYLE SETUP
# =============================================================================

def set_plot_style(
    style: str = 'seaborn-v0_8-whitegrid',
    context: str = 'paper',
    font_scale: float = 1.1,
    use_tex: bool = False,
):
    """
    Apply a consistent plot style for the entire project.

    Call this **once** at startup (e.g., in ``main.py``).

    Parameters
    ----------
    style : str
        Matplotlib style name.
    context : str
        Seaborn context ('paper', 'notebook', 'talk', 'poster').
    font_scale : float
        Scaling factor for font sizes.
    use_tex : bool
        Enable LaTeX rendering (requires TeX installation).
    """
    # Try the requested style, fall back gracefully
    try:
        plt.style.use(style)
    except OSError:
        try:
            plt.style.use('seaborn-whitegrid')
        except OSError:
            plt.style.use('ggplot')
            logger.debug("Fell back to 'ggplot' matplotlib style")

    if HAS_SEABORN:
        sns.set_context(context, font_scale=font_scale)

    # Global rcParams overrides
    plt.rcParams.update({
        'figure.figsize': (DEFAULT_FIG_WIDTH, DEFAULT_FIG_HEIGHT),
        'figure.dpi': DEFAULT_DPI,
        'savefig.dpi': SAVE_DPI,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        'font.size': FONT_SIZE_LABEL,
        'axes.titlesize': FONT_SIZE_TITLE,
        'axes.labelsize': FONT_SIZE_LABEL,
        'xtick.labelsize': FONT_SIZE_TICK,
        'ytick.labelsize': FONT_SIZE_TICK,
        'legend.fontsize': FONT_SIZE_LEGEND,
        'text.usetex': use_tex,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })

    logger.debug("Plot style configured")


def get_color_palette(n: Optional[int] = None) -> List[str]:
    """
    Return the project colour palette.

    Parameters
    ----------
    n : int, optional
        Number of colours needed. If > len(POLYMER_COLORS), cycles or
        falls back to a matplotlib colourmap.

    Returns
    -------
    list of str
        Hex colour strings.
    """
    base = list(POLYMER_COLORS.values())

    if n is None or n <= len(base):
        return base[:n] if n else base

    # Extend using a colourmap
    cmap = plt.cm.get_cmap('tab20', n)
    return [matplotlib.colors.rgb2hex(cmap(i)) for i in range(n)]


def get_polymer_color(polymer: str) -> str:
    """
    Get the colour for a specific polymer, with a default fallback.

    Parameters
    ----------
    polymer : str
        Polymer name.

    Returns
    -------
    str
        Hex colour string.
    """
    return POLYMER_COLORS.get(polymer, POLYMER_COLORS.get('Other', '#c7c7c7'))


# =============================================================================
# FIGURE SAVING
# =============================================================================

def save_figure(
    fig: plt.Figure,
    name: str,
    config: Optional[Any] = None,
    output_dir: Optional[Union[str, Path]] = None,
    formats: Sequence[str] = ('png', 'pdf'),
    dpi: int = SAVE_DPI,
    close: bool = True,
) -> List[Path]:
    """
    Save a matplotlib figure to disk in one or more formats.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save.
    name : str
        Base filename (without extension). May include subdirectories
        (e.g., ``'spectra/pe_overlay'``).
    config : Config, optional
        Project Config object. If provided, uses ``config.FIGURES_DIR``.
    output_dir : str or Path, optional
        Override output directory. Takes precedence over ``config``.
    formats : sequence of str
        File formats to save ('png', 'pdf', 'svg', 'eps').
    dpi : int
        Resolution.
    close : bool
        Close the figure after saving to free memory.

    Returns
    -------
    list of Path
        Paths to saved files.

    Examples
    --------
    >>> fig, ax = plt.subplots()
    >>> ax.plot([1, 2, 3])
    >>> paths = save_figure(fig, 'test_plot', config=config)
    """
    # Determine output directory
    if output_dir is not None:
        base_dir = Path(output_dir)
    elif config is not None:
        base_dir = config.FIGURES_DIR
    else:
        base_dir = Path('figures')

    saved_paths = []

    for fmt in formats:
        filepath = base_dir / f"{name}.{fmt}"
        filepath.parent.mkdir(parents=True, exist_ok=True)

        try:
            fig.savefig(filepath, dpi=dpi, format=fmt, bbox_inches='tight')
            saved_paths.append(filepath)
            logger.debug("Figure saved: %s", filepath)
        except Exception as e:
            logger.warning("Failed to save figure as %s: %s", fmt, e)

    if close:
        plt.close(fig)

    return saved_paths


# =============================================================================
# SPECTRAL PLOTTING
# =============================================================================

def plot_spectrum(
    wavenumbers: np.ndarray,
    intensities: np.ndarray,
    ax: Optional[plt.Axes] = None,
    label: Optional[str] = None,
    color: Optional[str] = None,
    title: Optional[str] = None,
    xlabel: str = 'Wavenumber (cm⁻¹)',
    ylabel: str = 'Absorbance',
    invert_x: bool = True,
    alpha: float = 1.0,
    linewidth: float = 1.0,
    show_regions: bool = False,
    **kwargs,
) -> plt.Axes:
    """
    Plot a single FTIR spectrum.

    Parameters
    ----------
    wavenumbers : np.ndarray
        Wavenumber values (cm⁻¹).
    intensities : np.ndarray
        Intensity values (absorbance or transmittance).
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. Created if None.
    label : str, optional
        Legend label.
    color : str, optional
        Line colour. Auto-selected from polymer palette if label matches.
    title : str, optional
        Axes title.
    xlabel, ylabel : str
        Axis labels.
    invert_x : bool
        Invert x-axis (conventional for FTIR: high → low wavenumber).
    alpha : float
        Line transparency.
    linewidth : float
        Line width.
    show_regions : bool
        Shade key FTIR functional-group regions.
    **kwargs
        Extra keyword arguments passed to ``ax.plot()``.

    Returns
    -------
    matplotlib.axes.Axes
        The axes object (for chaining or further customisation).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(DEFAULT_FIG_WIDTH, DEFAULT_FIG_HEIGHT))

    # Auto-select colour
    if color is None and label is not None:
        color = get_polymer_color(label)

    ax.plot(
        wavenumbers, intensities,
        label=label, color=color,
        alpha=alpha, linewidth=linewidth,
        **kwargs,
    )

    if title:
        ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if invert_x:
        ax.invert_xaxis()

    if show_regions:
        _shade_ftir_regions(ax)

    if label:
        ax.legend(loc='best', framealpha=0.8)

    return ax


def plot_spectra_overlay(
    wavenumbers: np.ndarray,
    spectra_dict: Dict[str, np.ndarray],
    ax: Optional[plt.Axes] = None,
    title: str = 'FTIR Spectra Overlay',
    xlabel: str = 'Wavenumber (cm⁻¹)',
    ylabel: str = 'Absorbance',
    invert_x: bool = True,
    alpha: float = 0.8,
    linewidth: float = 1.0,
    show_regions: bool = False,
    **kwargs,
) -> plt.Axes:
    """
    Overlay multiple spectra on a single axes.

    Parameters
    ----------
    wavenumbers : np.ndarray
        Shared wavenumber grid.
    spectra_dict : dict
        Mapping of {label: intensity_array}.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. Created if None.
    title : str
        Axes title.
    show_regions : bool
        Shade FTIR functional-group regions.
    **kwargs
        Extra kwargs forwarded to each ``ax.plot()`` call.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(DEFAULT_FIG_WIDTH, DEFAULT_FIG_HEIGHT))

    colors = get_color_palette(len(spectra_dict))

    for (label, intensities), color in zip(spectra_dict.items(), colors):
        polymer_color = get_polymer_color(label)
        if polymer_color == POLYMER_COLORS.get('Other'):
            polymer_color = color
        ax.plot(
            wavenumbers, intensities,
            label=label, color=polymer_color,
            alpha=alpha, linewidth=linewidth,
            **kwargs,
        )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if invert_x:
        ax.invert_xaxis()

    if show_regions:
        _shade_ftir_regions(ax)

    ax.legend(loc='best', framealpha=0.8, ncol=max(1, len(spectra_dict) // 6))

    return ax


def plot_spectra_grid(
    wavenumbers: np.ndarray,
    spectra_dict: Dict[str, np.ndarray],
    ncols: int = 3,
    figsize_per_subplot: Tuple[float, float] = (5, 3),
    show_regions: bool = False,
    suptitle: str = 'FTIR Spectra by Polymer',
    **kwargs,
) -> plt.Figure:
    """
    Plot spectra in a grid of subplots (one per polymer class).

    Parameters
    ----------
    wavenumbers : np.ndarray
        Shared wavenumber grid.
    spectra_dict : dict
        Mapping of {label: intensity_array}.
    ncols : int
        Number of columns in the grid.
    figsize_per_subplot : tuple
        Width, height per subplot.
    show_regions : bool
        Shade FTIR regions.
    suptitle : str
        Overall figure title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    n = len(spectra_dict)
    nrows = max(1, (n + ncols - 1) // ncols)
    figsize = (figsize_per_subplot[0] * ncols, figsize_per_subplot[1] * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    for idx, (label, intensities) in enumerate(spectra_dict.items()):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        plot_spectrum(
            wavenumbers, intensities,
            ax=ax, label=label, title=label,
            show_regions=show_regions,
            **kwargs,
        )

    # Hide empty subplots
    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.suptitle(suptitle, fontsize=FONT_SIZE_TITLE + 2, y=1.02)
    fig.tight_layout()

    return fig


def _shade_ftir_regions(ax: plt.Axes, alpha: float = 0.08):
    """Shade key FTIR functional-group regions on an axes."""
    region_colors = [
        '#ff0000', '#0000ff', '#00cc00', '#ff9900',
        '#9900cc', '#009999', '#666666',
    ]
    for (name, (lo, hi)), color in zip(FTIR_REGIONS.items(), region_colors):
        ax.axvspan(lo, hi, alpha=alpha, color=color, label=None)
        mid = (lo + hi) / 2
        # Place label at top of axes
        ax.text(
            mid, ax.get_ylim()[1] * 0.95,
            name, fontsize=6, ha='center', va='top',
            rotation=90, alpha=0.5,
        )


# =============================================================================
# CLASSIFICATION RESULT PLOTS
# =============================================================================

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    ax: Optional[plt.Axes] = None,
    title: str = 'Confusion Matrix',
    normalize: bool = True,
    cmap: str = 'Blues',
    fmt: str = '.1%',
    figsize: Tuple[float, float] = (10, 8),
    annotate: bool = True,
    colorbar: bool = True,
) -> plt.Axes:
    """
    Plot a confusion matrix heatmap.

    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix (n_classes × n_classes).
    class_names : list of str
        Class labels.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    title : str
        Title.
    normalize : bool
        Normalize rows to show proportions.
    cmap : str
        Colourmap.
    fmt : str
        Number format for annotations.
    figsize : tuple
        Figure size (used only if ax is None).
    annotate : bool
        Show values in cells.
    colorbar : bool
        Show colour bar.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        # Avoid division by zero
        row_sums = np.where(row_sums == 0, 1, row_sums)
        cm_display = cm.astype(float) / row_sums
    else:
        cm_display = cm.astype(float)
        fmt = 'd'

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    if HAS_SEABORN:
        sns.heatmap(
            cm_display,
            annot=annotate,
            fmt=fmt,
            cmap=cmap,
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
            cbar=colorbar,
            linewidths=0.5,
            linecolor='white',
            square=True,
        )
    else:
        im = ax.imshow(cm_display, interpolation='nearest', cmap=cmap)
        if colorbar:
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.set_yticklabels(class_names)

        if annotate:
            for i in range(len(class_names)):
                for j in range(len(class_names)):
                    val = cm_display[i, j]
                    text_color = 'white' if val > cm_display.max() / 2 else 'black'
                    ax.text(
                        j, i, format(val, fmt),
                        ha='center', va='center',
                        color=text_color, fontsize=8,
                    )

    ax.set_title(title)
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')

    return ax


def plot_roc_curves(
    fpr_dict: Dict[str, np.ndarray],
    tpr_dict: Dict[str, np.ndarray],
    auc_dict: Dict[str, float],
    ax: Optional[plt.Axes] = None,
    title: str = 'ROC Curves (One-vs-Rest)',
    figsize: Tuple[float, float] = (8, 8),
) -> plt.Axes:
    """
    Plot multi-class ROC curves.

    Parameters
    ----------
    fpr_dict : dict
        {class_name: fpr_array}.
    tpr_dict : dict
        {class_name: tpr_array}.
    auc_dict : dict
        {class_name: auc_score}.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    title : str
        Title.
    figsize : tuple
        Figure size.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    colors = get_color_palette(len(fpr_dict))

    for (cls_name, fpr), color in zip(fpr_dict.items(), colors):
        tpr = tpr_dict[cls_name]
        auc_val = auc_dict.get(cls_name, 0)
        polymer_color = get_polymer_color(cls_name)
        if polymer_color == POLYMER_COLORS.get('Other'):
            polymer_color = color
        ax.plot(
            fpr, tpr,
            label=f'{cls_name} (AUC={auc_val:.3f})',
            color=polymer_color,
            linewidth=1.5,
        )

    # Diagonal
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1)

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc='lower right', fontsize=8, framealpha=0.8)

    return ax


def plot_metric_comparison(
    results: Dict[str, Dict[str, float]],
    metric: str = 'macro_f1',
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    sort: bool = True,
    horizontal: bool = True,
    figsize: Tuple[float, float] = (10, 6),
    show_values: bool = True,
    ci: Optional[Dict[str, Tuple[float, float]]] = None,
) -> plt.Axes:
    """
    Bar chart comparing a metric across models.

    Parameters
    ----------
    results : dict
        {model_name: {metric_name: value, ...}}.
    metric : str
        Metric to compare.
    ax : matplotlib.axes.Axes, optional
        Axes.
    title : str, optional
        Title. Defaults to the metric name.
    sort : bool
        Sort bars by value.
    horizontal : bool
        Horizontal bars.
    figsize : tuple
        Figure size.
    show_values : bool
        Annotate bars with values.
    ci : dict, optional
        {model_name: (lower, upper)} for confidence intervals.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    # Extract values
    models = []
    values = []
    for model_name, metrics in results.items():
        if metric in metrics:
            models.append(model_name)
            values.append(metrics[metric])

    if sort:
        pairs = sorted(zip(values, models), reverse=True)
        values, models = zip(*pairs) if pairs else ([], [])
        values = list(values)
        models = list(models)

    colors = get_color_palette(len(models))
    positions = range(len(models))

    if horizontal:
        bars = ax.barh(positions, values, color=colors, alpha=0.85, edgecolor='white')
        ax.set_yticks(positions)
        ax.set_yticklabels(models)
        ax.set_xlabel(metric)
        ax.invert_yaxis()

        # Confidence intervals
        if ci:
            for i, model_name in enumerate(models):
                if model_name in ci:
                    lo, hi = ci[model_name]
                    ax.errorbar(
                        values[i], i,
                        xerr=[[values[i] - lo], [hi - values[i]]],
                        fmt='none', color='black', capsize=3,
                    )

        if show_values:
            for bar, val in zip(bars, values):
                ax.text(
                    bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                    f'{val:.4f}', va='center', fontsize=9,
                )
    else:
        bars = ax.bar(positions, values, color=colors, alpha=0.85, edgecolor='white')
        ax.set_xticks(positions)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_ylabel(metric)

        if show_values:
            for bar, val in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=9,
                )

    ax.set_title(title or metric)

    return ax


# =============================================================================
# PREPROCESSING COMPARISON
# =============================================================================

def plot_preprocessing_comparison(
    wavenumbers: np.ndarray,
    original: np.ndarray,
    processed_dict: Dict[str, np.ndarray],
    title: str = 'Preprocessing Comparison',
    figsize: Tuple[float, float] = (14, 8),
) -> plt.Figure:
    """
    Compare original spectrum against multiple preprocessing pipelines.

    Parameters
    ----------
    wavenumbers : np.ndarray
        Wavenumber grid.
    original : np.ndarray
        Original (raw) spectrum.
    processed_dict : dict
        {pipeline_name: processed_spectrum}.
    title : str
        Figure title.
    figsize : tuple
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
    """
    n = len(processed_dict) + 1  # +1 for original
    ncols = min(3, n)
    nrows = max(1, (n + ncols - 1) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    # Plot original
    ax0 = axes[0][0]
    ax0.plot(wavenumbers, original, color='black', linewidth=1)
    ax0.set_title('Original', fontsize=11)
    ax0.invert_xaxis()

    for idx, (name, spectrum) in enumerate(processed_dict.items(), start=1):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        ax.plot(wavenumbers, spectrum, color='#1f77b4', linewidth=1)
        ax.set_title(name, fontsize=11)
        ax.invert_xaxis()

    # Hide empty
    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.suptitle(title, fontsize=FONT_SIZE_TITLE + 2, y=1.02)
    fig.tight_layout()

    return fig


# =============================================================================
# ATTENTION / SHAP VISUALIZATION
# =============================================================================

def plot_spectral_importance(
    wavenumbers: np.ndarray,
    importance: np.ndarray,
    ax: Optional[plt.Axes] = None,
    title: str = 'Spectral Feature Importance',
    xlabel: str = 'Wavenumber (cm⁻¹)',
    ylabel: str = 'Importance',
    show_regions: bool = True,
    fill: bool = True,
    color: str = '#1f77b4',
    invert_x: bool = True,
    figsize: Tuple[float, float] = (12, 5),
) -> plt.Axes:
    """
    Plot feature importance (SHAP values, attention weights, etc.)
    aligned with wavenumber positions.

    Parameters
    ----------
    wavenumbers : np.ndarray
        Wavenumber grid.
    importance : np.ndarray
        Importance values (same length as wavenumbers).
    ax : matplotlib.axes.Axes, optional
        Axes.
    title : str
        Title.
    show_regions : bool
        Shade FTIR functional-group regions.
    fill : bool
        Fill under the curve.
    color : str
        Line / fill colour.
    invert_x : bool
        Invert x-axis.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    ax.plot(wavenumbers, importance, color=color, linewidth=1.2)

    if fill:
        ax.fill_between(wavenumbers, 0, importance, alpha=0.3, color=color)

    if show_regions:
        _shade_ftir_regions(ax, alpha=0.06)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if invert_x:
        ax.invert_xaxis()

    ax.axhline(y=0, color='grey', linewidth=0.5, linestyle='--')

    return ax


# =============================================================================
# TRAINING HISTORY
# =============================================================================

def plot_training_history(
    history: Dict[str, List[float]],
    ax: Optional[plt.Axes] = None,
    title: str = 'Training History',
    figsize: Tuple[float, float] = (10, 5),
) -> plt.Axes:
    """
    Plot training and validation loss / metric curves.

    Parameters
    ----------
    history : dict
        {metric_name: [epoch_1_value, epoch_2_value, ...]}.
        Common keys: 'train_loss', 'val_loss', 'train_f1', 'val_f1'.
    ax : matplotlib.axes.Axes, optional
        Axes.
    title : str
        Title.
    figsize : tuple
        Figure size.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    # Separate loss and metric curves
    loss_keys = [k for k in history if 'loss' in k.lower()]
    metric_keys = [k for k in history if 'loss' not in k.lower()]

    # If both loss and metrics exist, use twin axes
    if loss_keys and metric_keys:
        ax2 = ax.twinx()
    else:
        ax2 = None

    colors_loss = ['#d62728', '#ff7f0e']
    colors_metric = ['#1f77b4', '#2ca02c']

    for i, key in enumerate(loss_keys):
        color = colors_loss[i % len(colors_loss)]
        style = '-' if 'train' in key else '--'
        ax.plot(history[key], label=key, color=color, linestyle=style, linewidth=1.5)

    target_ax = ax2 if ax2 is not None else ax
    for i, key in enumerate(metric_keys):
        color = colors_metric[i % len(colors_metric)]
        style = '-' if 'train' in key else '--'
        target_ax.plot(history[key], label=key, color=color, linestyle=style, linewidth=1.5)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(title)

    if ax2 is not None:
        ax2.set_ylabel('Metric')

    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    if ax2 is not None:
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='best', framealpha=0.8)
    else:
        ax.legend(loc='best', framealpha=0.8)

    return ax


# =============================================================================
# DOMAIN SHIFT / LODO VISUALIZATION
# =============================================================================

def plot_lodo_results(
    results: Dict[str, Dict[str, float]],
    metric: str = 'macro_f1',
    ax: Optional[plt.Axes] = None,
    title: str = 'Leave-One-Dataset-Out Results',
    figsize: Tuple[float, float] = (10, 6),
    show_mean: bool = True,
) -> plt.Axes:
    """
    Visualise per-held-out-dataset performance for LODO evaluation.

    Parameters
    ----------
    results : dict
        {held_out_dataset: {metric_name: value, ...}}.
    metric : str
        Metric to display.
    ax : matplotlib.axes.Axes, optional
        Axes.
    title : str
        Title.
    show_mean : bool
        Draw a horizontal line at the mean.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    datasets = list(results.keys())
    values = [results[d].get(metric, 0) for d in datasets]

    colors = get_color_palette(len(datasets))
    positions = range(len(datasets))

    bars = ax.bar(positions, values, color=colors, alpha=0.85, edgecolor='white')
    ax.set_xticks(positions)
    ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.set_ylabel(metric)
    ax.set_title(title)

    # Annotate
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f'{val:.3f}', ha='center', va='bottom', fontsize=9,
        )

    if show_mean and values:
        mean_val = np.mean(values)
        ax.axhline(y=mean_val, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.text(
            len(datasets) - 0.5, mean_val + 0.01,
            f'Mean={mean_val:.3f}', color='red', fontsize=10, ha='right',
        )

    return ax


def plot_recovery_curve(
    ratios: List[float],
    scores: Dict[str, List[float]],
    ax: Optional[plt.Axes] = None,
    title: str = 'Few-Shot Recovery Curve',
    xlabel: str = 'Fraction of Weathered Data',
    ylabel: str = 'Macro-F1',
    figsize: Tuple[float, float] = (8, 6),
    show_baseline: Optional[float] = None,
) -> plt.Axes:
    """
    Plot the few-shot fine-tuning recovery curve.

    Parameters
    ----------
    ratios : list of float
        Fine-tuning data fractions (e.g., [0.0, 0.05, 0.10, 0.20]).
    scores : dict
        {model_name: [score_at_ratio_0, score_at_ratio_1, ...]}.
    ax : matplotlib.axes.Axes, optional
        Axes.
    title : str
        Title.
    show_baseline : float, optional
        Draw a horizontal line for the "pristine-only" baseline.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    colors = get_color_palette(len(scores))

    for (model_name, model_scores), color in zip(scores.items(), colors):
        ax.plot(
            ratios, model_scores,
            marker='o', label=model_name, color=color,
            linewidth=2, markersize=8,
        )

    if show_baseline is not None:
        ax.axhline(
            y=show_baseline, color='grey', linestyle='--',
            linewidth=1.5, alpha=0.7, label=f'Pristine baseline ({show_baseline:.3f})',
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(ratios)
    ax.set_xticklabels([f'{r:.0%}' for r in ratios])
    ax.legend(loc='best', framealpha=0.8)

    return ax


# =============================================================================
# MAIN (for testing)
# =============================================================================

if __name__ == '__main__':
    import tempfile

    print("Testing visualization utilities...")
    print("=" * 60)

    set_plot_style()

    with tempfile.TemporaryDirectory() as tmpdir:
        # Test single spectrum plot
        wn = np.linspace(400, 4000, 1800)
        ab = np.random.randn(1800) * 0.1 + 0.5
        ab[800:900] += 0.3  # Fake peak

        fig, ax = plt.subplots()
        plot_spectrum(wn, ab, ax=ax, label='PE', title='Test Spectrum', show_regions=True)
        paths = save_figure(fig, 'test_spectrum', output_dir=tmpdir, formats=['png'])
        assert paths[0].exists(), "Spectrum figure not saved"
        print(f"✓ Spectrum plot saved: {paths[0]}")

        # Test overlay
        spectra = {
            'PE': ab,
            'PP': ab + np.random.randn(1800) * 0.05,
            'PS': ab - 0.1 + np.random.randn(1800) * 0.05,
        }
        fig2, ax2 = plt.subplots()
        plot_spectra_overlay(wn, spectra, ax=ax2)
        paths2 = save_figure(fig2, 'test_overlay', output_dir=tmpdir)
        print(f"✓ Overlay plot saved: {paths2[0]}")

        # Test confusion matrix
        cm = np.array([[45, 3, 2], [1, 50, 4], [3, 2, 40]])
        fig3, ax3 = plt.subplots()
        plot_confusion_matrix(cm, ['PE', 'PP', 'PS'], ax=ax3)
        paths3 = save_figure(fig3, 'test_cm', output_dir=tmpdir)
        print(f"✓ Confusion matrix saved: {paths3[0]}")

        # Test metric comparison
        results = {
            'Random Forest': {'macro_f1': 0.94, 'accuracy': 0.95},
            'SVM': {'macro_f1': 0.92, 'accuracy': 0.93},
            'XGBoost': {'macro_f1': 0.93, 'accuracy': 0.94},
            'SE-CNN': {'macro_f1': 0.96, 'accuracy': 0.97},
        }
        fig4, ax4 = plt.subplots()
        plot_metric_comparison(results, metric='macro_f1', ax=ax4)
        paths4 = save_figure(fig4, 'test_comparison', output_dir=tmpdir)
        print(f"✓ Metric comparison saved: {paths4[0]}")

        # Test spectral importance
        importance = np.abs(np.random.randn(1800)) * 0.01
        importance[800:900] = 0.1  # Spike at "important" region
        fig5, ax5 = plt.subplots()
        plot_spectral_importance(wn, importance, ax=ax5, show_regions=True)
        paths5 = save_figure(fig5, 'test_importance', output_dir=tmpdir)
        print(f"✓ Spectral importance saved: {paths5[0]}")

        # Test training history
        history = {
            'train_loss': [0.5, 0.3, 0.2, 0.15, 0.1],
            'val_loss': [0.55, 0.35, 0.25, 0.22, 0.20],
            'train_f1': [0.7, 0.85, 0.90, 0.93, 0.95],
            'val_f1': [0.65, 0.80, 0.87, 0.89, 0.90],
        }
        fig6, ax6 = plt.subplots()
        plot_training_history(history, ax=ax6)
        paths6 = save_figure(fig6, 'test_history', output_dir=tmpdir)
        print(f"✓ Training history saved: {paths6[0]}")

        # Test recovery curve
        fig7, ax7 = plt.subplots()
        plot_recovery_curve(
            ratios=[0.0, 0.05, 0.10, 0.20],
            scores={
                'SE-CNN': [0.60, 0.72, 0.81, 0.89],
                'SVM': [0.55, 0.65, 0.75, 0.83],
            },
            ax=ax7,
            show_baseline=0.55,
        )
        paths7 = save_figure(fig7, 'test_recovery', output_dir=tmpdir)
        print(f"✓ Recovery curve saved: {paths7[0]}")

        # Test LODO results
        lodo = {
            'SLOPP': {'macro_f1': 0.88},
            'FLOPP': {'macro_f1': 0.82},
            'Data_Gov': {'macro_f1': 0.75},
            'Zenodo': {'macro_f1': 0.79},
            'OpenSpecy': {'macro_f1': 0.71},
        }
        fig8, ax8 = plt.subplots()
        plot_lodo_results(lodo, ax=ax8)
        paths8 = save_figure(fig8, 'test_lodo', output_dir=tmpdir)
        print(f"✓ LODO results saved: {paths8[0]}")

        # Test color palette
        palette = get_color_palette(20)
        assert len(palette) == 20, "Palette extension failed"
        print(f"✓ Color palette (20 colors): {palette[:3]}...")

    print("\n✅ All visualization utility tests passed!")