#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Logging Utilities for Microplastic FTIR Classification
========================================================

Centralized logging configuration providing:
    - Consistent formatting across all modules
    - File + console dual output
    - Color support (via ``rich`` or ``colorlog`` when available)
    - Helper functions for structured log messages
    - Stage headers, metric logging, DataFrame summaries

Design Principles:
    - Every module obtains its logger via ``get_logger(__name__)``
    - ``setup_logging()`` is called once at startup (by main.py or SessionManager)
    - Falls back gracefully if rich/colorlog are missing

Usage:
------
    >>> from microplastic_ftir.utils.logging_utils import get_logger, setup_logging
    >>>
    >>> # Call once at startup
    >>> setup_logging(level='INFO', log_dir='/outputs/logs')
    >>>
    >>> # In each module
    >>> logger = get_logger(__name__)
    >>> logger.info("Loading dataset: %s", dataset_name)

Author: Your Name
Date: 2024
"""

import os
import sys
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

# Optional imports
try:
    from rich.logging import RichHandler
    from rich.console import Console
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

try:
    import colorlog
    HAS_COLORLOG = True
except ImportError:
    HAS_COLORLOG = False


# =============================================================================
# CONSTANTS
# =============================================================================

DEFAULT_FORMAT = '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
DEFAULT_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
RICH_FORMAT = '%(message)s'  # Rich handler does its own formatting

# Map string level names to logging constants
LEVEL_MAP = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL,
}

# Track whether setup has been called
_logging_initialized = False


# =============================================================================
# CORE SETUP
# =============================================================================

def setup_logging(
    level: Union[str, int] = 'INFO',
    log_dir: Optional[Union[str, Path]] = None,
    log_filename: str = 'experiment.log',
    log_to_console: bool = True,
    log_to_file: bool = True,
    use_rich: bool = True,
    use_colors: bool = True,
    fmt: Optional[str] = None,
    date_fmt: Optional[str] = None,
    capture_warnings: bool = True,
) -> logging.Logger:
    """
    Configure the root logger for the entire project.

    Should be called **once** at application startup (e.g., in ``main.py``
    or when ``SessionManager`` is created). Subsequent calls are no-ops
    unless ``force=True`` is added to the call site (not exposed to keep
    the API simple — just reset handlers manually if needed).

    Parameters
    ----------
    level : str or int
        Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL').
    log_dir : str or Path, optional
        Directory for the log file. Created if it does not exist.
    log_filename : str
        Name of the log file.
    log_to_console : bool
        Attach a console (stderr) handler.
    log_to_file : bool
        Attach a file handler.
    use_rich : bool
        Use ``rich.logging.RichHandler`` for console output if available.
    use_colors : bool
        Use colored output (via colorlog) as a fallback if rich is absent.
    fmt : str, optional
        Override log format string.
    date_fmt : str, optional
        Override date format string.
    capture_warnings : bool
        Route Python ``warnings`` through the logging system.

    Returns
    -------
    logging.Logger
        The root logger (also configures all child loggers).
    """
    global _logging_initialized

    if _logging_initialized:
        return logging.getLogger()

    # Resolve level
    if isinstance(level, str):
        level = LEVEL_MAP.get(level.upper(), logging.INFO)

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove any existing handlers (avoid duplicates in notebooks)
    root_logger.handlers.clear()

    # Format strings
    log_fmt = fmt or DEFAULT_FORMAT
    log_date_fmt = date_fmt or DEFAULT_DATE_FORMAT

    # ----- Console Handler -----
    if log_to_console:
        console_handler = _create_console_handler(
            level=level,
            use_rich=use_rich,
            use_colors=use_colors,
            fmt=log_fmt,
            date_fmt=log_date_fmt,
        )
        root_logger.addHandler(console_handler)

    # ----- File Handler -----
    if log_to_file and log_dir is not None:
        file_handler = _create_file_handler(
            log_dir=log_dir,
            filename=log_filename,
            level=level,
            fmt=log_fmt,
            date_fmt=log_date_fmt,
        )
        root_logger.addHandler(file_handler)

    # Capture Python warnings
    if capture_warnings:
        logging.captureWarnings(True)
        # Make warnings logger use the same handlers
        warnings_logger = logging.getLogger('py.warnings')
        warnings_logger.setLevel(logging.WARNING)

    # Suppress noisy third-party loggers
    _suppress_noisy_loggers()

    _logging_initialized = True

    root_logger.debug(
        "Logging initialized — level=%s, console=%s, file=%s",
        logging.getLevelName(level),
        log_to_console,
        log_to_file and log_dir is not None,
    )

    return root_logger


def _create_console_handler(
    level: int,
    use_rich: bool,
    use_colors: bool,
    fmt: str,
    date_fmt: str,
) -> logging.Handler:
    """Create the best available console handler."""

    # Option 1: Rich (prettiest)
    if use_rich and HAS_RICH:
        handler = RichHandler(
            level=level,
            console=Console(stderr=True),
            show_time=True,
            show_level=True,
            show_path=False,
            markup=True,
            rich_tracebacks=True,
            tracebacks_show_locals=False,
        )
        handler.setFormatter(logging.Formatter(RICH_FORMAT))
        return handler

    # Option 2: colorlog (colored but simpler)
    if use_colors and HAS_COLORLOG:
        color_fmt = (
            '%(log_color)s%(asctime)s | %(levelname)-8s%(reset)s | '
            '%(name)s | %(message)s'
        )
        handler = colorlog.StreamHandler(sys.stderr)
        handler.setLevel(level)
        handler.setFormatter(colorlog.ColoredFormatter(
            color_fmt,
            datefmt=date_fmt,
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'bold_red',
            },
        ))
        return handler

    # Option 3: Plain StreamHandler (always available)
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(fmt, datefmt=date_fmt))
    return handler


def _create_file_handler(
    log_dir: Union[str, Path],
    filename: str,
    level: int,
    fmt: str,
    date_fmt: str,
) -> logging.Handler:
    """Create a file handler."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    filepath = log_dir / filename

    handler = logging.FileHandler(filepath, encoding='utf-8')
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(fmt, datefmt=date_fmt))

    return handler


def _suppress_noisy_loggers():
    """Reduce verbosity of known noisy third-party loggers."""
    noisy = [
        'matplotlib',
        'matplotlib.font_manager',
        'PIL',
        'PIL.PngImagePlugin',
        'urllib3',
        'numba',
        'h5py',
        'optuna',
        'optuna.trial',
        'lightgbm',
        'xgboost',
    ]
    for name in noisy:
        logging.getLogger(name).setLevel(logging.WARNING)


# =============================================================================
# LOGGER FACTORY
# =============================================================================

def get_logger(name: str) -> logging.Logger:
    """
    Get a named logger.

    This is the primary way every module should obtain its logger.
    Calling ``get_logger(__name__)`` ensures the logger hierarchy
    follows the package structure.

    Parameters
    ----------
    name : str
        Logger name — typically ``__name__``.

    Returns
    -------
    logging.Logger
        Configured logger instance.

    Examples
    --------
    >>> logger = get_logger(__name__)
    >>> logger.info("Processing started")
    """
    return logging.getLogger(name)


# =============================================================================
# STRUCTURED LOG HELPERS
# =============================================================================

def log_stage_header(
    logger: logging.Logger,
    stage_name: str,
    description: str = "",
    width: int = 70,
):
    """
    Log a visually prominent stage header.

    Parameters
    ----------
    logger : logging.Logger
        Logger instance.
    stage_name : str
        Stage identifier.
    description : str, optional
        Human-readable description.
    width : int
        Width of the separator line.

    Example Output
    --------------
    ::

        ======================================================================
         STAGE: data_loading — Loading raw datasets
        ======================================================================
    """
    separator = "=" * width
    title = f" STAGE: {stage_name}"
    if description:
        title += f" — {description}"

    logger.info(separator)
    logger.info(title)
    logger.info(separator)


def log_metric(
    logger: logging.Logger,
    name: str,
    value: Any,
    context: str = "",
    level: int = logging.INFO,
):
    """
    Log a single metric in a structured, grep-friendly format.

    Format: ``[METRIC] <context> | <name> = <value>``

    Parameters
    ----------
    logger : logging.Logger
        Logger instance.
    name : str
        Metric name (e.g., 'macro_f1', 'accuracy').
    value : Any
        Metric value (typically float).
    context : str, optional
        Additional context (e.g., model name, fold number).
    level : int
        Logging level.

    Examples
    --------
    >>> log_metric(logger, 'macro_f1', 0.9432, context='RF fold-3')
    [METRIC] RF fold-3 | macro_f1 = 0.9432
    """
    if isinstance(value, float):
        value_str = f"{value:.6f}"
    else:
        value_str = str(value)

    parts = ["[METRIC]"]
    if context:
        parts.append(f"{context} |")
    parts.append(f"{name} = {value_str}")

    logger.log(level, " ".join(parts))


def log_metrics_dict(
    logger: logging.Logger,
    metrics: Dict[str, Any],
    context: str = "",
    level: int = logging.INFO,
):
    """
    Log multiple metrics from a dictionary.

    Parameters
    ----------
    logger : logging.Logger
        Logger instance.
    metrics : dict
        Mapping of metric names to values.
    context : str, optional
        Context string.
    level : int
        Logging level.
    """
    for name, value in sorted(metrics.items()):
        log_metric(logger, name, value, context=context, level=level)


def log_dataframe_info(
    logger: logging.Logger,
    df: 'pd.DataFrame',
    name: str = "DataFrame",
    level: int = logging.INFO,
):
    """
    Log summary information about a pandas DataFrame.

    Parameters
    ----------
    logger : logging.Logger
        Logger instance.
    df : pd.DataFrame
        DataFrame to summarize.
    name : str
        Display name for the DataFrame.
    level : int
        Logging level.

    Example Output
    --------------
    ::

        [DF] Master — shape=(8500, 1802), columns=['wavenumber_400.0', ...],
             dtypes={float64: 1800, object: 2}, memory=12.3 MB
    """
    try:
        import pandas as pd

        memory_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
        dtype_counts = df.dtypes.value_counts().to_dict()
        dtype_str = ", ".join(f"{k}: {v}" for k, v in dtype_counts.items())

        cols_preview = list(df.columns[:5])
        if len(df.columns) > 5:
            cols_preview.append("...")

        logger.log(
            level,
            "[DF] %s — shape=%s, columns=%s, dtypes={%s}, memory=%.1f MB",
            name,
            df.shape,
            cols_preview,
            dtype_str,
            memory_mb,
        )
    except Exception as e:
        logger.log(level, "[DF] %s — (info unavailable: %s)", name, e)


def log_class_distribution(
    logger: logging.Logger,
    labels,
    name: str = "labels",
    level: int = logging.INFO,
):
    """
    Log class distribution for classification labels.

    Parameters
    ----------
    logger : logging.Logger
        Logger instance.
    labels : array-like
        Class labels.
    name : str
        Display name.
    level : int
        Logging level.
    """
    try:
        import pandas as pd

        series = pd.Series(labels)
        counts = series.value_counts().sort_index()
        total = len(series)

        logger.log(level, "[DIST] %s — %d samples, %d classes:", name, total, len(counts))
        for cls, count in counts.items():
            pct = count / total * 100
            logger.log(level, "  %-30s: %5d (%5.1f%%)", cls, count, pct)
    except Exception as e:
        logger.log(level, "[DIST] %s — (info unavailable: %s)", name, e)


def log_separator(
    logger: logging.Logger,
    char: str = "-",
    width: int = 70,
    level: int = logging.INFO,
):
    """Log a simple separator line."""
    logger.log(level, char * width)


# =============================================================================
# TIMER LOGGING
# =============================================================================

class LogTimer:
    """
    Context manager that logs elapsed time for a code block.

    Parameters
    ----------
    logger : logging.Logger
        Logger to write to.
    label : str
        Description of the timed operation.
    level : int
        Logging level.

    Examples
    --------
    >>> logger = get_logger(__name__)
    >>> with LogTimer(logger, "model training"):
    ...     model.fit(X, y)
    2024-01-15 10:30:45 | INFO     | module | [TIMER] model training: 45.2s
    """

    def __init__(
        self,
        logger: logging.Logger,
        label: str = "",
        level: int = logging.INFO,
    ):
        self.logger = logger
        self.label = label
        self.level = level
        self.elapsed: float = 0.0
        self._start: Optional[float] = None

    def __enter__(self) -> 'LogTimer':
        import time
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        import time
        self.elapsed = time.perf_counter() - self._start

        if self.elapsed < 60:
            time_str = f"{self.elapsed:.2f}s"
        elif self.elapsed < 3600:
            time_str = f"{self.elapsed / 60:.2f}m"
        else:
            time_str = f"{self.elapsed / 3600:.2f}h"

        self.logger.log(self.level, "[TIMER] %s: %s", self.label, time_str)
        return False


# =============================================================================
# MAIN (for testing)
# =============================================================================

if __name__ == '__main__':
    print("Testing logging utilities...")
    print("=" * 60)

    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        # Setup logging
        root = setup_logging(
            level='DEBUG',
            log_dir=tmpdir,
            log_filename='test.log',
        )

        # Get a module logger
        logger = get_logger('test_module')

        # Test stage header
        log_stage_header(logger, 'data_loading', 'Loading raw datasets')

        # Test metric logging
        log_metric(logger, 'macro_f1', 0.9432, context='RF fold-3')

        # Test metrics dict
        log_metrics_dict(logger, {
            'accuracy': 0.95,
            'macro_f1': 0.94,
            'cohen_kappa': 0.91,
        }, context='SVM final')

        # Test separator
        log_separator(logger)

        # Test class distribution
        import random
        labels = random.choices(['PE', 'PP', 'PS', 'PET', 'PVC'], k=200)
        log_class_distribution(logger, labels, name='train_set')

        # Test timer
        import time
        with LogTimer(logger, "simulated work"):
            time.sleep(0.1)

        # Verify log file was created
        log_file = Path(tmpdir) / 'test.log'
        assert log_file.exists(), "Log file not created"
        content = log_file.read_text()
        assert '[METRIC]' in content, "Metrics not in log file"
        print(f"\n✓ Log file created: {log_file}")
        print(f"  Size: {log_file.stat().st_size} bytes")
        print(f"  Lines: {len(content.splitlines())}")

    # Reset for future tests
    _logging_initialized = False

    print("\n✅ All logging utility tests passed!")