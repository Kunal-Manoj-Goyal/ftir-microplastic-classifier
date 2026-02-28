#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utilities Package for Microplastic FTIR Classification
=======================================================

Shared utility modules used across the entire project:
    - checkpoint: Lightweight checkpoint helpers (complementing SessionManager)
    - logging_utils: Centralized logging configuration
    - visualization: Plotting utilities for spectra, metrics, and results

These utilities are designed to be:
    - Environment-agnostic (Kaggle, Colab, Local)
    - Fail-safe (graceful degradation if optional deps missing)
    - Consistent (unified styling, formatting, color schemes)
"""

from microplastic_ftir.utils.checkpoint import (
    save_pickle,
    load_pickle,
    save_json,
    load_json,
    ensure_dir,
    file_hash,
)

from microplastic_ftir.utils.logging_utils import (
    get_logger,
    setup_logging,
    log_stage_header,
    log_metric,
    log_dataframe_info,
)

from microplastic_ftir.utils.visualization import (
    set_plot_style,
    get_color_palette,
    save_figure,
)

__all__ = [
    # Checkpoint utilities
    "save_pickle",
    "load_pickle",
    "save_json",
    "load_json",
    "ensure_dir",
    "file_hash",
    # Logging utilities
    "get_logger",
    "setup_logging",
    "log_stage_header",
    "log_metric",
    "log_dataframe_info",
    # Visualization utilities
    "set_plot_style",
    "get_color_palette",
    "save_figure",
]