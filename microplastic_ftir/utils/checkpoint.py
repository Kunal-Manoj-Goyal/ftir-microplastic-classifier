#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Checkpoint Utilities for Microplastic FTIR Classification
==========================================================

Lightweight, standalone checkpoint helpers that complement the full
SessionManager. These functions provide simple save/load operations
without requiring session state, making them useful for:
    - Quick intermediate saves within a stage
    - Saving/loading individual objects (arrays, DataFrames, dicts)
    - File hashing for integrity verification
    - Directory management

These are the "low-level" primitives; SessionManager (session_manager.py)
builds higher-level stage-aware logic on top of these patterns.

Usage:
------
    >>> from microplastic_ftir.utils.checkpoint import save_pickle, load_pickle
    >>>
    >>> save_pickle(my_data, '/path/to/checkpoint.pkl')
    >>> restored = load_pickle('/path/to/checkpoint.pkl')
    >>>
    >>> from microplastic_ftir.utils.checkpoint import save_json, load_json
    >>> save_json(my_dict, '/path/to/config.json')

Author: Your Name
Date: 2024
"""

import os
import json
import pickle
import hashlib
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Union
from datetime import datetime

# Optional dependency imports
try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


# =============================================================================
# DIRECTORY UTILITIES
# =============================================================================

def ensure_dir(path: Union[str, Path], parents: bool = True) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    If ``path`` looks like a file path (has a suffix), the parent
    directory is created instead.

    Parameters
    ----------
    path : str or Path
        Directory path (or file path whose parent should exist).
    parents : bool
        If True, create parent directories as needed.

    Returns
    -------
    Path
        The directory path that was ensured to exist.

    Examples
    --------
    >>> ensure_dir('/outputs/figures')
    PosixPath('/outputs/figures')

    >>> # Also works with file paths — creates parent dir
    >>> ensure_dir('/outputs/figures/plot.png')
    PosixPath('/outputs/figures')
    """
    path = Path(path)

    # If it looks like a file path, ensure the parent
    if path.suffix:
        dir_path = path.parent
    else:
        dir_path = path

    dir_path.mkdir(parents=parents, exist_ok=True)
    return dir_path


# =============================================================================
# PICKLE SAVE / LOAD
# =============================================================================

def save_pickle(
    data: Any,
    filepath: Union[str, Path],
    use_joblib: bool = True,
    compress: int = 3,
    protocol: int = pickle.HIGHEST_PROTOCOL,
) -> Path:
    """
    Save data to a pickle (or joblib) file.

    Prefers joblib when available because it handles large NumPy arrays
    and scikit-learn estimators more efficiently.

    Parameters
    ----------
    data : Any
        Object to serialize.
    filepath : str or Path
        Destination file path.
    use_joblib : bool
        Use joblib if available (recommended for large arrays / sklearn).
    compress : int
        Joblib compression level (0–9). Ignored for plain pickle.
    protocol : int
        Pickle protocol version.

    Returns
    -------
    Path
        Path to the saved file.

    Raises
    ------
    OSError
        If the file cannot be written.

    Examples
    --------
    >>> save_pickle({'a': 1, 'b': [2, 3]}, 'test.pkl')
    PosixPath('test.pkl')
    """
    filepath = Path(filepath)
    ensure_dir(filepath)

    if use_joblib and HAS_JOBLIB:
        joblib.dump(data, filepath, compress=compress, protocol=protocol)
    else:
        with open(filepath, 'wb') as f:
            pickle.dump(data, f, protocol=protocol)

    return filepath


def load_pickle(
    filepath: Union[str, Path],
    use_joblib: bool = True,
    default: Any = None,
) -> Any:
    """
    Load data from a pickle (or joblib) file.

    Parameters
    ----------
    filepath : str or Path
        Source file path.
    use_joblib : bool
        Attempt joblib.load first (auto-detects format).
    default : Any, optional
        Value to return if the file does not exist.
        If None and file missing, raises FileNotFoundError.

    Returns
    -------
    Any
        Deserialized object.

    Raises
    ------
    FileNotFoundError
        If file does not exist and no default is provided.

    Examples
    --------
    >>> data = load_pickle('test.pkl')
    >>> data = load_pickle('missing.pkl', default={})
    {}
    """
    filepath = Path(filepath)

    if not filepath.exists():
        if default is not None:
            return default
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

    if use_joblib and HAS_JOBLIB:
        try:
            return joblib.load(filepath)
        except Exception:
            # Fall back to plain pickle
            pass

    with open(filepath, 'rb') as f:
        return pickle.load(f)


# =============================================================================
# JSON SAVE / LOAD
# =============================================================================

def _json_default_serializer(obj: Any) -> Any:
    """
    Custom JSON serializer for objects not handled by the default encoder.

    Handles: datetime, Path, numpy scalars/arrays, pandas objects, sets, bytes.
    """
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, set):
        return sorted(list(obj))
    if isinstance(obj, bytes):
        return obj.decode('utf-8', errors='replace')

    if HAS_NUMPY:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)

    if HAS_PANDAS:
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        if isinstance(obj, pd.Series):
            return obj.to_dict()

    # Last resort
    return str(obj)


def save_json(
    data: Any,
    filepath: Union[str, Path],
    indent: int = 2,
    sort_keys: bool = False,
    ensure_ascii: bool = False,
) -> Path:
    """
    Save data to a JSON file with robust serialization.

    Parameters
    ----------
    data : Any
        JSON-serializable object (dict, list, etc.).
    filepath : str or Path
        Destination file path.
    indent : int
        Indentation level for pretty-printing.
    sort_keys : bool
        Sort dictionary keys.
    ensure_ascii : bool
        If False, allow non-ASCII characters.

    Returns
    -------
    Path
        Path to the saved file.

    Examples
    --------
    >>> save_json({'metric': 0.95, 'model': 'SVM'}, 'results.json')
    """
    filepath = Path(filepath)
    ensure_dir(filepath)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(
            data, f,
            indent=indent,
            sort_keys=sort_keys,
            ensure_ascii=ensure_ascii,
            default=_json_default_serializer,
        )

    return filepath


def load_json(
    filepath: Union[str, Path],
    default: Any = None,
) -> Any:
    """
    Load data from a JSON file.

    Parameters
    ----------
    filepath : str or Path
        Source file path.
    default : Any, optional
        Value to return if the file does not exist.

    Returns
    -------
    Any
        Deserialized JSON object.

    Raises
    ------
    FileNotFoundError
        If file does not exist and no default is provided.
    """
    filepath = Path(filepath)

    if not filepath.exists():
        if default is not None:
            return default
        raise FileNotFoundError(f"JSON file not found: {filepath}")

    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


# =============================================================================
# NUMPY SAVE / LOAD (convenience wrappers)
# =============================================================================

def save_numpy(
    array,
    filepath: Union[str, Path],
    compressed: bool = True,
) -> Path:
    """
    Save a NumPy array to disk.

    Parameters
    ----------
    array : np.ndarray
        Array to save.
    filepath : str or Path
        Destination file path (.npy or .npz).
    compressed : bool
        Use compressed .npz format.

    Returns
    -------
    Path
        Path to the saved file.
    """
    if not HAS_NUMPY:
        raise ImportError("NumPy is required for save_numpy")

    filepath = Path(filepath)
    ensure_dir(filepath)

    if compressed:
        filepath = filepath.with_suffix('.npz')
        np.savez_compressed(filepath, data=array)
    else:
        filepath = filepath.with_suffix('.npy')
        np.save(filepath, array)

    return filepath


def load_numpy(
    filepath: Union[str, Path],
    key: str = 'data',
) -> 'np.ndarray':
    """
    Load a NumPy array from disk.

    Parameters
    ----------
    filepath : str or Path
        Source file path (.npy or .npz).
    key : str
        Key to use for .npz files.

    Returns
    -------
    np.ndarray
        Loaded array.
    """
    if not HAS_NUMPY:
        raise ImportError("NumPy is required for load_numpy")

    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"NumPy file not found: {filepath}")

    if filepath.suffix == '.npz':
        with np.load(filepath, allow_pickle=True) as npz:
            return npz[key]
    else:
        return np.load(filepath, allow_pickle=True)


# =============================================================================
# FILE HASHING
# =============================================================================

def file_hash(
    filepath: Union[str, Path],
    algorithm: str = 'md5',
    chunk_size: int = 8192,
) -> str:
    """
    Compute a hash digest of a file for integrity checking.

    Parameters
    ----------
    filepath : str or Path
        File to hash.
    algorithm : str
        Hash algorithm ('md5', 'sha1', 'sha256').
    chunk_size : int
        Read chunk size in bytes.

    Returns
    -------
    str
        Hex digest string.

    Examples
    --------
    >>> h = file_hash('model.pkl')
    >>> print(h)
    'a3f2b8c9...'
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Cannot hash — file not found: {filepath}")

    hasher = hashlib.new(algorithm)

    with open(filepath, 'rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)

    return hasher.hexdigest()


# =============================================================================
# DATAFRAME SAVE / LOAD (convenience wrappers)
# =============================================================================

def save_dataframe(
    df: 'pd.DataFrame',
    filepath: Union[str, Path],
    index: bool = False,
    **kwargs,
) -> Path:
    """
    Save a pandas DataFrame, auto-detecting format from extension.

    Supported: .csv, .parquet, .feather, .pkl

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save.
    filepath : str or Path
        Destination path.
    index : bool
        Whether to write the row index.
    **kwargs
        Extra keyword arguments forwarded to the pandas writer.

    Returns
    -------
    Path
        Path to the saved file.
    """
    if not HAS_PANDAS:
        raise ImportError("pandas is required for save_dataframe")

    filepath = Path(filepath)
    ensure_dir(filepath)

    suffix = filepath.suffix.lower()

    if suffix == '.csv':
        df.to_csv(filepath, index=index, **kwargs)
    elif suffix == '.parquet':
        df.to_parquet(filepath, index=index, **kwargs)
    elif suffix in ('.pkl', '.pickle'):
        df.to_pickle(filepath, **kwargs)
    elif suffix == '.feather':
        df.to_feather(filepath, **kwargs)
    else:
        # Default to CSV
        filepath = filepath.with_suffix('.csv')
        df.to_csv(filepath, index=index, **kwargs)
        warnings.warn(
            f"Unrecognized extension '{suffix}', saved as CSV: {filepath}"
        )

    return filepath


def load_dataframe(
    filepath: Union[str, Path],
    **kwargs,
) -> 'pd.DataFrame':
    """
    Load a pandas DataFrame, auto-detecting format from extension.

    Parameters
    ----------
    filepath : str or Path
        Source file path.
    **kwargs
        Extra keyword arguments forwarded to the pandas reader.

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame.
    """
    if not HAS_PANDAS:
        raise ImportError("pandas is required for load_dataframe")

    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"DataFrame file not found: {filepath}")

    suffix = filepath.suffix.lower()

    if suffix == '.csv':
        return pd.read_csv(filepath, **kwargs)
    elif suffix == '.parquet':
        return pd.read_parquet(filepath, **kwargs)
    elif suffix in ('.pkl', '.pickle'):
        return pd.read_pickle(filepath, **kwargs)
    elif suffix == '.feather':
        return pd.read_feather(filepath, **kwargs)
    else:
        # Attempt CSV
        return pd.read_csv(filepath, **kwargs)


# =============================================================================
# TIMING UTILITIES
# =============================================================================

class Timer:
    """
    Simple context-manager timer for profiling code blocks.

    Examples
    --------
    >>> with Timer("data loading"):
    ...     data = load_big_file()
    [Timer] data loading: 12.34s
    """

    def __init__(self, label: str = "", verbose: bool = True):
        self.label = label
        self.verbose = verbose
        self.elapsed: float = 0.0
        self._start: Optional[float] = None

    def __enter__(self) -> 'Timer':
        import time
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        import time
        self.elapsed = time.perf_counter() - self._start

        if self.verbose and self.label:
            if self.elapsed < 60:
                time_str = f"{self.elapsed:.2f}s"
            elif self.elapsed < 3600:
                time_str = f"{self.elapsed / 60:.2f}m"
            else:
                time_str = f"{self.elapsed / 3600:.2f}h"
            print(f"[Timer] {self.label}: {time_str}")

        return False


# =============================================================================
# MAIN (for testing)
# =============================================================================

if __name__ == '__main__':
    import tempfile

    print("Testing checkpoint utilities...")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Test pickle
        pkl_path = Path(tmpdir) / 'test.pkl'
        save_pickle({'a': 1, 'b': [2, 3]}, pkl_path)
        loaded = load_pickle(pkl_path)
        assert loaded == {'a': 1, 'b': [2, 3]}, "Pickle round-trip failed"
        print("✓ Pickle save/load")

        # Test JSON
        json_path = Path(tmpdir) / 'test.json'
        save_json({'metric': 0.95, 'model': 'SVM'}, json_path)
        loaded = load_json(json_path)
        assert loaded['metric'] == 0.95, "JSON round-trip failed"
        print("✓ JSON save/load")

        # Test file hash
        h = file_hash(pkl_path)
        assert len(h) == 32, "MD5 hash length unexpected"
        print(f"✓ File hash: {h[:16]}...")

        # Test ensure_dir
        nested = Path(tmpdir) / 'a' / 'b' / 'c'
        result = ensure_dir(nested)
        assert result.exists(), "ensure_dir failed"
        print("✓ ensure_dir")

        # Test default on missing file
        missing = load_pickle(Path(tmpdir) / 'nope.pkl', default={'empty': True})
        assert missing == {'empty': True}, "Default return failed"
        print("✓ Default on missing file")

        # Test Timer
        with Timer("test sleep"):
            import time
            time.sleep(0.1)
        print("✓ Timer")

        # Test numpy if available
        if HAS_NUMPY:
            arr = np.random.randn(100, 50)
            np_path = Path(tmpdir) / 'test_array'
            save_numpy(arr, np_path)
            loaded_arr = load_numpy(np_path.with_suffix('.npz'))
            assert np.allclose(arr, loaded_arr), "NumPy round-trip failed"
            print("✓ NumPy save/load")

        # Test DataFrame if available
        if HAS_PANDAS:
            df = pd.DataFrame({'x': [1, 2, 3], 'y': ['a', 'b', 'c']})
            csv_path = Path(tmpdir) / 'test.csv'
            save_dataframe(df, csv_path)
            loaded_df = load_dataframe(csv_path)
            assert list(loaded_df.columns) == ['x', 'y'], "DataFrame round-trip failed"
            print("✓ DataFrame save/load (CSV)")

    print("\n✅ All checkpoint utility tests passed!")