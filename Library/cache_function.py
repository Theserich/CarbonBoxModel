import os
import time
import socket
import hashlib
import json
import pickle
from datetime import datetime, timezone
from functools import wraps

import numpy as np
from scipy.interpolate import interp1d
import inspect

from Library.dfFunctions import datadict
import logging

logger = logging.getLogger('Main logger')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

def cache_results_simple(file_format="npz", cache_dir="cache", recalc=False,
                          print_debug=True, label_fn=None, key_fn=None,
                          float_decimals=4):
    """
    key_fn: if given, called as key_fn(**bound_arguments) using the function's
        real parameter names (with defaults applied). Its return value is
        used to build the cache key INSTEAD OF hashing the array arguments.
        Use this for anything derived from measurement data (means, fits,
        interpolations) -- those can differ at the bit level across
        machines/numpy/BLAS versions even when "correct". Stable identifiers
        (year, eventyear, dt, N, ...) do not have this problem, so keying on
        them directly is the robust fix rather than trying to tolerance-round
        floats that may or may not actually agree.

    float_decimals: only used as a fallback when key_fn is not given. Rounds
        float arrays/scalars to this many decimal places before hashing, to
        absorb tiny cross-platform floating point noise. Because your
        arguments can span very different magnitudes (e.g. ~1e-12 vs ~1e3),
        a single absolute decimal count is a blunt instrument -- prefer
        key_fn whenever you can express a stable identity for the call.
        Leave as None to disable (hash exact bytes, old behavior).
    """
    assert file_format in ["json", "pickle", "npz"]
    os.makedirs(cache_dir, exist_ok=True)

    def round_floats(arr):
        if float_decimals is None:
            return arr
        return np.round(arr, decimals=float_decimals)

    def normalize(obj):
        if isinstance(obj, np.ndarray):
            arr = np.ascontiguousarray(obj)
            if arr.dtype == object:
                return (
                    "__ndarray_object__",
                    arr.shape,
                    tuple(normalize(x) for x in arr.flat)
                )
            if np.issubdtype(arr.dtype, np.floating):
                nan_mask = np.isnan(arr)
                arr2 = arr.copy()
                arr2[nan_mask] = 0.0
                arr2 = round_floats(arr2)
                return (
                    "__ndarray_float__",
                    arr.shape,
                    str(arr.dtype),
                    arr2.tobytes(),
                    nan_mask.tobytes()
                )
            return ("__ndarray__", arr.shape, str(arr.dtype), arr.tobytes())
        elif isinstance(obj, datadict):
            return ("__datadict__", tuple(sorted((k, normalize(v)) for k, v in obj.items())))
        elif isinstance(obj, interp1d):
            return (
                "__interp1d__",
                round_floats(obj.x).tobytes(),
                round_floats(obj.y).tobytes(),
                obj.bounds_error,
                str(obj.fill_value),
            )
        elif callable(obj) and (inspect.isfunction(obj) or inspect.ismethod(obj)):
            return ("__callable__", normalize_source(obj))
        elif isinstance(obj, (list, tuple)):
            return tuple(normalize(i) for i in obj)
        elif isinstance(obj, dict):
            return tuple(sorted((k, normalize(v)) for k, v in obj.items()))
        elif isinstance(obj, (float, np.floating)):
            if np.isnan(obj):
                return ("__nan__",)
            if float_decimals is None:
                return ("__float__", obj)
            return ("__float__", round(float(obj), float_decimals))
        else:
            return obj

    def normalize_source(func_or_obj):
        try:
            src = inspect.getsource(func_or_obj)
        except (OSError, TypeError):
            return repr(func_or_obj)
        src = src.replace('\r\n', '\n').replace('\r', '\n')
        lines = [line.rstrip() for line in src.split('\n')]
        lines = [line for line in lines if line.strip() != '']
        return '\n'.join(lines)

    def to_json(obj):
        if isinstance(obj, np.ndarray):
            return {"__type__": "ndarray", "dtype": str(obj.dtype),
                    "shape": obj.shape, "data": obj.tolist()}
        elif isinstance(obj, interp1d):
            return {"__type__": "interp1d", "x": obj.x.tolist(), "y": obj.y.tolist(),
                    "bounds_error": obj.bounds_error, "fill_value": obj.fill_value}
        elif isinstance(obj, (list, tuple)):
            return [to_json(i) for i in obj]
        elif isinstance(obj, datadict):
            return {"__type__": "datadict", "data": {k: to_json(v) for k, v in obj.items()}}
        elif isinstance(obj, dict):
            return {k: to_json(v) for k, v in obj.items()}
        else:
            return obj

    def from_json(obj):
        if isinstance(obj, dict) and "__type__" in obj:
            if obj["__type__"] == "ndarray":
                return np.array(obj["data"], dtype=obj["dtype"]).reshape(obj["shape"])
            if obj["__type__"] == "interp1d":
                return interp1d(obj["x"], obj["y"], bounds_error=obj["bounds_error"], fill_value=obj["fill_value"])
            if obj["__type__"] == "datadict":
                return datadict({k: from_json(v) for k, v in obj["data"].items()})
        elif isinstance(obj, list):
            return [from_json(i) for i in obj]
        elif isinstance(obj, dict):
            return {k: from_json(v) for k, v in obj.items()}
        return obj

    def save_npz(result, path):
        if not isinstance(result, (tuple, list)):
            raise ValueError(
                f"npz format requires the cached function to return a tuple or list of numpy arrays, "
                f"got {type(result).__name__} instead."
            )
        arrays = {}
        for i, v in enumerate(result):
            try:
                arrays[f"arr_{i}"] = np.asarray(v)
            except Exception as e:
                raise ValueError(
                    f"npz format: element {i} of the result could not be converted to a numpy array. "
                    f"Use file_format='json' or 'pickle' for complex return types. Original error: {e}"
                )
        tmp_path = f"{path}.tmp-{os.getpid()}-{time.time_ns()}"
        np.savez(tmp_path, n_arrays=len(result), **arrays)
        os.replace(tmp_path + ".npz", path + ".npz")

    def load_npz(path):
        d = np.load(path + ".npz", allow_pickle=False)
        n = int(d["n_arrays"])
        return tuple(d[f"arr_{i}"] for i in range(n))

    def write_manifest(entry):
        task_id = os.environ.get("SLURM_ARRAY_TASK_ID", f"pid{os.getpid()}")
        manifest_path = os.path.join(cache_dir, f"_manifest_{task_id}.jsonl")
        try:
            with open(manifest_path, "a") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except Exception as e:
            if print_debug:
                logger.info(f"Could not write manifest entry: {e}")

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            bound = None
            if key_fn is not None or label_fn is not None:
                try:
                    sig = inspect.signature(func)
                    bound = sig.bind(*args, **kwargs)
                    bound.apply_defaults()
                except Exception as e:
                    if print_debug:
                        logger.info(f"Could not bind arguments for key_fn/label_fn ({e})")

            # --- Hash / key ---
            if key_fn is not None and bound is not None:
                try:
                    key_identity = key_fn(**bound.arguments)
                    hash_input = repr((func.__name__, key_identity))
                except Exception as e:
                    if print_debug:
                        logger.info(f"key_fn failed ({e}); falling back to array hashing")
                    hash_input = repr((func.__name__, normalize(args)))
            else:
                hash_input = repr((func.__name__, normalize(args)))
            key = hashlib.sha256(hash_input.encode('utf-8')).hexdigest()

            # --- Human-readable label (cosmetic only) ---
            label = None
            if label_fn is not None and bound is not None:
                try:
                    label = label_fn(**bound.arguments)
                except Exception as e:
                    if print_debug:
                        logger.info(f"label_fn failed ({e}); falling back to hash-only filename")
            prefix = f"{label}__" if label is not None else ""

            # --- File paths ---
            if file_format == "npz":
                cache_path = os.path.join(cache_dir, f"{prefix}{key}")
                cache_file = cache_path + ".npz"
            else:
                cache_file = os.path.join(cache_dir, f"{prefix}{key}.{file_format}")
                cache_path = cache_file

            # --- Load ---
            if os.path.exists(cache_file) and not recalc:
                if print_debug:
                    logger.info(f"{func.__name__}: Loading cached result from {cache_file}")
                write_manifest({"func": func.__name__, "label": label, "key": key,
                                 "status": "hit", "file": cache_file,
                                 "time": datetime.now(timezone.utc).isoformat()})
                if file_format == "npz":
                    return load_npz(cache_path)
                with open(cache_file, "rb" if file_format == "pickle" else "r") as f:
                    data = pickle.load(f) if file_format == "pickle" else json.load(f)
                return from_json(data) if file_format == "json" else data

            # --- Compute ---
            t0 = time.time()
            try:
                result = func(*args, **kwargs)
            except Exception as e:
                write_manifest({"func": func.__name__, "label": label, "key": key,
                                 "status": "failed", "error": repr(e),
                                 "duration_s": time.time() - t0, "host": socket.gethostname(),
                                 "time": datetime.now(timezone.utc).isoformat()})
                raise
            duration = time.time() - t0

            # --- Save ---
            if file_format == "npz":
                save_npz(result, cache_path)
            elif file_format == "pickle":
                with open(cache_file, "wb") as f:
                    pickle.dump(result, f)
            else:
                with open(cache_file, "w") as f:
                    json.dump(to_json(result), f, indent=4)

            if print_debug:
                logger.info(f"{func.__name__}: Saved result to {cache_file}")

            write_manifest({"func": func.__name__, "label": label, "key": key,
                             "status": "computed", "file": cache_file,
                             "duration_s": duration, "host": socket.gethostname(),
                             "time": datetime.now(timezone.utc).isoformat()})
            return result
        return wrapper
    return decorator

def cache_results(file_format="npz", cache_dir="cache", recalc=False, print_debug=True):
    assert file_format in ["json", "pickle", "npz"]
    os.makedirs(cache_dir, exist_ok=True)

    def normalize(obj):
        if isinstance(obj, np.ndarray):
            arr = np.ascontiguousarray(obj)
            if arr.dtype == object:
                return (
                    "__ndarray_object__",
                    arr.shape,
                    tuple(normalize(x) for x in arr.flat)
                )
            if np.issubdtype(arr.dtype, np.floating):
                nan_mask = np.isnan(arr)
                arr2 = arr.copy()
                arr2[nan_mask] = 0.0
                return (
                    "__ndarray_float__",
                    arr.shape,
                    str(arr.dtype),
                    arr2.tobytes(),
                    nan_mask.tobytes()
                )
            return ("__ndarray__", arr.shape, str(arr.dtype), arr.tobytes())
        elif isinstance(obj, datadict):
            return ("__datadict__", tuple(sorted((k, normalize(v)) for k, v in obj.items())))
        elif isinstance(obj, interp1d):
            return (
                "__interp1d__",
                obj.x.tobytes(),
                obj.y.tobytes(),
                obj.bounds_error,
                str(obj.fill_value),
            )
        elif callable(obj) and (inspect.isfunction(obj) or inspect.ismethod(obj)):
            # Hash functions/callables by their normalized source code,
            # not by pickling the function object itself (which is
            # platform/identity dependent and can differ across machines).
            return ("__callable__", normalize_source(obj))
        elif isinstance(obj, (list, tuple)):
            return tuple(normalize(i) for i in obj)
        elif isinstance(obj, dict):
            return tuple(sorted((k, normalize(v)) for k, v in obj.items()))
        else:
            return obj

    def normalize_source(func_or_obj):
        """
        Return a platform-independent, whitespace/line-ending-insensitive
        representation of a function's source code, for stable hashing
        across Windows/Linux and across git checkouts.
        """
        try:
            src = inspect.getsource(func_or_obj)
        except (OSError, TypeError):
            return repr(func_or_obj)
        # Normalize line endings
        src = src.replace('\r\n', '\n').replace('\r', '\n')
        # Strip trailing whitespace per line and drop blank lines,
        # so harmless formatting changes don't bust the cache.
        lines = [line.rstrip() for line in src.split('\n')]
        lines = [line for line in lines if line.strip() != '']
        return '\n'.join(lines)

    def to_json(obj):
        if isinstance(obj, np.ndarray):
            return {
                "__type__": "ndarray",
                "dtype": str(obj.dtype),
                "shape": obj.shape,
                "data": obj.tolist()
            }
        elif isinstance(obj, interp1d):
            return {
                "__type__": "interp1d",
                "x": obj.x.tolist(),
                "y": obj.y.tolist(),
                "bounds_error": obj.bounds_error,
                "fill_value": obj.fill_value
            }
        elif isinstance(obj, (list, tuple)):
            return [to_json(i) for i in obj]
        elif isinstance(obj, datadict):
            return {
                "__type__": "datadict",
                "data": {k: to_json(v) for k, v in obj.items()}
            }
        elif isinstance(obj, dict):
            return {k: to_json(v) for k, v in obj.items()}
        else:
            return obj

    def from_json(obj):
        if isinstance(obj, dict) and "__type__" in obj:
            if obj["__type__"] == "ndarray":
                return np.array(obj["data"], dtype=obj["dtype"]).reshape(obj["shape"])
            if obj["__type__"] == "interp1d":
                return interp1d(obj["x"], obj["y"], bounds_error=obj["bounds_error"], fill_value=obj["fill_value"])
            if obj["__type__"] == "datadict":
                return datadict({k: from_json(v) for k, v in obj["data"].items()})
        elif isinstance(obj, list):
            return [from_json(i) for i in obj]
        elif isinstance(obj, dict):
            return {k: from_json(v) for k, v in obj.items()}
        return obj

    def save_npz(result, path):
        if not isinstance(result, (tuple, list)):
            raise ValueError(
                f"npz format requires the cached function to return a tuple or list of numpy arrays, "
                f"got {type(result).__name__} instead."
            )
        arrays = {}
        for i, v in enumerate(result):
            try:
                arrays[f"arr_{i}"] = np.asarray(v)
            except Exception as e:
                raise ValueError(
                    f"npz format: element {i} of the result could not be converted to a numpy array. "
                    f"Use file_format='json' or 'pickle' for complex return types. Original error: {e}"
                )
        np.savez(path, n_arrays=len(result), **arrays)

    def load_npz(path):
        d = np.load(path + ".npz", allow_pickle=False)
        n = int(d["n_arrays"])
        return tuple(d[f"arr_{i}"] for i in range(n))

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # --- Hash ---
            # Use the normalized source (line-ending and blank-line
            # insensitive) instead of raw inspect.getsource(), so the
            # same function produces the same hash on Windows and Linux.
            func_source = normalize_source(func)

            norm_args = normalize(args)
            norm_kwargs = normalize(kwargs)

            # Build the hash input as a JSON-serializable-ish, deterministic
            # string instead of pickle bytes. Pickle protocol output can
            # vary subtly with object identity/module paths across
            # platforms/interpreter versions; repr() of our already-fully-
            # normalized (plain python/numpy-bytes) structure is stable.
            hash_input = repr((func.__name__, func_source, norm_args, norm_kwargs))
            key = hashlib.sha256(hash_input.encode('utf-8')).hexdigest()

            # --- File paths ---
            if file_format == "npz":
                cache_path = os.path.join(cache_dir, key)
                cache_file = cache_path + ".npz"
            else:
                cache_file = os.path.join(cache_dir, f"{key}.{file_format}")
                cache_path = cache_file

            # --- Load ---
            if os.path.exists(cache_file) and not recalc:
                if print_debug:
                    logger.info(f"{func.__name__}: Loading cached result from {cache_file}")
                if file_format == "npz":
                    return load_npz(cache_path)
                with open(cache_file, "rb" if file_format == "pickle" else "r") as f:
                    data = pickle.load(f) if file_format == "pickle" else json.load(f)
                return from_json(data) if file_format == "json" else data

            # --- Compute ---
            result = func(*args, **kwargs)

            # --- Save ---
            if file_format == "npz":
                save_npz(result, cache_path)
            elif file_format == "pickle":
                with open(cache_file, "wb") as f:
                    pickle.dump(result, f)
            else:
                with open(cache_file, "w") as f:
                    json.dump(to_json(result), f, indent=4)

            if print_debug:
                logger.info(f"{func.__name__}: Saved result to {cache_file}")

            return result

        return wrapper

    return decorator