from time import time
import numpy as np
import os
import hashlib
import json
import pickle
import functools
from functools import wraps
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
                          print_debug=True, label_fn=None, float_decimals=8):
    """
    float_decimals: number of decimal places floats are rounded to *only for
    the purposes of computing the cache key*. This makes the hash robust to
    platform-level floating-point noise (different BLAS/LAPACK builds,
    different numpy versions, AVX vs non-AVX code paths, cluster vs laptop,
    etc.) that can otherwise make bit-identical-looking values hash
    differently. It does NOT affect the actual stored/returned data -- only
    the bytes that go into the sha256 key.
    """
    assert file_format in ["json", "pickle", "npz"]
    os.makedirs(cache_dir, exist_ok=True)

    def round_floats(arr):
        """Round a float array to float_decimals before taking bytes, so
        tiny last-bit differences from platform/BLAS variation collapse to
        the same hash. NaNs are handled separately by the caller."""
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
            # Hash functions/callables by their normalized source code,
            # not by pickling the function object itself (which is
            # platform/identity dependent and can differ across machines).
            return ("__callable__", normalize_source(obj))
        elif isinstance(obj, (list, tuple)):
            return tuple(normalize(i) for i in obj)
        elif isinstance(obj, dict):
            return tuple(sorted((k, normalize(v)) for k, v in obj.items()))
        elif isinstance(obj, (float, np.floating)):
            # Plain python/numpy float scalars (e.g. eventyear=-3279.0,
            # a bare kwarg, etc.) get the same rounding treatment so a
            # scalar computed slightly differently on two machines still
            # hashes identically.
            if np.isnan(obj):
                return ("__nan__",)
            return ("__float__", round(float(obj), float_decimals))
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
        src = src.replace('\r\n', '\n').replace('\r', '\n')
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
        """Write atomically: build the archive under a temp name in the same
        directory, then os.replace() it into place. os.replace is atomic on
        POSIX filesystems, so a job that gets OOM-killed or hits the walltime
        limit mid-write can never leave a corrupt/truncated cache file behind
        -- readers only ever see either nothing or a complete file."""
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
        """One manifest file per process (keyed by SLURM array task id if
        present, else PID) so concurrent array tasks on a shared/NFS scratch
        filesystem never write to the same file at once. Merge them later
        with `cat _manifest_*.jsonl > all_manifest.jsonl` for a full picture
        of what ran, what hit cache, what failed, and how long it took."""
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
            # --- Hash ---
            # Only the function name and its normalized positional args
            # go into the hash. Kwargs and source code are intentionally
            # excluded from the cache key. Floats are rounded (see
            # float_decimals) so the key is stable across machines/BLAS.
            norm_args = normalize(args)

            hash_input = repr((func.__name__, norm_args))
            key = hashlib.sha256(hash_input.encode('utf-8')).hexdigest()

            # --- Human-readable label ---
            # Purely cosmetic: doesn't affect the hash/correctness, just makes
            # the cache directory browsable (e.g. `ls | sort` shows years in
            # order) instead of being an opaque list of hex strings.
            label = None
            if label_fn is not None:
                try:
                    sig = inspect.signature(func)
                    bound = sig.bind(*args, **kwargs)
                    bound.apply_defaults()
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
                write_manifest({
                    "func": func.__name__, "label": label, "key": key,
                    "status": "hit", "file": cache_file,
                    "time": datetime.now(timezone.utc).isoformat(),
                })
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
                write_manifest({
                    "func": func.__name__, "label": label, "key": key,
                    "status": "failed", "error": repr(e),
                    "duration_s": time.time() - t0,
                    "host": socket.gethostname(),
                    "time": datetime.now(timezone.utc).isoformat(),
                })
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

            write_manifest({
                "func": func.__name__, "label": label, "key": key,
                "status": "computed", "file": cache_file,
                "duration_s": duration, "host": socket.gethostname(),
                "time": datetime.now(timezone.utc).isoformat(),
            })
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