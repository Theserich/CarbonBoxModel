from time import time
import numpy as np
import os
import hashlib
import json
import pickle
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
        elif isinstance(obj, (list, tuple)):
            return tuple(normalize(i) for i in obj)
        elif isinstance(obj, dict):
            return tuple(sorted((k, normalize(v)) for k, v in obj.items()))
        else:
            return obj

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
        """
        Save a tuple/list of numpy arrays to an .npz file.
        'path' should NOT include the .npz extension (np.savez adds it).
        """
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
        """
        Load a tuple of numpy arrays from an .npz file.
        'path' should NOT include the .npz extension.
        """
        d = np.load(path + ".npz", allow_pickle=False)
        n = int(d["n_arrays"])
        return tuple(d[f"arr_{i}"] for i in range(n))

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # --- Hash ---
            try:
                func_source = inspect.getsource(func)
            except OSError:
                func_source = repr(func)

            norm_args = normalize(args)
            norm_kwargs = normalize(kwargs)
            hash_bytes = pickle.dumps((func.__name__, func_source, norm_args, norm_kwargs), protocol=5)
            key = hashlib.sha256(hash_bytes).hexdigest()

            # --- File paths ---
            # For npz, np.savez automatically appends .npz, so we keep two variables:
            #   cache_file     — the actual file on disk (used for os.path.exists)
            #   cache_path     — the path passed to np.savez / np.load (without .npz)
            if file_format == "npz":
                cache_path = os.path.join(cache_dir, key)       # no extension
                cache_file = cache_path + ".npz"                 # actual file
            else:
                cache_file = os.path.join(cache_dir, f"{key}.{file_format}")
                cache_path = cache_file                          # same for json/pickle

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
            else:  # json
                with open(cache_file, "w") as f:
                    json.dump(to_json(result), f, indent=4)

            if print_debug:
                logger.info(f"{func.__name__}: Saved result to {cache_file}")

            return result

        return wrapper

    return decorator