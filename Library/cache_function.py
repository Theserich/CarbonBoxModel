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
formatter = logging.Formatter(
    "%(levelname)s:%(name)s:%(message)s"
)
logger.addHandler(handler)
#

def cache_results(file_format="json", cache_dir="cache",recalc=False,print_debug=True):
    assert file_format in ["json", "pickle"]
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
                return interp1d(obj["x"],obj["y"], bounds_error=obj["bounds_error"], fill_value=obj["fill_value"])
            if obj["__type__"] == "datadict":
                return datadict({k: from_json(v) for k, v in obj["data"].items()})

        elif isinstance(obj, list):
            return [from_json(i) for i in obj]
        elif isinstance(obj, dict):
            return {k: from_json(v) for k, v in obj.items()}
        return obj
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                func_source = inspect.getsource(func)
            except OSError:
                func_source = repr(func)

            norm_args = normalize(args)
            norm_kwargs = normalize(kwargs)

            hash_bytes = pickle.dumps((func.__name__, func_source, norm_args, norm_kwargs), protocol=5)
            key = hashlib.sha256(hash_bytes).hexdigest()
            #hash_input = f"{func.__name__}|{func_source}|{norm_args}|{norm_kwargs}"
            #key = hashlib.sha256(hash_input.encode()).hexdigest()

            cache_file = os.path.join(cache_dir, f"{key}.{file_format}")

            # Load
            if os.path.exists(cache_file) and recalc == False:
                if print_debug:
                    logger.info(f"{func.__name__}: Loading cached result from {cache_file}")
                with open(cache_file, "rb" if file_format == "pickle" else "r") as f:
                    data = pickle.load(f) if file_format == "pickle" else json.load(f)

                    return from_json(data) if file_format == "json" else data

            # Compute
            result = func(*args, **kwargs)

            # Save
            with open(cache_file, "wb" if file_format == "pickle" else "w") as f:
                if file_format == "pickle":
                    pickle.dump(result, f)
                else:
                    json.dump(to_json(result), f, indent=4)
            if print_debug:
                logger.info(f"{func.__name__}:Saved result to {cache_file}")
            return result

        return wrapper

    return decorator
