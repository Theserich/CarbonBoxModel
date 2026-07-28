import os
import hashlib
import ast
import dis
import textwrap
import json
import pickle
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


def _hash_func_ast(fn):
    """
    Structural hash of a function's source, insensitive to whitespace,
    indentation style, comments, and blank lines -- but based on
    normalized source text rather than ast.dump(), since ast.dump()'s
    output format is NOT guaranteed stable across Python versions
    (node shapes changed across 3.8/3.9/3.12 etc.), which breaks
    cross-machine cache reuse even when the function logic is identical.
    """
    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError):
        return repr(fn)
    src = src.replace('\r\n', '\n').replace('\r', '\n')
    src = textwrap.dedent(src)
    lines = [line.rstrip() for line in src.split('\n')]
    lines = [line for line in lines if line.strip() != '' and not line.strip().startswith('#')]
    return '\n'.join(lines)

def cache_results_Cluster(cache_dir="cache", recalc=False,
                           print_debug=True, label_fn=None, key_fn=None,
                           float_decimals=4):
    """
    npz-only cache. Every parameter of the wrapped function -- whether passed
    positionally, by keyword, or left at its default -- is included in the
    cache key (unless key_fn is given, in which case key_fn is authoritative
    and full-argument hashing is skipped entirely).

    The key is built from the function's fully-resolved argument set
    (inspect.signature(...).bind(...).apply_defaults()), NOT from the raw
    args/kwargs the caller happened to pass. Otherwise, a parameter that's
    always left at its default would be invisible to the hash.

    Callable arguments (e.g. logprior) are hashed by their AST structure,
    not their raw source text -- this makes the hash insensitive to
    whitespace/indentation style (tabs vs spaces), comments, and blank
    lines, so the *same* function defined/copy-pasted across different
    projects or files hashes identically, while genuinely different logic
    (a changed constant, operator, or distribution) still busts the cache.

    key_fn: if given, called as key_fn(**bound_arguments) using the function's
        real parameter names (with defaults applied). Its return value is
        used to build the cache key INSTEAD OF hashing the full argument set.
        Use this for anything derived from measurement data (means, fits,
        interpolations) -- those can differ at the bit level across
        machines/numpy/BLAS versions even when "correct". Stable identifiers
        (year, eventyear, dt, N, ...) do not have this problem, so keying on
        them directly is the robust fix rather than trying to tolerance-round
        floats that may or may not actually agree.

    float_decimals: only used as a fallback when key_fn is not given. Rounds
        float arrays/scalars to this many decimal places before hashing, to
        absorb tiny cross-platform floating point noise. Leave as None to
        disable (hash exact bytes).
    """
    os.makedirs(cache_dir, exist_ok=True)

    def round_floats(arr):
        if float_decimals is None:
            return arr
        return np.round(arr, decimals=float_decimals)

    def normalize_source(func_or_obj):
        """
        Return a representation of a function's *logic* that's insensitive
        to whitespace, indentation style (tabs vs spaces), comments, and
        blank lines -- so the same function defined independently in two
        different projects/files hashes identically. Falls back to
        text-based normalization if the source can't be parsed as a
        standalone AST (e.g. built-ins, some lambda edge cases).
        """
        try:
            src = inspect.getsource(func_or_obj)
        except (OSError, TypeError):
            return repr(func_or_obj)
        try:
            tree = ast.parse(textwrap.dedent(src))
            # No whitespace, no comments, no line numbers -- just the
            # actual structure/logic of the function.
            return ast.dump(tree, annotate_fields=False, include_attributes=False)
        except SyntaxError:
            src = src.replace('\r\n', '\n').replace('\r', '\n')
            lines = [line.rstrip() for line in src.split('\n')]
            lines = [line for line in lines if line.strip() != '']
            return '\n'.join(lines)

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
            return tuple(sorted(((str(k), normalize(v)) for k, v in obj.items())))
        elif isinstance(obj, (bool, np.bool_)):
            # Must come before the int check: bool is a subclass of int in
            # Python, and True/False should not collide with 1/0.
            return ("__bool__", bool(obj))
        elif isinstance(obj, (int, np.integer)):
            # Normalizes plain Python int and every numpy integer dtype
            # (np.int32, np.int64, ...) to the same representation.
            return ("__int__", int(obj))
        elif isinstance(obj, (float, np.floating)):
            if np.isnan(obj):
                return ("__nan__",)
            if float_decimals is None:
                return ("__float__", float(obj))
            return ("__float__", round(float(obj), float_decimals))
        else:
            return obj

    def save_npz(result, path):
        """Atomic write: build under a temp name, then os.replace() into
        place, so a job killed mid-write (OOM/walltime) can never leave a
        corrupt cache file -- readers only ever see nothing or a complete file."""
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
                    f"Original error: {e}"
                )
        import time
        tmp_path = f"{path}.tmp-{os.getpid()}-{time.time_ns()}"
        np.savez(tmp_path, n_arrays=len(result), **arrays)
        os.replace(tmp_path + ".npz", path + ".npz")

    def load_npz(path):
        d = np.load(path + ".npz", allow_pickle=False)
        n = int(d["n_arrays"])
        return tuple(d[f"arr_{i}"] for i in range(n))

    def decorator(func):
        try:
            sig = inspect.signature(func)
        except (TypeError, ValueError):
            sig = None

        @wraps(func)
        def wrapper(*args, **kwargs):
            # --- Resolve the FULL effective argument set ---
            bound = None
            if sig is not None:
                try:
                    bound = sig.bind(*args, **kwargs)
                    bound.apply_defaults()
                except Exception as e:
                    if print_debug:
                        logger.info(f"Could not bind arguments ({e}); falling back to raw args/kwargs")

            # --- Hash / key ---
            if key_fn is not None and bound is not None:
                try:
                    key_identity = key_fn(**bound.arguments)
                    hash_input = repr((func.__name__, key_identity))
                except Exception as e:
                    if print_debug:
                        logger.info(f"key_fn failed ({e}); falling back to full-argument hashing")
                    if bound is not None:
                        hash_input = repr((func.__name__, normalize(bound.arguments)))
                    else:
                        hash_input = repr((func.__name__, normalize(args), normalize(kwargs)))
            elif bound is not None:
                hash_input = repr((func.__name__, normalize(bound.arguments)))
            else:
                hash_input = repr((func.__name__, normalize(args), normalize(kwargs)))
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
            cache_path = os.path.join(cache_dir, f"{prefix}{key}")
            cache_file = cache_path + ".npz"

            # --- Load ---
            if os.path.exists(cache_file) and not recalc:
                if print_debug:
                    logger.info(f"{func.__name__}: Loading cached result from {cache_file}")
                return load_npz(cache_path)

            # --- Compute ---
            result = func(*args, **kwargs)

            # --- Save ---
            save_npz(result, cache_path)

            if print_debug:
                logger.info(f"{func.__name__}: Saved result to {cache_file}")

            return result
        return wrapper
    return decorator


def cache_results2(file_format="npz", cache_dir="cache", recalc=False, print_debug=True):
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
            func_source = normalize_source(func)
            norm_args = normalize(args)
            norm_kwargs = normalize(kwargs)
            hash_input = repr((func.__name__, func_source, norm_args, norm_kwargs))
            key = hashlib.sha256(hash_input.encode('utf-8')).hexdigest()
            if file_format == "npz":
                cache_path = os.path.join(cache_dir, key)
                cache_file = cache_path + ".npz"
            else:
                cache_file = os.path.join(cache_dir, f"{key}.{file_format}")
                cache_path = cache_file
            if os.path.exists(cache_file) and not recalc:
                if print_debug:
                    logger.info(f"{func.__name__}: Loading cached result from {cache_file}")
                if file_format == "npz":
                    return load_npz(cache_path)
                with open(cache_file, "rb" if file_format == "pickle" else "r") as f:
                    data = pickle.load(f) if file_format == "pickle" else json.load(f)
                return from_json(data) if file_format == "json" else data
            result = func(*args, **kwargs)
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
        elif callable(obj):
            try:
                source = inspect.getsource(obj)
            except OSError:
                code = obj.__code__
                source = repr({
                    "bytecode": [(i.opname, i.argval) for i in dis.Bytecode(obj)],
                    "consts": code.co_consts,
                    "varnames": code.co_varnames,
                    "name": code.co_qualname,
                })
            return ("__callable__", obj.__qualname__, source)
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
