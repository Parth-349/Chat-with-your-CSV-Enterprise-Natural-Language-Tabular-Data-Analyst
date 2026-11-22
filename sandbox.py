"""
Zero-Trust Execution Sandbox and Tabular Data Utilities for Chat-with-your-CSV.
Enforces strict AST allowlist, process/timeout isolation, environment scrubbing,
and robust data loading/sanitization.
"""

import ast
import datetime
import difflib
import io
import math
import multiprocessing as mp
import os
import re
import statistics
import string
import sys
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns


# ==============================================================================
# 1. Zero-Trust AST Sandbox Allowlist & Rules (Jury Q1 Defense)
# ==============================================================================

ALLOWED_MODULES = {
    "pandas",
    "numpy",
    "matplotlib",
    "matplotlib.pyplot",
    "seaborn",
    "plotly",
    "plotly.express",
    "plotly.graph_objects",
    "math",
    "statistics",
    "datetime",
}

PROHIBITED_CALLS = {
    "open",
    "eval",
    "exec",
    "compile",
    "__import__",
    "getattr",
    "setattr",
    "delattr",
    "globals",
    "locals",
    "vars",
    "dir",
    "input",
    "breakpoint",
}

PROHIBITED_METHODS = {
    # Pandas / Disk / Network I/O
    "read_csv",
    "read_table",
    "read_fwf",
    "read_clipboard",
    "read_excel",
    "read_json",
    "read_parquet",
    "read_sql",
    "read_pickle",
    "read_html",
    "read_xml",
    "read_stata",
    "read_sas",
    "read_spss",
    "read_orc",
    "read_feather",
    "read_gbq",
    "write_json",
    "to_csv",
    "to_clipboard",
    "to_excel",
    "to_json",
    "to_parquet",
    "to_pickle",
    "to_xml",
    "to_stata",
    "to_feather",
    "to_orc",
    "to_gbq",
    "to_hdf",
    "to_sql",
    "HDFStore",
    # NumPy / Matplotlib / Seaborn / Plotly I/O & network
    "savefig",
    "write_html",
    "write_image",
    "show",
    "imread",
    "imsave",
    "load",
    "save",
    "savez",
    "savez_compressed",
    "loadtxt",
    "genfromtxt",
    "fromregex",
    "memmap",
    "fromfile",
    "tofile",
    "load_dataset",
    # Dynamic string evaluation & formatting escapes
    "eval",
    "query",
    "format_map",
}

PROHIBITED_MODULE_ATTRS = {
    "os",
    "sys",
    "subprocess",
    "shutil",
    "pathlib",
    "importlib",
    "builtins",
    "modules",
    "environ",
    "getenv",
    "system",
    "popen",
    "spawn",
    "socket",
    "urllib",
    "requests",
    "http",
    "ctypes",
    "inspect",
    "traceback",
    "codecs",
}

SAFE_BUILTINS: Dict[str, Any] = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "isinstance": isinstance,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "print": print,
    "range": range,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
    "True": True,
    "False": False,
    "None": None,
    # Standard exceptions for benign control flow
    "Exception": Exception,
    "ValueError": ValueError,
    "TypeError": TypeError,
    "KeyError": KeyError,
    "IndexError": IndexError,
}


def _is_safe_format_call(node: ast.Attribute) -> Tuple[bool, Optional[str]]:
    """Checks literal format strings for field traversal before execution."""
    try:
        for _, field_name, format_spec, _ in string.Formatter().parse(node.value.value):
            if field_name is not None:
                if any(ch in field_name for ch in (".", "[", "]")) or "__" in field_name:
                    return False, "Attribute/item traversal inside '.format()' is prohibited."
            if format_spec and ("{" in format_spec or "__" in format_spec):
                return False, "Nested/dunder format specs inside '.format()' are prohibited."
    except ValueError as e:
        return False, f"Invalid format string: {e}"

    return True, None


def _safe_fmt_attr(obj: Any) -> Any:
    """Guards dynamically obtained string formats before returning their method."""
    if isinstance(obj, str):
        try:
            for _, field_name, format_spec, _ in string.Formatter().parse(obj):
                if field_name is not None and (
                    any(ch in field_name for ch in (".", "[", "]")) or "__" in field_name
                ):
                    raise PermissionError(
                        "Security Restriction: Attribute/item traversal inside '.format()' is prohibited."
                    )
                if format_spec and ("{" in format_spec or "__" in format_spec):
                    raise PermissionError(
                        "Security Restriction: Attribute/item traversal inside '.format()' is prohibited."
                    )
        except ValueError:
            # Preserve Python's normal invalid-format-string error at the call site.
            pass
    return obj.format


def validate_code_ast(code: str) -> Tuple[bool, Optional[str]]:
    """
    Validates Python code against a strict AST allowlist BEFORE execution.
    Permits innocent string literals and bracketed column names while rejecting:
      - Unauthorized module imports (e.g. os, sys, subprocess, importlib)
      - Dunder & private attribute access (__class__, _libs, plt.sys, etc.)
      - Dangerous built-in calls (open, eval, exec, getattr, globals, etc.)
      - Disk, network, database I/O, and dynamic eval/query methods
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"Syntax Error: {e.msg} (line {e.lineno})"

    for node in ast.walk(tree):
        # 1. Check Import statements
        if isinstance(node, ast.Import):
            for alias in node.names:
                mod_name = alias.name
                root_pkg = mod_name.split(".")[0]
                if mod_name not in ALLOWED_MODULES and root_pkg not in ALLOWED_MODULES:
                    return False, f"Importing module '{mod_name}' is prohibited. Only standard data science libraries are permitted."

        # 2. Check ImportFrom statements
        elif isinstance(node, ast.ImportFrom):
            if node.level > 0:
                return False, "Relative imports are prohibited."
            if node.module:
                mod_name = node.module
                root_pkg = mod_name.split(".")[0]
                if mod_name not in ALLOWED_MODULES and root_pkg not in ALLOWED_MODULES:
                    return False, f"Importing from module '{node.module}' is prohibited."

        # 3. Check Attribute access (dunders, private attrs, I/O methods, module attrs)
        elif isinstance(node, ast.Attribute):
            attr = node.attr
            if attr == "format":
                if not isinstance(node.ctx, ast.Load):
                    return False, "Assigning or deleting '.format' attributes is prohibited."
                if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                    ok_fmt, fmt_err = _is_safe_format_call(node)
                    if not ok_fmt:
                        return False, fmt_err
            if attr.startswith("_") or attr.endswith("__"):
                return False, f"Private/dunder attribute access ('{attr}') is prohibited for security."
            if attr in PROHIBITED_METHODS:
                return False, f"File, network, or dynamic eval method '{attr}' is prohibited in the sandbox."
            if attr in PROHIBITED_MODULE_ATTRS:
                return False, f"Access to runtime/system attribute '{attr}' is prohibited."

        # 5. Check Identifiers (dunder names, dangerous builtin calls/references)
        elif isinstance(node, ast.Name):
            name = node.id
            if name.startswith("__") or name.endswith("__"):
                return False, f"Dunder identifier access ('{name}') is prohibited."
            if name in PROHIBITED_CALLS:
                return False, f"Call or reference to prohibited function '{name}' is blocked."

    return True, None


# ==============================================================================
# 2. Isolated Worker Process & Execution Sandbox
# ==============================================================================

def _isolated_worker(code: str, df: pd.DataFrame, conn: Any) -> None:
    """
    Target function executed inside an isolated child worker process.
    - Scrubs all sensitive API keys from environment variables.
    - Runs in restricted globals with SAFE_BUILTINS and pre-injected libraries.
    - Captures stdout, return values, Matplotlib PNG bytes, and Plotly JSON.
    """
    # 1. Scrub sensitive environment variables inside child process
    for key in list(os.environ.keys()):
        key_upper = key.upper()
        if any(term in key_upper for term in ["API_KEY", "SECRET", "TOKEN", "PASSWORD", "GEMINI", "GOOGLE", "OPENAI", "GROQ"]):
            os.environ.pop(key, None)

    # 2. Redirect stdout
    redirected_output = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = redirected_output

    error = None
    figure_png = None
    plotly_json = None
    result_val = None
    result_df = None

    # Setup matplotlib/seaborn isolated for this process
    plt.close("all")
    sns.set_theme(style="whitegrid", palette="deep")

    exec_globals: Dict[str, Any] = {
        "__builtins__": SAFE_BUILTINS,
        "__safe_fmt_attr__": _safe_fmt_attr,
        "df": df.copy(),
        "pd": pd,
        "np": np,
        "plt": plt,
        "sns": sns,
        "px": px,
        "go": go,
        "math": math,
        "statistics": statistics,
        "datetime": datetime,
        "result": None,
        "result_df": None,
        "result_fig": None,
    }

    try:
        class SafeFormatTransformer(ast.NodeTransformer):
            def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
                node.value = self.visit(node.value)
                if node.attr == "format" and isinstance(node.ctx, ast.Load):
                    return ast.copy_location(
                        ast.Call(
                            func=ast.Name(id="__safe_fmt_attr__", ctx=ast.Load()),
                            args=[node.value],
                            keywords=[],
                        ),
                        node,
                    )
                return node

        tree = SafeFormatTransformer().visit(ast.parse(code))
        ast.fix_missing_locations(tree)
        exec(compile(tree, "<sandbox>", "exec"), exec_globals)

        # 3. Capture Matplotlib figure if generated (with memory-bomb dimension guard)
        if plt.get_fignums():
            fig = plt.gcf()
            w, h = fig.get_size_inches()
            if w > 20 or h > 20:
                fig.set_size_inches(min(w, 20), min(h, 20))
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
            buf.seek(0)
            figure_png = buf.getvalue()
            plt.close("all")

        result_val = exec_globals.get("result")
        result_df = exec_globals.get("result_df")
        result_fig = exec_globals.get("result_fig")

        if result_fig is None and isinstance(result_val, go.Figure):
            result_fig = result_val
            result_val = None
        if result_fig is not None:
            if not isinstance(result_fig, go.Figure):
                raise TypeError("result_fig must be a Plotly graph_objects.Figure.")
            plotly_json = result_fig.to_json()

        # 4. Handle Series to DataFrame promotion preserving labels/index
        if result_df is None:
            if isinstance(result_val, pd.DataFrame):
                result_df = result_val
                result_val = None
            elif isinstance(result_val, pd.Series):
                idx_name = result_val.index.name or "index"
                val_name = result_val.name or "value"
                s_df = result_val.reset_index()
                if len(s_df.columns) == 2 and s_df.columns[0] == "index" and idx_name == "index":
                    s_df.columns = ["category", val_name]
                result_df = s_df
                result_val = None

    except PermissionError as e:
        message = str(e)
        error = message if message.startswith("Security Restriction: ") else f"Security Restriction: {message}"
    except Exception as e:
        error = f"{type(e).__name__}: {str(e)}"
    finally:
        sys.stdout = old_stdout

    stdout_str = redirected_output.getvalue().strip()

    try:
        conn.send({
            "stdout": stdout_str,
            "result": result_val,
            "result_df": result_df,
            "figure_png": figure_png,
            "plotly_json": plotly_json,
            "error": error,
        })
    except Exception as q_err:
        conn.send({
            "stdout": stdout_str,
            "result": str(result_val) if result_val is not None else None,
            "result_df": None,
            "figure_png": figure_png,
            "plotly_json": plotly_json,
            "error": f"SerializationError: {q_err}",
        })
    finally:
        conn.close()


def execute_pandas_code(code: str, df: pd.DataFrame, timeout_seconds: int = 10) -> Dict[str, Any]:
    """
    Executes Python/pandas code securely within an isolated subprocess worker.
    Enforces AST pre-validation and timeout isolation.
    """
    cleaned_code = re.sub(r"^```(?:python)?\s*", "", code, flags=re.MULTILINE)
    cleaned_code = re.sub(r"^```\s*$", "", cleaned_code, flags=re.MULTILINE).strip()

    # Pre-execution AST security validation
    is_valid, ast_err = validate_code_ast(cleaned_code)
    if not is_valid:
        return {
            "stdout": "",
            "result": None,
            "result_df": None,
            "figure_png": None,
            "plotly_json": None,
            "figure": None,
            "error": f"Security Restriction: {ast_err}",
            "cleaned_code": cleaned_code,
        }

    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    process = ctx.Process(target=_isolated_worker, args=(cleaned_code, df, child_conn))
    process.start()
    child_conn.close()

    if parent_conn.poll(timeout=timeout_seconds):
        try:
            res = parent_conn.recv()
        except EOFError:
            res = {
                "stdout": "",
                "result": None,
                "result_df": None,
                "figure_png": None,
                "plotly_json": None,
                "error": "Process closed pipe without sending data.",
            }
        process.join(timeout=1)
        if process.is_alive():
            process.terminate()
            process.join()

        res["figure"] = res.get("figure_png")
        res["cleaned_code"] = cleaned_code
        return res
    else:
        if process.is_alive():
            process.terminate()
            process.join(timeout=1)
            if process.is_alive():
                try:
                    process.kill()
                except Exception:
                    pass
                process.join()

        return {
            "stdout": "",
            "result": None,
            "result_df": None,
            "figure_png": None,
            "plotly_json": None,
            "figure": None,
            "error": f"Execution Timeout: Query took longer than {timeout_seconds} seconds to execute. This may be caused by an infinite loop or heavy computational join.",
            "cleaned_code": cleaned_code,
        }


# ==============================================================================
# 3. Robust CSV Data Loading & Edge Case Handling
# ==============================================================================

def load_csv_data(file_bytes_or_path: Any, filename: str = "") -> pd.DataFrame:
    """
    Loads CSV data cleanly with automatic encoding fallback, delimiter auto-detection,
    and safe handling for empty/0-row datasets.
    """
    raw_bytes: bytes
    if isinstance(file_bytes_or_path, (str, bytes, bytearray)):
        if isinstance(file_bytes_or_path, str):
            if os.path.exists(file_bytes_or_path):
                with open(file_bytes_or_path, "rb") as f:
                    raw_bytes = f.read()
            else:
                raw_bytes = file_bytes_or_path.encode("utf-8")
        else:
            raw_bytes = bytes(file_bytes_or_path)
    elif hasattr(file_bytes_or_path, "read"):
        if hasattr(file_bytes_or_path, "seek"):
            file_bytes_or_path.seek(0)
        raw_bytes = file_bytes_or_path.read()
        if isinstance(raw_bytes, str):
            raw_bytes = raw_bytes.encode("utf-8")
    else:
        raise ValueError(f"Unsupported input type for CSV loader: {type(file_bytes_or_path)}")

    if len(raw_bytes.strip()) == 0:
        raise pd.errors.EmptyDataError("The uploaded CSV file is empty (0 bytes).")

    encodings_to_try = ["utf-8", "utf-8-sig", "latin1"]
    last_error: Optional[Exception] = None

    for enc in encodings_to_try:
        try:
            bio = io.BytesIO(raw_bytes)
            return pd.read_csv(bio, encoding=enc)
        except UnicodeDecodeError as ue:
            last_error = ue
            continue
        except pd.errors.ParserError:
            try:
                bio = io.BytesIO(raw_bytes)
                return pd.read_csv(bio, encoding=enc, sep=None, engine="python")
            except Exception as pe:
                last_error = pe
                continue
        except Exception as e:
            last_error = e
            break

    if last_error is not None:
        raise last_error

    return pd.DataFrame()


# ==============================================================================
# 4. Schema Sanitization & Fuzzy Column Error Handling (Jury Q2 Defense)
# ==============================================================================

def sanitize_text(text: Any, max_len: int = 60) -> str:
    """Strips control characters, newlines, and escapes special delimiters."""
    s = str(text)
    s = re.sub(r"[\r\n\t\x00-\x1f\x7f-\x9f]", " ", s).strip()
    s = s.replace("`", "'")
    if len(s) > max_len:
        s = s[: max_len - 3] + "..."
    return s


def get_df_schema_prompt(df: pd.DataFrame) -> str:
    """Generates a sanitized schema description of the DataFrame for LLM prompts."""
    lines = [
        f"DataFrame shape: {df.shape[0]} rows, {df.shape[1]} columns",
        "Columns and Data Types:",
    ]
    for col in df.columns:
        clean_col = sanitize_text(col, max_len=50)
        dtype = str(df[col].dtype)
        null_count = int(df[col].isnull().sum())
        sample_vals = [sanitize_text(x, max_len=40) for x in df[col].dropna().unique()[:3]]
        sample_str = ", ".join(sample_vals) if sample_vals else "None"
        lines.append(f" - `{clean_col}` ({dtype}) | {null_count} nulls | Examples: [{sample_str}]")

    lines.append("\nFirst 3 rows preview:")
    preview_df = df.iloc[:, :20] if df.shape[1] > 20 else df
    lines.append(preview_df.head(3).to_string(index=False))
    return "\n".join(lines)


def classify_failure(error_msg: str, df: Optional[pd.DataFrame] = None) -> Tuple[str, str]:
    """
    Classifies execution and translation failures into structured categories
    with intelligent fuzzy column-name matching (difflib).
    """
    err_lower = error_msg.lower()

    if "security restriction" in err_lower or "prohibited" in err_lower:
        return (
            "Security Blocked",
            f"The query was blocked by the Zero-Trust Sandbox policy.\n\n"
            f"**Policy Notice:** Only safe in-memory data operations are permitted. Disk, OS, network, or dunder access is forbidden.\n\n"
            f"*Details: `{error_msg}`*",
        )

    if "execution timeout" in err_lower or "timed out" in err_lower:
        return (
            "Execution Timeout",
            "The query exceeded the execution time limit (10 seconds).\n\n"
            "**Possible Causes:** An infinite loop, heavy cartesian product, or unindexed cross-join was triggered.\n\n"
            "💡 *Tip: Try filtering rows before joining or simplify the query.*",
        )

    if "keyerror" in err_lower or "not in index" in err_lower or "unknown column" in err_lower:
        col_match = re.search(
            r"""KeyError:\s*["'](?:\[\s*["'])?([^"']+)""",
            error_msg,
        )
        misspelled = col_match.group(1) if col_match else None

        fuzzy_suggestion = ""
        if misspelled and df is not None:
            matches = difflib.get_close_matches(misspelled, [str(c) for c in df.columns], n=3, cutoff=0.5)
            if matches:
                suggested_str = ", ".join([f"`{m}`" for m in matches])
                fuzzy_suggestion = f"\n\n🔍 **Did you mean:** {suggested_str}?"

        available_cols = ", ".join([f"`{c}`" for c in df.columns[:8]]) if df is not None else ""
        return (
            "Unknown Column",
            f"The query attempted to reference a column that does not exist in the dataset: `{misspelled or 'Unknown'}`.{fuzzy_suggestion}\n\n"
            f"**Available columns:** {available_cols}...",
        )

    if any(k in err_lower for k in ["api key", "quota", "rate limit", "401", "403", "429", "resource_exhausted"]):
        return (
            "Provider/API Error",
            f"An error occurred communicating with the AI service.\n\n"
            f"*Details: `{error_msg}`*\n\n"
            f"💡 *Tip: Check `.env` quota or failover backend status.*",
        )

    return (
        "Translation / Pandas Execution Error",
        f"The generated pandas code encountered an error during execution.\n\n"
        f"**Error Details:** `{error_msg}`\n\n"
        f"💡 *Tip: Try rephrasing your question or naming specific columns and aggregations.*",
    )


def generate_suggested_prompts(df: pd.DataFrame) -> List[str]:
    """Dynamically creates dataset-aware sample questions."""
    prompts = []
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()

    binary_cols = [c for c in numeric_cols if df[c].nunique() == 2]
    continuous_numeric = [c for c in numeric_cols if df[c].nunique() > 5]

    target_candidates = [c for c in binary_cols if any(k in c.lower() for k in ["liked", "target", "label", "class"])]
    target = target_candidates[0] if target_candidates else (binary_cols[0] if binary_cols else None)

    if target and continuous_numeric:
        feature = continuous_numeric[0]
        unique_vals = sorted(df[target].dropna().unique().tolist())
        prompts.append(f"Compare average '{feature}' for '{target}' == {unique_vals[-1]} vs {unique_vals[0]}")

    if continuous_numeric:
        prompts.append(f"Plot the distribution of '{continuous_numeric[0]}'")
        if len(continuous_numeric) >= 2:
            prompts.append(f"Show a scatter plot of '{continuous_numeric[0]}' vs '{continuous_numeric[1]}'")
            prompts.append("Plot a correlation heatmap of numeric features")
        prompts.append(f"Which 5 rows have the highest '{continuous_numeric[0]}'?")

    if categorical_cols:
        prompts.append(f"What are the top categories in '{categorical_cols[0]}'?")

    if not prompts:
        prompts = [
            "Show summary statistics for the dataset",
            "Are there any missing or duplicate values?",
            "Show the first 5 rows",
            "Show the count of rows and columns",
        ]

    return prompts[:4]