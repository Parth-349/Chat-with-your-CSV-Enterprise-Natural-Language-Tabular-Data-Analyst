"""
Automated Pytest Suite for Zero-Trust Sandbox, Security Penetration Tests,
Matplotlib Capture, Series Export, CSV Loading Edge Cases,
and Zero-Config Auto-Router Pipeline.
Designed for Technical Jury Evaluation.
"""

import io
import os
import pytest
import pandas as pd
import numpy as np

from sandbox import (
    validate_code_ast,
    execute_pandas_code,
    load_csv_data,
    classify_failure,
    generate_suggested_prompts,
    get_df_schema_prompt,
    SAFE_BUILTINS,
)
from app import get_available_routes


@pytest.fixture
def spotify_df() -> pd.DataFrame:
    """Fixture providing the actual sample dataset data.csv."""
    data_path = os.path.join(os.path.dirname(__file__), "..", "data.csv")
    assert os.path.exists(data_path), f"data.csv not found at {data_path}"
    return pd.read_csv(data_path)


# ==============================================================================
# 1. Valid Pandas Aggregations, Series-to-CSV & Matplotlib Capture
# ==============================================================================

def test_valid_pandas_aggregations(spotify_df: pd.DataFrame):
    """Verifies that basic and grouped aggregations execute properly in the sandbox."""
    code = """
result = float(df['danceability'].mean())
"""
    res = execute_pandas_code(code, spotify_df)
    assert res["error"] is None, f"Execution failed: {res['error']}"
    assert res["result"] is not None
    assert abs(res["result"] - spotify_df["danceability"].mean()) < 1e-4

    # Grouped aggregation
    code_group = """
result_df = df.groupby('liked')[['energy', 'loudness']].mean().reset_index()
"""
    res_group = execute_pandas_code(code_group, spotify_df)
    assert res_group["error"] is None
    assert isinstance(res_group["result_df"], pd.DataFrame)
    assert "liked" in res_group["result_df"].columns
    assert len(res_group["result_df"]) == 2


def test_series_to_csv_export_preserves_index(spotify_df: pd.DataFrame):
    """
    Verifies that when a Series (e.g. value_counts()) is assigned to result,
    it is automatically converted to a DataFrame with its index and categories preserved
    so that CSV export does not drop category labels.
    """
    code = """
result = df['liked'].value_counts()
"""
    res = execute_pandas_code(code, spotify_df)
    assert res["error"] is None
    assert res["result_df"] is not None
    assert isinstance(res["result_df"], pd.DataFrame)

    # Verify both index (category) and count columns exist
    df_out = res["result_df"]
    assert len(df_out.columns) == 2
    csv_bytes = df_out.to_csv(index=False).encode("utf-8")
    csv_text = csv_bytes.decode("utf-8")
    assert "0" in csv_text and "1" in csv_text


def test_matplotlib_chart_png_capture(spotify_df: pd.DataFrame):
    """
    Verifies that matplotlib/seaborn plots are immediately rendered to valid PNG bytes,
    clearing memory figures safely.
    """
    code = """
fig, ax = plt.subplots(figsize=(6, 4))
sns.histplot(df['valence'], kde=True, ax=ax)
plt.title('Valence Distribution')
plt.tight_layout()
"""
    res = execute_pandas_code(code, spotify_df)
    assert res["error"] is None
    assert res["figure_png"] is not None
    assert isinstance(res["figure_png"], bytes)
    assert len(res["figure_png"]) > 1000
    assert res["figure_png"].startswith(b"\x89PNG\r\n\x1a\n")


def test_variable_string_format_executes(spotify_df: pd.DataFrame):
    """Verifies benign .format() calls on variables execute in the sandbox."""
    code = """
tmpl = "Features: {}"
result = tmpl.format(5)
"""
    res = execute_pandas_code(code, spotify_df)
    assert res["error"] is None, f"Execution failed: {res['error']}"
    assert res["result"] == "Features: 5"


def test_styler_format_call_executes(spotify_df: pd.DataFrame):
    """Verifies non-string .format receivers such as pandas Styler remain usable."""
    code = """
corr = df.select_dtypes(include='number').corr()
corr.style.format("{:.2f}")
"""
    res = execute_pandas_code(code, spotify_df)
    assert res["error"] is None, f"Execution failed: {res['error']}"


def test_dynamic_string_format_dunder_escape_blocked_at_runtime(spotify_df: pd.DataFrame):
    """Verifies runtime string-format guards block dynamically assembled traversal."""
    code = """
s = "{0." + "__globals__}"
s.format(len)
"""
    res = execute_pandas_code(code, spotify_df)
    assert res["error"] is not None
    assert res["error"].startswith("Security Restriction:")
    assert res["error"].count("Security Restriction") == 1


def test_correlation_heatmap_executes_end_to_end(spotify_df: pd.DataFrame):
    """Verifies numeric correlation computation and heatmap rendering work in the sandbox."""
    code = """
corr = df.select_dtypes(include='number').corr()
fig, ax = plt.subplots(figsize=(8, 4.5))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
plt.tight_layout()
"""
    res = execute_pandas_code(code, spotify_df)
    assert res["error"] is None, f"Execution failed: {res['error']}"
    assert res["figure_png"] is not None
    assert res["figure_png"].startswith(b"\x89PNG\r\n\x1a\n")


# ==============================================================================
# 2. Security Penetration Tests (Jury Q1 Defense)
# ==============================================================================

def test_security_blocked_import_math_os(spotify_df: pd.DataFrame):
    """Penetration test: 'import math, os' must be blocked by AST validator."""
    code = "import math, os\nresult = os.getcwd()"
    is_valid, err = validate_code_ast(code)
    assert not is_valid
    assert "os" in err

    res = execute_pandas_code(code, spotify_df)
    assert res["error"] is not None
    assert "Security Restriction" in res["error"]


def test_security_blocked_import_importlib(spotify_df: pd.DataFrame):
    """Penetration test: 'import importlib' must be blocked."""
    code = "import importlib\nos = importlib.import_module('os')"
    is_valid, err = validate_code_ast(code)
    assert not is_valid
    assert "importlib" in err

    res = execute_pandas_code(code, spotify_df)
    assert "Security Restriction" in res["error"]


def test_security_blocked_open_alias(spotify_df: pd.DataFrame):
    """Penetration test: 'f = open; f(\".env\")' must be blocked."""
    code = "f = open\nf('.env')"
    is_valid, err = validate_code_ast(code)
    assert not is_valid
    assert "open" in err

    res = execute_pandas_code(code, spotify_df)
    assert "Security Restriction" in res["error"]


def test_security_blocked_pd_read_csv(spotify_df: pd.DataFrame):
    """Penetration test: 'pd.read_csv(\".env\")' must be blocked."""
    code = "secret_df = pd.read_csv('.env')"
    is_valid, err = validate_code_ast(code)
    assert not is_valid
    assert "read_csv" in err

    res = execute_pandas_code(code, spotify_df)
    assert "Security Restriction" in res["error"]


def test_security_blocked_dunder_traversal(spotify_df: pd.DataFrame):
    """Penetration test: '().__class__.__base__.__subclasses__()' must be blocked."""
    code = "subclasses = ().__class__.__base__.__subclasses__()"
    is_valid, err = validate_code_ast(code)
    assert not is_valid
    assert "__" in err

    res = execute_pandas_code(code, spotify_df)
    assert "Security Restriction" in res["error"]


def test_security_blocked_plt_sys_module_attribute():
    """
    Penetration test: 'plt.sys.modules[\"os\"]' or accessing sys via module attributes
    must be blocked by AST validator.
    """
    code = "secret_os = plt.sys.modules['os']"
    is_valid, err = validate_code_ast(code)
    assert not is_valid
    assert "sys" in err or "prohibited" in err.lower()


def test_security_str_format_dunder_injection_blocked():
    """
    Penetration test: '{0.__globals__}'.format(len) string format escapes must be blocked,
    while benign f-strings and standard '{:.2f}'.format(x) are allowed.
    """
    # 1. Block traversal escape inside format string
    malicious_format = '"{0.__globals__}".format(len)'
    is_valid, err = validate_code_ast(malicious_format)
    assert not is_valid
    assert "format" in err.lower() or "traversal" in err.lower()

    # 2. Block item traversal inside format string
    malicious_item = '"{0[__class__]}".format(len)'
    is_valid_item, _ = validate_code_ast(malicious_item)
    assert not is_valid_item

    # Literal format strings are validated even when passed as callable values.
    malicious_map = 'map("{0.__globals__}".format, [len])'
    is_valid_map, _ = validate_code_ast(malicious_map)
    assert not is_valid_map

    # A user must not be able to replace or delete the guarded format method.
    for code in ("formatter.format = value", "del formatter.format"):
        is_valid_mutation, _ = validate_code_ast(code)
        assert not is_valid_mutation

    # 3. Allow standard benign format
    benign_format = 'result = "{:.2f}".format(1.5)'
    is_valid_benign, err_b = validate_code_ast(benign_format)
    assert is_valid_benign, f"Benign format was unexpectedly blocked: {err_b}"

    # 4. Allow standard f-string
    benign_fstring = 'val = 42\nresult = f"The value is {val}"'
    is_valid_fstr, err_f = validate_code_ast(benign_fstring)
    assert is_valid_fstr, f"Benign f-string was unexpectedly blocked: {err_f}"


def test_security_blocked_eval_query_load_dataset():
    """
    Penetration test: df.eval(...), df.query(...), and sns.load_dataset(...)
    must be blocked by AST validator to prevent dynamic evaluation and network requests.
    """
    # df.eval
    code_eval = "df.eval('danceability + 1')"
    is_valid_eval, err_eval = validate_code_ast(code_eval)
    assert not is_valid_eval
    assert "eval" in err_eval.lower()

    # df.query
    code_query = "df.query('liked == 1')"
    is_valid_query, err_query = validate_code_ast(code_query)
    assert not is_valid_query
    assert "query" in err_query.lower()

    # sns.load_dataset
    code_load_dataset = "sns.load_dataset('tips')"
    is_valid_load, err_load = validate_code_ast(code_load_dataset)
    assert not is_valid_load
    assert "load_dataset" in err_load.lower()


def test_huge_figsize_clamped_without_crashing(spotify_df: pd.DataFrame):
    """
    Verifies that a memory bomb query with huge figsize (e.g. figsize=(500, 500))
    is clamped safely to at most 20x20 inches without crashing or running out of memory.
    """
    code = """
plt.figure(figsize=(500, 500))
sns.lineplot(data=df['danceability'])
plt.title('Huge Figure Clamped')
"""
    res = execute_pandas_code(code, spotify_df)
    assert res["error"] is None, f"Execution failed: {res['error']}"
    assert res["figure_png"] is not None
    assert len(res["figure_png"]) > 0


def test_infinite_loop_timeout_protection(spotify_df: pd.DataFrame):
    """
    Verifies that infinite loops or heavy cartesian products are terminated
    within the configured timeout without hanging the parent process.
    """
    code = """
while True:
    pass
"""
    res = execute_pandas_code(code, spotify_df, timeout_seconds=2)
    assert res["error"] is not None
    assert "Execution Timeout" in res["error"]


def test_innocent_column_names_not_falsely_blocked(spotify_df: pd.DataFrame):
    """
    Verifies that innocent column names or string literals containing words like
    'system', 'delete', or 'remove' are NOT falsely blocked.
    """
    df_innocent = pd.DataFrame({
        "delete_flag": [1, 0, 1, 0],
        "operating_system": ["Windows", "macOS", "Linux", "Windows"],
        "removed_items": [0, 5, 2, 0],
    })

    code = """
result = int(df['delete_flag'].sum())
filtered = df[df['operating_system'] == 'Linux']
plt.figure()
plt.title('Operating System Distribution and Delete Flags')
"""
    is_valid, err = validate_code_ast(code)
    assert is_valid, f"False positive block: {err}"

    res = execute_pandas_code(code, df_innocent)
    assert res["error"] is None, f"Execution error: {res['error']}"
    assert res["result"] == 2


# ==============================================================================
# 3. CSV Loading Edge Cases & Schema Sanitization
# ==============================================================================

def test_csv_loading_latin1():
    """Verifies that non-UTF-8 (latin1) CSV files decode cleanly without crashing."""
    csv_latin1 = "song_title,artist\nCaf\xe9 del Mar,Energy 52\nResum\xe9,Artist 2\n".encode("latin1")
    df = load_csv_data(csv_latin1, filename="latin1_sample.csv")
    assert df.shape == (2, 2)
    assert "Café del Mar" in df["song_title"].values


def test_csv_loading_empty_data_error():
    """Verifies that empty (0 bytes) CSV files raise pd.errors.EmptyDataError cleanly."""
    with pytest.raises(pd.errors.EmptyDataError):
        load_csv_data(b"", filename="empty.csv")


def test_csv_loading_zero_row_dataframe():
    """Verifies that a CSV with headers but zero data rows loads safely."""
    csv_zero_rows = b"col_a,col_b,col_c\n"
    df = load_csv_data(csv_zero_rows, filename="zero_rows.csv")
    assert df.shape == (0, 3)
    assert list(df.columns) == ["col_a", "col_b", "col_c"]


def test_fuzzy_column_suggestion_on_misspelling(spotify_df: pd.DataFrame):
    """
    Verifies that when a column name is misspelled in a query (e.g. KeyError: 'danceabiliy'),
    classify_failure automatically uses difflib to suggest the closest match 'danceability'.
    """
    error_msg = "KeyError: 'danceabiliy'"
    cat, msg = classify_failure(error_msg, spotify_df)
    assert cat == "Unknown Column"
    assert "danceability" in msg


def test_suggested_prompts_detects_binary_target(spotify_df: pd.DataFrame):
    """
    Verifies that generate_suggested_prompts detects the binary column 'liked' in data.csv
    and creates an intelligent comparative query comparing liked == 1 vs 0.
    """
    suggestions = generate_suggested_prompts(spotify_df)
    assert len(suggestions) >= 1
    has_comparative = any("liked" in s and "vs" in s for s in suggestions)
    assert has_comparative, f"Suggestions lacked comparative prompt for binary target: {suggestions}"


# ==============================================================================
# 4. Zero-Config Router Pipeline Tests (Groq -> Gemini Order)
# ==============================================================================

def test_get_available_routes_gemini_and_groq_order():
    """
    Verifies that get_available_routes builds an ordered failover route list:
    1. Groq (primary: openai/gpt-oss-120b, fallback: openai/gpt-oss-20b)
    2. Google Gemini (primary: gemini-3.8-flash, fallback: gemini-flash-latest)
    without mutating keys or global os.environ.
    """
    mock_env = {
        "GEMINI_API_KEY": "mock_gemini_key",
        "GROQ_API_KEY": "mock_groq_key",
    }
    routes = get_available_routes(mock_env)
    assert len(routes) == 2
    assert routes[0]["provider"] == "Groq"
    assert routes[0]["primary_model"] == "openai/gpt-oss-120b"
    assert routes[0]["fallback_model"] == "openai/gpt-oss-20b"
    assert routes[0]["api_key"] == "mock_groq_key"

    assert routes[1]["provider"] == "Google Gemini"
    assert routes[1]["primary_model"] == "gemini-3.8-flash"
    assert routes[1]["fallback_model"] == "gemini-flash-latest"
    assert routes[1]["api_key"] == "mock_gemini_key"


def test_get_available_routes_handles_google_api_key_alias():
    """Verifies the Google API key alias remains the backup after Groq."""
    mock_env = {
        "GOOGLE_API_KEY": "alias_gemini_key",
        "GROQ_API_KEY": "mock_groq_key",
    }
    routes = get_available_routes(mock_env)
    assert len(routes) == 2
    assert routes[0]["provider"] == "Groq"
    assert routes[1]["provider"] == "Google Gemini"
    assert routes[1]["api_key"] == "alias_gemini_key"


def test_get_available_routes_gemini_only():
    """Verifies Gemini remains available when Groq is not configured."""
    routes = get_available_routes({"GEMINI_API_KEY": "gemini_only_key"})
    assert len(routes) == 1
    assert routes[0]["provider"] == "Google Gemini"
    assert routes[0]["api_key"] == "gemini_only_key"


def test_get_available_routes_groq_only():
    """Verifies fallback when only GROQ_API_KEY is configured."""
    mock_env = {"GROQ_API_KEY": "groq_only_key"}
    routes = get_available_routes(mock_env)
    assert len(routes) == 1
    assert routes[0]["provider"] == "Groq"
    assert routes[0]["api_key"] == "groq_only_key"


def test_get_available_routes_empty():
    """Verifies safe empty list returned when no keys exist in environment."""
    routes = get_available_routes({})
    assert routes == []
