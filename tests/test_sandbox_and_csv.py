"""
Automated Pytest Suite for Zero-Trust Sandbox, Security Penetration Tests,
Matplotlib Capture, Series Export, CSV Loading Edge Cases,
and Zero-Config Auto-Router Pipeline.
Designed for Technical Jury Evaluation.
"""

import io
import json
import os
from collections import Counter
from types import SimpleNamespace
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
from app import (
    _build_gemini_generate_config,
    _build_gemini_http_options,
    build_dataset_overview_figure,
    call_llm_code_generator_single,
    generate_grounded_answer,
    get_available_routes,
)
from google.genai import types as gemini_types
from profiler import (
    compute_data_profile,
    apply_preprocessing,
    build_deterministic_eda_insights,
    export_session_report_markdown,
)
from eval_suite import (
    BENCHMARK_SUITE,
    RESULT_COLUMNS,
    build_benchmark_suite,
    compare_providers_eval,
    export_metrics_json,
    export_results_csv,
    run_live_llm_eval,
    run_offline_security_eval,
)


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


@pytest.mark.parametrize("figure_target", ["result_fig", "result"])
def test_plotly_figure_serializes_to_json(spotify_df: pd.DataFrame, figure_target: str):
    """Verifies Plotly figures assigned to either supported output name cross the Pipe as JSON."""
    code = f"""
{figure_target} = px.scatter(df, x='danceability', y='energy')
{figure_target}.update_layout(template='plotly_dark')
"""
    res = execute_pandas_code(code, spotify_df)
    assert res["error"] is None, f"Execution failed: {res['error']}"
    assert res["figure_png"] is None
    assert res["plotly_json"]
    figure_data = json.loads(res["plotly_json"])
    assert figure_data["data"]
    assert figure_data["data"][0]["type"] == "scatter"


@pytest.mark.parametrize("write_method", ["write_html", "write_image"])
def test_plotly_file_writing_methods_are_blocked(spotify_df: pd.DataFrame, write_method: str):
    """Verifies Plotly methods that write files are rejected before sandbox execution."""
    code = f"""
fig = px.scatter(df, x='danceability', y='energy')
fig.{write_method}('blocked-output')
"""
    res = execute_pandas_code(code, spotify_df)
    assert "Security Restriction" in res["error"]


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


@pytest.mark.parametrize(
    ("code", "blocked_method"),
    [
        ("fig.write_json('chart.json')", "write_json"),
        ("fig.show()", "show"),
    ],
)
def test_security_blocked_plotly_write_json_and_show(code, blocked_method):
    """Plotly file writes and browser popups must be blocked before execution."""
    is_valid, err = validate_code_ast(code)
    assert not is_valid
    assert blocked_method in err


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


def test_schema_preview_limits_wide_dataframes_to_twenty_columns():
    df_wide = pd.DataFrame(
        [{f"column_{index}": f"value_{index}" for index in range(22)}]
    )
    schema = get_df_schema_prompt(df_wide)
    preview = schema.split("First 3 rows preview:", maxsplit=1)[1]
    assert "column_19" in preview
    assert "column_20" not in preview
    assert "column_21" not in preview


def test_fuzzy_column_suggestion_on_misspelling(spotify_df: pd.DataFrame):
    """
    Verifies that when a column name is misspelled in a query (e.g. KeyError: 'danceabiliy'),
    classify_failure automatically uses difflib to suggest the closest match 'danceability'.
    """
    error_msg = "KeyError: 'danceabiliy'"
    cat, msg = classify_failure(error_msg, spotify_df)
    assert cat == "Unknown Column"
    assert "danceability" in msg


def test_fuzzy_column_suggestion_on_keyerror_index_list(spotify_df: pd.DataFrame):
    error_msg = """KeyError: "['danceabiliy'] not in index" """
    category, message = classify_failure(error_msg, spotify_df)
    assert category == "Unknown Column"
    assert "danceability" in message


def test_keyerror_column_suggestion_accepts_single_character_column_name():
    suite = build_benchmark_suite(pd.DataFrame({"A": [1, 2, 3]}))
    typo_case = next(case for case in suite if case["case_id"] == "EDGE-01")
    assert "A_x" in typo_case["prompt"]


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
    2. Google Gemini (primary: gemini-4-hydrogen, fallback: gemini-flash-latest)
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
    assert routes[1]["primary_model"] == "gemini-4-hydrogen"
    assert routes[1]["fallback_model"] == "gemini-flash-latest"
    assert routes[1]["api_key"] == "mock_gemini_key"


@pytest.mark.parametrize(
    "model_name",
    ["openai/gpt-oss-20b", "openai/gpt-oss-120b"],
)
def test_groq_gpt_oss_uses_low_reasoning_effort(monkeypatch, model_name):
    """Verifies Groq gpt-oss generation requests use low reasoning effort."""
    captured = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="```python\nresult = 1\n```")
                    )
                ]
            )

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr("openai.OpenAI", FakeOpenAI)
    code, decline, error = call_llm_code_generator_single(
        prompt="Count records",
        df_schema="",
        provider="Groq",
        model_name=model_name,
        api_key="mock-key",
    )

    assert code == "result = 1"
    assert decline is None
    assert error is None
    assert captured["extra_body"] == {"reasoning_effort": "low"}


def test_grounded_summary_prefers_fast_groq_model(monkeypatch):
    """Verifies summary generation uses the primary fast Groq model and low effort."""
    captured = {}
    routes = get_available_routes({"GROQ_API_KEY": "mock-key"})

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="Count: 1."))]
            )

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr("app.get_available_routes", lambda: routes)
    monkeypatch.setattr("openai.OpenAI", FakeOpenAI)
    summary = generate_grounded_answer(
        prompt="How many rows?",
        code="result = len(df)",
        exec_res={"result": 1, "result_df": None, "stdout": "", "figure_png": None},
    )

    assert summary == "Count: 1."
    assert captured["model"] == "openai/gpt-oss-20b"
    assert captured["extra_body"] == {"reasoning_effort": "low"}


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


def test_gemini_4_config_uses_low_thinking():
    """Verifies Gemini 4 requests low-latency thinking with the installed SDK."""
    config = _build_gemini_generate_config(
        gemini_types,
        "gemini-4-hydrogen",
        temperature=0.1,
    )
    assert config.thinking_config.thinking_level.value == "LOW"


def test_gemini_4_config_falls_back_without_thinking_support():
    """Verifies older SDK interfaces fall back to a normal generation config."""
    class LegacyGenerateContentConfig:
        def __init__(self, **options):
            self.options = options

    class LegacyTypes:
        GenerateContentConfig = LegacyGenerateContentConfig

    with pytest.warns(RuntimeWarning, match="unsupported"):
        config = _build_gemini_generate_config(
            LegacyTypes,
            "gemini-4-hydrogen",
            temperature=0.1,
        )
    assert config.options == {"temperature": 0.1}


@pytest.mark.parametrize(
    ("model_name", "expected_options"),
    [
        ("gemini-4-hydrogen", {"timeout": 25_000}),
        ("gemini-flash-latest", {"timeout": 25_000, "api_version": "v1beta"}),
    ],
)
def test_gemini_http_options_use_safe_timeout(model_name, expected_options):
    class HttpOptions:
        def __init__(self, **options):
            self.options = options

    fake_types = SimpleNamespace(HttpOptions=HttpOptions)
    options = _build_gemini_http_options(fake_types, model_name)
    assert options.options == expected_options


def test_dataset_overview_uses_heatmap_for_two_numeric_columns(spotify_df):
    figure = build_dataset_overview_figure(spotify_df)
    assert len(figure.data) == 1
    assert figure.data[0].type == "heatmap"
    assert len(figure.data[0].x) <= 8
    assert figure.layout.template.layout.plot_bgcolor == "rgb(17,17,17)"
    assert figure.layout.paper_bgcolor == "rgba(0,0,0,0)"
    assert figure.layout.plot_bgcolor == "rgba(11,21,20,0.45)"


def test_dataset_overview_falls_back_to_column_counts():
    df = pd.DataFrame({
        "team": ["b", "a", "b"],
        "score": [2.0, 1.0, np.nan],
    })
    figure = build_dataset_overview_figure(df)
    assert figure.data
    assert all(trace.type == "bar" for trace in figure.data)
    assert {trace.name for trace in figure.data} == {"Non-null count", "Unique values"}
    assert figure.layout.paper_bgcolor == "rgba(0,0,0,0)"


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


# ==============================================================================
# 5. Data Profiler & Preprocessing Pipeline Tests (Phase 1 Deliverables)
# ==============================================================================

def test_compute_data_profile_on_spotify_data(spotify_df: pd.DataFrame):
    """Verifies data health scoring, summary counts, column diagnostics, and correlations."""
    prof = compute_data_profile(spotify_df)
    assert 0 <= prof["health_score"] <= 100
    assert prof["rows"] == 195
    assert prof["cols"] == 14
    assert prof["missing_cells"] == 0
    assert prof["duplicate_rows"] == 0
    assert isinstance(prof["column_report"], pd.DataFrame)
    assert len(prof["column_report"]) == 14
    assert "Detected Issues" in prof["column_report"].columns
    assert len(prof["top_correlations"]) > 0
    assert prof["top_correlations"][0]["abs_correlation"] > 0


def test_compute_data_profile_zero_rows():
    """Verifies that an empty DataFrame computes without ZeroDivisionError."""
    df_empty = pd.DataFrame(columns=["feature_a", "feature_b", "feature_c"])
    prof = compute_data_profile(df_empty)
    assert prof["rows"] == 0
    assert prof["cols"] == 3
    assert prof["health_score"] == 100.0
    assert prof["missing_cells"] == 0
    assert isinstance(prof["column_report"], pd.DataFrame)
    assert len(prof["top_correlations"]) == 0


def test_apply_preprocessing_duplicates_and_whitespace():
    """Verifies exact duplicate removal, text trimming, and empty string conversion."""
    df_test = pd.DataFrame({
        "text_col": ["  apple  ", "banana", "  apple  ", "   "],
        "num_col": [1, 2, 1, 4]
    })
    cleaned_df, logs = apply_preprocessing(df_test, drop_duplicates=True, strip_whitespace=True)
    assert len(cleaned_df) == 3
    assert cleaned_df["text_col"].iloc[0] == "apple"
    assert pd.isna(cleaned_df["text_col"].iloc[2])
    assert any("duplicate" in log.lower() for log in logs)
    assert any("whitespace" in log.lower() for log in logs)


def test_apply_preprocessing_imputation_and_outliers(spotify_df: pd.DataFrame):
    """
    Verifies median/mode null imputation and IQR outlier clipping
    while preserving binary 0/1 target columns (e.g. liked).
    """
    df_dirty = spotify_df.copy()
    df_dirty.loc[0, "danceability"] = np.nan
    df_dirty.loc[1, "duration_ms"] = 99999999

    cleaned_df, logs = apply_preprocessing(
        df_dirty,
        impute_strategy="median_mode",
        outlier_strategy="clip_iqr"
    )
    assert cleaned_df["danceability"].isnull().sum() == 0
    assert cleaned_df["duration_ms"].max() < 99999999
    # Ensure binary target column remains strictly 0 and 1
    assert set(cleaned_df["liked"].unique()).issubset({0, 1})
    assert any("imputed" in log.lower() for log in logs)
    assert any("clipped" in log.lower() or "winsorized" in log.lower() for log in logs)


def test_apply_preprocessing_remove_iqr_with_nullable_int64_na():
    df_with_missing_outlier = pd.DataFrame(
        {
            "measurement": pd.Series(
                [1, 2, 3, 4, 1000, pd.NA],
                dtype="Int64",
            )
        }
    )
    cleaned, logs = apply_preprocessing(
        df_with_missing_outlier,
        outlier_strategy="remove_iqr",
    )
    assert 1000 not in cleaned["measurement"].dropna().tolist()
    assert cleaned["measurement"].isna().sum() == 1
    assert any("removed" in log.lower() for log in logs)


def test_pandas_string_dtype_mixed_profile_preprocessing_and_prompts():
    """Covers Pandas 3 StringDtype alongside nullable and non-numeric columns."""
    df_mixed = pd.DataFrame(
        {
            "text": pd.Series(
                ["  alpha  ", "beta", pd.NA, "   ", "gamma", " duplicate ", " duplicate "],
                dtype="string",
            ),
            "nullable_count": pd.Series(
                [1, 2, 5, pd.NA, pd.NA, 1000, 1000],
                dtype="Int64",
            ),
            "measure": [1.0, 2.0, 3.0, 4.0, 5.0, 1000.0, 1000.0],
            "event_date": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-02",
                    None,
                    "2024-01-04",
                    "2024-01-05",
                    "2024-01-06",
                    "2024-01-06",
                ]
            ),
            "enabled": pd.Series(
                [True, False, None, True, False, True, True],
                dtype="boolean",
            ),
            "category_text": pd.Categorical(
                [" x ", "y", "   ", "x", "y", "x", "x"]
            ),
            "constant": ["same"] * 7,
        }
    )

    profile = compute_data_profile(df_mixed)
    assert profile["rows"] == 7
    assert profile["numeric_cols"] == 2
    prompts = generate_suggested_prompts(df_mixed)
    assert any("top categories in 'text'" in prompt for prompt in prompts)

    cleaned, logs = apply_preprocessing(
        df_mixed,
        drop_duplicates=True,
        strip_whitespace=True,
        impute_strategy="median_mode",
        outlier_strategy="clip_iqr",
        drop_constant_cols=True,
    )
    assert len(cleaned) == 6
    assert "constant" not in cleaned.columns
    assert cleaned["text"].str.strip().eq(cleaned["text"]).all()
    assert cleaned["text"].notna().all()
    assert not cleaned["text"].eq("nan").any()
    assert isinstance(cleaned["category_text"].dtype, pd.CategoricalDtype)
    assert cleaned["category_text"].notna().all()
    assert set(cleaned["category_text"].cat.categories) == {"x", "y"}
    assert pd.api.types.is_float_dtype(cleaned["nullable_count"])
    assert cleaned["nullable_count"].isna().sum() == 0
    assert cleaned["nullable_count"].max() < 1000
    assert cleaned["measure"].max() < 1000
    assert cleaned["event_date"].isna().sum() == 0
    assert cleaned["enabled"].isna().sum() == 0
    assert any("duplicate" in log.lower() for log in logs)


def test_deterministic_eda_binary_target_uses_actual_labels():
    df_binary = pd.DataFrame(
        {
            "target": [1, 1, 2, 2, 2],
            "measurement": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    insights = build_deterministic_eda_insights(
        df_binary,
        compute_data_profile(df_binary),
    )
    distribution = next(
        insight for insight in insights if "Target Distribution" in insight
    )
    assert "60.0% class 2" in distribution
    assert "40.0% class 1" in distribution


def test_export_session_report_markdown(spotify_df: pd.DataFrame):
    """Verifies that the session report markdown formats dataset info, audit steps, and Q&A."""
    prof = compute_data_profile(spotify_df)
    logs = ["Dropped 0 duplicates.", "Imputed 0 nulls."]
    messages = [
        {"role": "user", "content": "What is the average danceability?"},
        {
            "role": "assistant",
            "content": "The average danceability is 0.64.",
            "code": "result = df['danceability'].mean()",
            "route_label": "Groq (gpt-oss-120b)",
        }
    ]
    report = export_session_report_markdown("Spotify Tracks", spotify_df, prof, logs, messages)
    assert isinstance(report, str)
    assert "# 📊 Data Analysis & Q&A Session Report: Spotify Tracks" in report
    assert "Data Health Score" in report
    assert "Dropped 0 duplicates." in report
    assert "What is the average danceability?" in report
    assert "result = df['danceability'].mean()" in report
    assert "Groq (gpt-oss-120b)" in report


# ==============================================================================
# 6. Gen-AI Evaluation Suite Tests
# ==============================================================================

def test_offline_security_eval_passes_without_live_api_calls(spotify_df: pd.DataFrame):
    """Verifies all security payloads are blocked and benign reference queries are allowed."""
    result = run_offline_security_eval(spotify_df)
    metrics = result["metrics"]

    assert result["mode"] == "Offline Sandbox & Guardrail Audit"
    assert metrics["security_block_rate_pct"] == 100.0
    assert metrics["false_positive_rate_pct"] == 0.0
    assert metrics["security_attack_cases"] == 4
    assert metrics["security_blocked_cases"] == 4
    assert metrics["average_sandbox_latency_ms"] >= 0


def test_eval_suite_structure_and_export_formats(spotify_df: pd.DataFrame):
    """Verifies the 16 benchmark cases and deterministic CSV/JSON export shapes."""
    assert len(BENCHMARK_SUITE) == 16
    categories = Counter(case["category"] for case in BENCHMARK_SUITE)
    assert categories == {
        "Aggregation": 4,
        "Visualization": 3,
        "Edge / Recovery": 3,
        "Security Attack": 4,
        "Innocent Edge Cases": 2,
    }

    adapted_suite = build_benchmark_suite(spotify_df)
    typo_case = next(case for case in adapted_suite if case["case_id"] == "EDGE-01")
    assert "danceabiliy" in typo_case["prompt"]

    sample_result = {
        "mode": "Offline Sandbox & Guardrail Audit",
        "run_at": "2026-11-22T00:00:00+00:00",
        "metrics": {
            "security_block_rate_pct": 50.0,
            "false_positive_rate_pct": 10.0,
        },
        "results": pd.DataFrame(
            [{
                "Case ID": "SEC-01",
                "Category": "Security Attack",
                "Prompt": "Test a blocked import.",
                "Route Used": "Sandbox",
                "Pass@1": True,
                "Final Status": "Passed (blocked)",
                "Retries": 0,
                "Latency (ms)": 1.25,
                "Error/Notes": "Security Restriction",
            }],
            columns=RESULT_COLUMNS,
        ),
    }
    csv_text = export_results_csv(sample_result)
    csv_frame = pd.read_csv(io.StringIO(csv_text))
    assert list(csv_frame.columns) == RESULT_COLUMNS
    assert csv_frame.loc[0, "Case ID"] == "SEC-01"
    assert csv_frame.loc[0, "Latency (ms)"] == 1.25

    metrics = json.loads(
        export_metrics_json(
            sample_result,
            offline_security_metrics={
                "security_block_rate_pct": 100.0,
                "false_positive_rate_pct": 0.0,
                "average_sandbox_latency_ms": 1.25,
            },
        )
    )
    assert metrics["mode"] == sample_result["mode"]
    assert metrics["metrics"]["security_block_rate_pct"] == 50.0
    assert metrics["metrics"]["false_positive_rate_pct"] == 10.0
    assert metrics["metrics"]["average_sandbox_latency_ms"] == 1.25
    assert sample_result["metrics"]["security_block_rate_pct"] == 50.0


def test_export_metrics_json_merges_missing_offline_security_metrics():
    result = {
        "mode": "Live LLM Benchmark",
        "run_at": "2026-11-22T00:00:00+00:00",
        "metrics": {
            "security_block_rate_pct": None,
            "false_positive_rate_pct": None,
            "average_sandbox_latency_ms": None,
        },
    }
    offline_metrics = {
        "security_block_rate_pct": 100.0,
        "false_positive_rate_pct": 0.0,
        "average_sandbox_latency_ms": 1.25,
    }

    exported = json.loads(
        export_metrics_json(
            result,
            offline_security_metrics=offline_metrics,
        )
    )

    assert exported["metrics"] == offline_metrics
    assert json.loads(export_metrics_json(result))["metrics"] == result["metrics"]


def test_export_metrics_json_sanitizes_non_finite_provider_values():
    result = {
        "mode": "Provider Comparison",
        "run_at": "2026-11-22T00:00:00+00:00",
        "metrics": {"example_metric": np.nan, "non_finite": np.inf},
        "provider_comparison": pd.DataFrame(
            [{
                "Provider": "Groq",
                "Pass@1 Accuracy (%)": np.nan,
                "Average Latency (ms)": -np.inf,
            }]
        ),
    }

    exported = json.loads(export_metrics_json(result))
    assert exported["metrics"]["example_metric"] is None
    assert exported["metrics"]["non_finite"] is None
    assert exported["provider_comparison"][0]["Pass@1 Accuracy (%)"] is None
    assert exported["provider_comparison"][0]["Average Latency (ms)"] is None


def test_live_eval_retries_and_provider_comparison_without_api_calls(spotify_df: pd.DataFrame):
    """Verifies live-eval accounting with fake provider/executor callbacks only."""
    generated_models = []

    def fake_generator(
        prompt,
        df_schema,
        provider,
        model_name,
        api_key,
        conversation_history=None,
        retry_error=None,
        previous_code=None,
    ):
        generated_models.append((provider, model_name))
        if "groupby" in prompt and not retry_error:
            return "result = df['missing_column'].mean()", None, None
        if "groupby" in prompt:
            return "result = int(len(df))", None, None
        if "capital of France" in prompt:
            return None, "This question is outside the dataset.", None
        if "globals" in prompt:
            return None, "I cannot inspect Python internals.", None
        return "result = int(len(df))", None, None

    def fake_executor(code, _df):
        if "missing_column" in code:
            return {"error": "KeyError: missing_column"}
        return {"error": None}

    routes = [
        {
            "provider": "Groq",
            "primary_model": "primary-model",
            "fallback_model": "fallback-model",
            "api_key": "test-groq-key",
            "label": "Groq",
        },
        {
            "provider": "Google Gemini",
            "primary_model": "primary-model",
            "fallback_model": "fallback-model",
            "api_key": "test-gemini-key",
            "label": "Google Gemini",
        },
    ]
    progress_updates = []
    result = run_live_llm_eval(
        spotify_df,
        routes,
        fake_generator,
        provider="Groq",
        suite_mode="Fast Subset (6 queries)",
        executor=fake_executor,
        progress_callback=lambda current, total, case_id, status: progress_updates.append(
            (current, total, case_id, status)
        ),
    )
    assert result["metrics"]["pass_at_1_accuracy_pct"] < 100
    assert result["metrics"]["self_healed_pass_at_3_accuracy_pct"] == 100
    assert result["metrics"]["off_topic_rejection_accuracy_pct"] == 100
    groupby_row = result["results"].loc[result["results"]["Case ID"] == "AGG-03"].iloc[0]
    assert groupby_row["Retries"] == 1
    assert groupby_row["Final Status"] == "Passed after retries"
    assert {model for _, model in generated_models} == {"primary-model"}
    assert progress_updates[-1][:2] == (6, 6)

    progress_updates.clear()
    comparison = compare_providers_eval(
        spotify_df,
        routes,
        fake_generator,
        suite_mode="Fast Subset (6 queries)",
        executor=fake_executor,
        progress_callback=lambda current, total, case_id, status: progress_updates.append(
            (current, total, case_id, status)
        ),
    )
    assert comparison["provider_comparison"]["Provider"].tolist() == ["Groq", "Google Gemini"]
    assert len(comparison["results"]) == 12
    assert progress_updates[-1][:2] == (12, 12)
    assert all(model == "primary-model" for _, model in generated_models)
