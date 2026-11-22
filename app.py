"""
Chat-with-your-CSV: Enterprise Natural-Language Tabular Data Analyst
Upgraded with:
- Zero-Config .env Auto-Router with Ordered Automatic Failover (Groq -> Gemini)
- Modern Obsidian-Teal & Champagne Glassmorphism UI
- Multi-Tab Workspace: 💬 AI Data Chat + 🧹 Data Profiler & Preprocessing + 🧪 Evaluation
- Automated Data Health Scoring (0-100) & Column Diagnostics
- Reversible Interactive Preprocessing Pipeline with Audit Trail
- Session Report (.md) Export
- Zero-Trust AST + Allowlist Execution Sandbox (via sandbox.py)
- Streamlit width compatibility helper (Streamlit >= 1.50 width='stretch')
- Post-Execution Grounded Summarization (quoting verified computed numbers)
- Dynamic Fuzzy Column Matching (difflib) & Structured Decline Handling
"""

import html
import io
import os
import re
import sys
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend safe for server/Streamlit usage
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import seaborn as sns
import streamlit as st
from dotenv import load_dotenv

# Load environment variables without overwriting existing runtime variables
load_dotenv(override=False)

# Import Sandbox and Tabular utilities
from sandbox import (
    ALLOWED_MODULES,
    PROHIBITED_CALLS,
    PROHIBITED_METHODS,
    SAFE_BUILTINS,
    classify_failure,
    execute_pandas_code,
    generate_suggested_prompts,
    get_df_schema_prompt,
    load_csv_data,
    sanitize_text,
    validate_code_ast,
)

# Import Profiler and Preprocessing utilities
from profiler import (
    apply_preprocessing,
    build_deterministic_eda_insights,
    compute_data_profile,
    export_session_report_markdown,
)
from eval_suite import (
    compare_providers_eval,
    export_metrics_json,
    export_results_csv,
    run_live_llm_eval,
    run_offline_security_eval,
)


# ==============================================================================
# 1. Streamlit Version Compatibility Helper
# ==============================================================================

def get_width_kwargs(use_stretch: bool = True) -> Dict[str, Any]:
    """
    Streamlit width compatibility helper:
    Uses width='stretch' on Streamlit >= 1.50 and use_container_width=True on older versions.
    Avoids deprecation warnings across all Streamlit versions.
    """
    try:
        import packaging.version
        if packaging.version.parse(st.__version__) >= packaging.version.parse("1.50.0"):
            return {"width": "stretch"} if use_stretch else {}
    except Exception:
        pass
    return {"use_container_width": True} if use_stretch else {}


# ==============================================================================
# 2. Cached Data Loading
# ==============================================================================

@st.cache_data(show_spinner=False)
def load_csv_data_cached(file_bytes: bytes, filename: str = "") -> pd.DataFrame:
    """Cached wrapper around load_csv_data using file bytes for deterministic hashing."""
    return load_csv_data(file_bytes, filename=filename)


# ==============================================================================
# 3. Zero-Config .env Auto-Backend Router
# ==============================================================================

def get_available_routes(env_dict: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
    """
    Inspects environment variables and constructs the prioritized list of LLM routes.
    Order of preference:
      1. Groq (primary: openai/gpt-oss-120b, fallback: openai/gpt-oss-20b)
      2. Google Gemini (primary: gemini-4-hydrogen, fallback: gemini-flash-latest)
    Does NOT mutate global os.environ.
    """
    env = env_dict if env_dict is not None else os.environ
    routes = []

    groq_key = (env.get("GROQ_API_KEY") or "").strip().strip("'\"")
    if groq_key:
        routes.append({
            "provider": "Groq",
            "primary_model": "openai/gpt-oss-120b",
            "fallback_model": "openai/gpt-oss-20b",
            "api_key": groq_key,
            "label": "Groq (gpt-oss-120b)",
        })

    gemini_key = (env.get("GEMINI_API_KEY") or env.get("GOOGLE_API_KEY") or "").strip().strip("'\"")
    if gemini_key:
        routes.append({
            "provider": "Google Gemini",
            "primary_model": "gemini-4-hydrogen",
            "fallback_model": "gemini-flash-latest",
            "api_key": gemini_key,
            "label": "Google Gemini (gemini-4-hydrogen)",
        })

    return routes


# ==============================================================================
# 4. LLM Code Generator with Failover & Decline Handling
# ==============================================================================

def _build_gemini_generate_config(
    types: Any,
    model_name: str,
    temperature: float,
    system_instruction: Optional[str] = None,
) -> Any:
    config_options: Dict[str, Any] = {"temperature": temperature}
    if system_instruction is not None:
        config_options["system_instruction"] = system_instruction

    if model_name.startswith("gemini-4-"):
        try:
            config_options["thinking_config"] = types.ThinkingConfig(thinking_level="LOW")
            return types.GenerateContentConfig(**config_options)
        except (AttributeError, TypeError, ValueError) as error:
            config_options.pop("thinking_config", None)
            warnings.warn(
                f"Gemini 4 LOW thinking is unsupported by this google-genai installation; "
                f"using the default thinking configuration ({error}).",
                RuntimeWarning,
                stacklevel=2,
            )

    return types.GenerateContentConfig(**config_options)


def _build_gemini_http_options(types: Any, model_name: str) -> Any:
    options: Dict[str, Any] = {"timeout": 25_000}
    if "-latest" in model_name:
        options["api_version"] = "v1beta"
    return types.HttpOptions(**options)


def call_llm_code_generator_single(
    prompt: str,
    df_schema: str,
    provider: str,
    model_name: str,
    api_key: str,
    conversation_history: Optional[List[str]] = None,
    retry_error: Optional[str] = None,
    previous_code: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Calls a single LLM model to generate Python code or structured decline.
    Passes API keys directly to client constructors without mutating os.environ.
    Returns: (code_str, decline_reason, error_str)
    """
    clean_key = api_key.strip().strip("'\"")
    clean_model = model_name.replace("models/", "")

    system_instruction = f"""You are an expert Python data analyst and Streamlit data scientist assistant.
The user has a pandas DataFrame loaded in variable `df`.

{df_schema}

CRITICAL ZERO-TRUST SECURITY POLICY:
You are strictly restricted to in-memory data analysis and visualization on the loaded DataFrame `df`.
Under no circumstances should you generate code that modifies, deletes, reads, or interacts with the host OS, file system, disk, environment variables, or network (e.g., os, sys, subprocess, shutil, open, pathlib, socket, etc.).
Dunder attribute or name access (such as __class__, __subclasses__, __builtins__, __globals__) is strictly blocked by our AST sandbox.
Never access runtime system modules or attributes (such as plt.sys or df.eval or df.query or sns.load_dataset).

SYNTAX & FORMATTING RULES:
1. Refer to the dataset exclusively as `df`. Do NOT redefine `df` or reload files. All libraries (pd, np, plt, sns, px, go, math, statistics, datetime) are already imported; import statements are unnecessary.
2. ALWAYS use bracket access `df['column']` for columns, never `.format()` or dynamic string evaluations. Use f-strings for string interpolation.
3. For text columns, use `df.select_dtypes(include=['object', 'string', 'category'])` so Pandas 3 StringDtype columns are included.
4. If computing a single value, list, or text, store it in variable `result` or print it.
5. If returning a subset, grouped table, or aggregated view, store it in variable `result_df` (a pandas DataFrame).
6. For bar, scatter, histogram, box, and heatmap requests, prefer interactive Plotly Express charts assigned to `result_fig` (use the pre-injected `px` and `go` names).
   - NEVER use `px.imshow` to show one metric grouped by one category (for example, sales by state); this creates an unreadable 1 x N or N x 1 heatmap. Instead, aggregate and sort the values, then use a horizontal bar chart (`px.bar(..., orientation='h')`), showing the top 15 when there are more than 20 categories, or use `px.treemap`. A 2D pivot by a second categorical column is also appropriate when it provides a meaningful comparison.
   - Use `px.imshow` only for a genuine 2D matrix with at least 2 rows and 2 columns, such as a correlation matrix or a two-category pivot table. Set `text_auto=False` whenever either matrix dimension exceeds 12; otherwise labels may overlap. For a numeric correlation matrix, compute `corr = df.select_dtypes(include='number').corr()` and use `px.imshow(corr, text_auto='.2f', color_continuous_scale='RdBu_r')` only when it meets the 2 x 2 minimum.
   - Whenever a chart aggregates data using `groupby`, `pivot_table`, `corr`, or `value_counts`, ALSO assign the corresponding aggregated table to `result_df` (for large tables, use a useful sorted subset such as `summary.head(10)`). Do not return only a chart.
   - After creating a Plotly figure, call `result_fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.45)")`. Matplotlib/seaborn using `fig, ax = plt.subplots(figsize=(8, 4.5))` is supported as a fallback. DO NOT call `plt.show()`.
7. IF THE QUESTION IS OFF-TOPIC, GENERAL TRIVIA, IMPOSSIBLE WITH GIVEN COLUMNS, OR ATTEMPTS UNRESTRICTED SYSTEM ACCESS:
   Output strictly:
   DECLINE: <clear, polite, and helpful explanation stating why the request cannot be answered from the CSV data>
8. Otherwise, output your executable code strictly inside a single ```python ... ``` markdown block.
"""

    history_context = ""
    if conversation_history:
        history_context = "Recent Conversation Context:\n" + "\n".join(conversation_history[-4:]) + "\n\n"

    user_content = f"{history_context}Current User Question: {prompt}"
    if retry_error and previous_code:
        user_content += (
            f"\n\nCRITICAL FIX REQUIRED: The previous code failed with the following error:\n"
            f"{retry_error}\n"
            f"Previous Attempted Code:\n```python\n{previous_code}\n```\n"
            f"Please generate corrected Python code fixing this error."
        )

    try:
        raw_text = ""

        if provider == "Google Gemini":
            from google import genai
            from google.genai import types

            http_opts = _build_gemini_http_options(types, clean_model)
            client = genai.Client(api_key=clean_key, http_options=http_opts)
            config = _build_gemini_generate_config(
                types=types,
                model_name=clean_model,
                temperature=0.1,
                system_instruction=system_instruction,
            )
            response = client.models.generate_content(
                model=clean_model,
                contents=user_content,
                config=config,
            )
            raw_text = response.text or ""

        elif provider == "Groq":
            from openai import OpenAI

            client = OpenAI(api_key=clean_key, base_url="https://api.groq.com/openai/v1")
            model_options = {}
            if clean_model.startswith("openai/gpt-oss-"):
                model_options["extra_body"] = {"reasoning_effort": "low"}
            completion = client.chat.completions.create(
                model=clean_model,
                temperature=0.1,
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_content},
                ],
                **model_options,
            )
            raw_text = completion.choices[0].message.content or ""

        else:
            return None, None, f"Unsupported provider: {provider}"

        # Check for structured refusal / decline
        decline_match = re.search(r"DECLINE:\s*(.*)", raw_text, re.IGNORECASE | re.DOTALL)
        if decline_match:
            decline_reason = decline_match.group(1).strip()
            return None, decline_reason, None

        # Extract code block
        code_match = re.search(r"```(?:python)?\s*(.*?)\s*```", raw_text, re.DOTALL)
        code = code_match.group(1).strip() if code_match else ""

        if not code and raw_text.strip():
            # If no code block found, treat whole text as decline/explanation
            return None, raw_text.strip(), None

        return code, None, None

    except Exception as e:
        err_msg = str(e)
        return None, None, f"{type(e).__name__}: {err_msg}"


def generate_executive_briefing(prompt: str) -> str:
    """Generate a natural-language briefing with provider/model failover."""
    routes = get_available_routes()
    if not routes:
        return "AI briefing unavailable: no API keys are configured."

    errors = []
    system_instruction = (
        "You are a principal data scientist. Provide a concise executive briefing "
        "grounded only in the supplied computed facts. Do not generate Python code "
        "or invent statistics."
    )
    for route in routes:
        models = [route.get("primary_model"), route.get("fallback_model")]
        for model in (candidate for candidate in models if candidate):
            try:
                if route["provider"] == "Groq":
                    from openai import OpenAI

                    client = OpenAI(
                        api_key=route["api_key"],
                        base_url="https://api.groq.com/openai/v1",
                    )
                    model_options = {}
                    if model.startswith("openai/gpt-oss-"):
                        model_options["extra_body"] = {"reasoning_effort": "low"}
                    completion = client.chat.completions.create(
                        model=model,
                        temperature=0.2,
                        max_tokens=500,
                        messages=[
                            {"role": "system", "content": system_instruction},
                            {"role": "user", "content": prompt},
                        ],
                        **model_options,
                    )
                    briefing = completion.choices[0].message.content or ""
                elif route["provider"] == "Google Gemini":
                    from google import genai
                    from google.genai import types

                    client = genai.Client(
                        api_key=route["api_key"],
                        http_options=_build_gemini_http_options(types, model),
                    )
                    response = client.models.generate_content(
                        model=model,
                        contents=prompt,
                        config=_build_gemini_generate_config(
                            types=types,
                            model_name=model,
                            temperature=0.2,
                            system_instruction=system_instruction,
                        ),
                    )
                    briefing = response.text or ""
                else:
                    errors.append(f"Unsupported provider: {route['provider']}")
                    continue

                if briefing.strip():
                    return briefing.strip()
                errors.append(f"{route['provider']} ({model}) returned an empty briefing")
            except Exception as ex:
                errors.append(f"{route['provider']} ({model}): {type(ex).__name__}: {ex}")

    return "AI briefing unavailable after provider failover. " + " | ".join(errors)


def execute_query_pipeline(
    prompt: str,
    df_schema: str,
    df: pd.DataFrame,
    conversation_history: Optional[List[str]] = None,
) -> Tuple[Optional[str], Optional[Dict[str, Any]], Optional[str], Optional[str], Optional[str], Dict[str, Any]]:
    """
    Executes the query using the ordered auto-failover pipeline:
    1. Groq (primary -> fallback model)
    2. Gemini (primary -> fallback model)
    Returns: (code, exec_res, decline_reason, active_route_label, error_message, telemetry)
    """
    telemetry: Dict[str, Any] = {
        "intent_status": f"Analyzed {len(df.columns)} columns",
        "route_used": "Not run",
        "llm_latency_ms": 0.0,
        "ast_status": "Not run",
        "sandbox_exec_time_ms": 0.0,
        "self_healing_retries": 0,
        "grounded_status": "Not run",
    }

    def execute_with_telemetry(candidate_code: str) -> Dict[str, Any]:
        ast_valid, ast_error = validate_code_ast(candidate_code)
        telemetry["ast_status"] = (
            "AST Validated ✓" if ast_valid else f"AST Rejected: {ast_error}"
        )
        started = time.perf_counter()
        result = execute_pandas_code(candidate_code, df)
        telemetry["sandbox_exec_time_ms"] += (time.perf_counter() - started) * 1000
        return result

    routes = get_available_routes()
    if not routes:
        return (
            None,
            None,
            None,
            None,
            "No active API keys found in `.env`. Please add `GROQ_API_KEY` or `GEMINI_API_KEY` to your `.env` file.",
            telemetry,
        )

    last_error = ""

    for route in routes:
        provider = route["provider"]
        api_key = route["api_key"]
        models_to_try = [route["primary_model"], route["fallback_model"]]

        for model in models_to_try:
            telemetry["route_used"] = f"{provider} ({model})"
            llm_started = time.perf_counter()
            code, decline, err = call_llm_code_generator_single(
                prompt=prompt,
                df_schema=df_schema,
                provider=provider,
                model_name=model,
                api_key=api_key,
                conversation_history=conversation_history,
            )
            telemetry["llm_latency_ms"] += (time.perf_counter() - llm_started) * 1000

            if err:
                last_error = f"[{provider} - {model}] {err}"
                continue  # Try fallback model or next route

            if decline:
                telemetry["intent_status"] = "DECLINE"
                telemetry["grounded_status"] = "Not run (request declined)"
                return None, None, decline, route["label"], None, telemetry

            if code:
                exec_res = execute_with_telemetry(code)

                # Self-healing retry loop (up to 2 retries for recoverable errors)
                attempts = 0
                max_retries = 2
                current_code = code

                while exec_res["error"] and attempts < max_retries:
                    err_lower = exec_res["error"].lower()
                    # Do not retry on security restrictions or timeouts
                    if "security restriction" in err_lower or "execution timeout" in err_lower:
                        break

                    attempts += 1
                    telemetry["self_healing_retries"] = attempts
                    llm_started = time.perf_counter()
                    fixed_code, fixed_decline, retry_err = call_llm_code_generator_single(
                        prompt=prompt,
                        df_schema=df_schema,
                        provider=provider,
                        model_name=model,
                        api_key=api_key,
                        conversation_history=conversation_history,
                        retry_error=exec_res["error"],
                        previous_code=current_code,
                    )
                    telemetry["llm_latency_ms"] += (time.perf_counter() - llm_started) * 1000

                    if fixed_decline:
                        telemetry["intent_status"] = "DECLINE during recovery"
                        telemetry["grounded_status"] = "Not run (request declined)"
                        return None, None, fixed_decline, route["label"], None, telemetry

                    if fixed_code and not retry_err:
                        current_code = fixed_code
                        exec_res = execute_with_telemetry(current_code)
                    else:
                        break

                telemetry["self_healing_retries"] = attempts
                return current_code, exec_res, None, route["label"], None, telemetry

    return (
        None,
        None,
        None,
        None,
        f"All AI provider routes failed. Last error: {last_error}",
        telemetry,
    )


# ==============================================================================
# 5. Post-Execution Grounded Answer Summarizer (Jury Q2 Defense)
# ==============================================================================

def generate_grounded_answer(
    prompt: str,
    code: str,
    exec_res: Dict[str, Any],
) -> str:
    """
    Generates a concise, grounded natural-language explanation based strictly
    on verified execution outputs (numbers, tables, charts) to eliminate pre-execution hallucination.
    """
    result_val = exec_res.get("result")
    result_df = exec_res.get("result_df")
    stdout = exec_res.get("stdout")
    has_chart = (
        exec_res.get("figure_png") is not None
        or exec_res.get("plotly_json") is not None
    )

    facts_parts = []
    if result_val is not None:
        facts_parts.append(f"Result Value: {result_val}")
    if result_df is not None:
        facts_parts.append(f"Result Table Preview ({len(result_df)} total rows):\n{result_df.head(5).to_string(index=False)}")
    if stdout:
        facts_parts.append(f"Console Output:\n{stdout}")
    if has_chart:
        facts_parts.append("Chart: Successfully rendered.")

    facts_summary = "\n".join(facts_parts) if facts_parts else "The query executed with no printed output."

    summary_prompt = f"""User Question: {prompt}
Executed Python Code:
```python
{code}
```

ACTUAL COMPUTED RESULTS:
{facts_summary}

Instructions:
Provide a concise, friendly, plain-English summary (1-3 sentences) directly answering the user's question.
Cite verified numeric findings from `result` and/or `result_df` in the ACTUAL COMPUTED RESULTS above.
For chart queries, explain the actual data insights (such as top categories, comparisons, or correlations) using the verified result table. Never summarize the Python implementation steps or describe "what the code does" instead of reporting data findings.
Do NOT guess, extrapolate, or invent any numbers not present in the results.
"""

    routes = get_available_routes()
    for route in routes:
        clean_key = route["api_key"]
        if route["provider"] == "Groq":
            models = [route.get("fallback_model"), route.get("primary_model")]
        else:
            models = [route.get("primary_model"), route.get("fallback_model")]
        for model in (candidate for candidate in models if candidate):
            try:
                if route["provider"] == "Groq":
                    from openai import OpenAI

                    client = OpenAI(api_key=clean_key, base_url="https://api.groq.com/openai/v1")
                    model_options = {}
                    if model.startswith("openai/gpt-oss-"):
                        model_options["extra_body"] = {"reasoning_effort": "low"}
                    completion = client.chat.completions.create(
                        model=model,
                        temperature=0.2,
                        messages=[{"role": "user", "content": summary_prompt}],
                        max_tokens=250,
                        **model_options,
                    )
                    grounded_text = completion.choices[0].message.content or ""
                    if grounded_text.strip():
                        return grounded_text.strip()

                elif route["provider"] == "Google Gemini":
                    from google import genai
                    from google.genai import types

                    client = genai.Client(
                        api_key=clean_key,
                        http_options=_build_gemini_http_options(types, model),
                    )
                    response = client.models.generate_content(
                        model=model,
                        contents=summary_prompt,
                        config=_build_gemini_generate_config(
                            types=types,
                            model_name=model,
                            temperature=0.2,
                        ),
                    )
                    grounded_text = response.text or ""
                    if grounded_text.strip():
                        return grounded_text.strip()
            except Exception:
                continue

    # Fast-path deterministic fallback if all summarizer LLM calls fail
    if result_val is not None:
        return f"Based on your dataset, the query computed: **{result_val}**."
    elif result_df is not None:
        return f"Here are the top results from your dataset ({len(result_df)} rows computed)."
    elif has_chart:
        return "The requested chart has been visualized below."
    elif stdout:
        return f"Query output:\n{stdout}"
    else:
        return "The query executed successfully on your dataset."


def render_pipeline_trace(telemetry: Dict[str, Any]) -> None:
    """Render the four query-processing stages in a compact chat expander."""
    with st.expander("🧠 Agent Pipeline Trace", expanded=False):
        st.markdown(f"**1. Intent & Schema Grounding:** {telemetry.get('intent_status', 'Not run')}")
        st.markdown(
            f"**2. Code Synthesis:** Route used: {telemetry.get('route_used', 'Not run')} · "
            f"LLM latency: {telemetry.get('llm_latency_ms', 0.0):.1f} ms"
        )
        st.markdown(
            f"**3. Zero-Trust AST Sandbox:** {telemetry.get('ast_status', 'Not run')} · "
            f"Sandbox exec time: {telemetry.get('sandbox_exec_time_ms', 0.0):.1f} ms · "
            f"Self-Healing Retries: {telemetry.get('self_healing_retries', 0)}"
        )
        st.markdown(
            f"**4. Grounded Numeric Summarizer:** "
            f"{telemetry.get('grounded_status', 'Not run')}"
        )


def render_plotly_chart(
    plotly_json: str,
    key_suffix: str,
    width_kwargs: Dict[str, Any],
) -> None:
    """Render a sandbox-serialized Plotly figure and offer a self-contained HTML download."""
    figure = pio.from_json(plotly_json)
    st.plotly_chart(figure, key=f"plotly_chart_{key_suffix}", **width_kwargs)
    st.download_button(
        label="🌐 Download Interactive Chart (HTML)",
        data=figure.to_html(full_html=True, include_plotlyjs=True),
        file_name="interactive_chart.html",
        mime="text/html",
        key=f"dl_html_{key_suffix}",
    )


def build_column_sparkline(df: pd.DataFrame) -> str:
    """Build a compact completeness sparkline using only inline SVG."""
    completeness = df.notna().mean().fillna(0).tolist() if len(df.columns) else []
    if not completeness:
        return '<svg viewBox="0 0 180 44" style="width:100%;height:auto;max-height:54px" role="img" aria-label="No columns"><text x="4" y="25" fill="#9aa9a5" font-size="12">No column data</text></svg>'

    width = max(180, min(440, len(completeness) * 12))
    step = (width - 12) / max(len(completeness) - 1, 1)
    dots = []
    for index, value in enumerate(completeness):
        x = 6 + index * step
        y = 38 - float(value) * 30
        color = "#34d399" if value >= 0.9 else "#e0bd79"
        dots.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.1" fill="{color}" />')
    return (
        f'<svg viewBox="0 0 {width} 44" style="width:100%;height:auto;max-height:54px" role="img" aria-label="Column completeness">'
        f'{"".join(dots)}</svg>'
    )


def build_dataset_overview_figure(df: pd.DataFrame) -> go.Figure:
    """Build a correlation heatmap or compact column-quality overview."""
    numeric_df = df.select_dtypes(include="number")
    if len(df) >= 2 and numeric_df.shape[1] >= 2:
        selected_columns = numeric_df.columns[:8]
        correlation = numeric_df.loc[:, selected_columns].corr()
        figure = px.imshow(
            correlation,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale=["#34d399", "#0b1514", "#e0bd79"],
            zmin=-1,
            zmax=1,
            labels={"x": "Feature", "y": "Feature", "color": "r"},
            title="Numeric Feature Correlations",
        )
        figure.update_xaxes(tickangle=-35, side="bottom", title=None)
        figure.update_yaxes(title=None)
        figure.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(11,21,20,0.45)",
            height=390,
            margin={"l": 55, "r": 24, "t": 58, "b": 100},
        )
        return figure

    column_counts = pd.DataFrame({
        "Column": [str(column) for column in df.columns],
        "Non-null count": [int(df[column].notna().sum()) for column in df.columns],
        "Unique values": [int(df[column].nunique(dropna=True)) for column in df.columns],
    })
    count_data = column_counts.melt(
        id_vars="Column",
        var_name="Measure",
        value_name="Count",
    )
    figure = px.bar(
        count_data,
        x="Count",
        y="Column",
        color="Measure",
        barmode="group",
        orientation="h",
        text_auto=True,
        color_discrete_map={
            "Non-null count": "#34d399",
            "Unique values": "#e0bd79",
        },
        title="Column Completeness & Cardinality",
    )
    figure.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(11,21,20,0.45)",
        height=max(300, min(520, 80 + len(column_counts) * 32)),
        margin={"l": 24, "r": 24, "t": 58, "b": 35},
        legend_title_text="",
    )
    figure.update_xaxes(title=None, gridcolor="rgba(167,243,208,0.10)")
    figure.update_yaxes(title=None, autorange="reversed")
    return figure


def _style_evaluation_figure(figure: go.Figure, height: int = 340) -> go.Figure:
    figure.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(11,21,20,0.45)",
        font={"color": "#f1f5f4"},
        height=height,
        margin={"l": 28, "r": 20, "t": 40, "b": 45},
        legend_title_text="",
    )
    figure.update_xaxes(showgrid=False)
    figure.update_yaxes(gridcolor="rgba(167,243,208,0.12)")
    return figure


# ==============================================================================
# 6. Streamlit User Interface (Translucent Glassmorphism & Tabs)
# ==============================================================================

def main():
    # --- Page Configuration ---
    st.set_page_config(
        page_title="Chat-with-your-CSV",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    width_kwargs = get_width_kwargs(use_stretch=True)

    # --- Modern Translucent Glassmorphism CSS ---
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

        :root {
            --obsidian: #050908;
            --glass: rgba(11, 21, 20, 0.72);
            --glass-border: rgba(167, 243, 208, 0.11);
            --champagne: #e0bd79;
            --mint: #34d399;
            --text-main: #f1f5f4;
            --text-muted: #9aa9a5;
        }

        .stApp {
            background-color: var(--obsidian);
            background-image:
                radial-gradient(circle at 12% 12%, rgba(16, 85, 72, 0.22) 0%, transparent 38%),
                radial-gradient(circle at 88% 88%, rgba(224, 189, 121, 0.08) 0%, transparent 36%),
                radial-gradient(circle at 52% 48%, rgba(20, 60, 55, 0.18) 0%, transparent 48%);
            background-attachment: fixed;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            color: var(--text-main);
        }
        [data-testid="stAppViewContainer"] {
            background: transparent;
        }
        [data-testid="stHeader"] {
            background: rgba(5, 9, 8, 0.78);
            backdrop-filter: blur(18px);
        }
        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2rem;
        }

        [data-testid="stSidebar"] {
            background: rgba(7, 16, 14, 0.94) !important;
            backdrop-filter: blur(22px) !important;
            -webkit-backdrop-filter: blur(22px) !important;
            border-right: 1px solid rgba(167, 243, 208, 0.11) !important;
        }

        .hero-container {
            padding: 0.85rem 1.15rem;
            border-radius: 18px;
            background: rgba(11, 21, 20, 0.82);
            backdrop-filter: blur(22px) saturate(140%);
            -webkit-backdrop-filter: blur(22px) saturate(140%);
            border: 1px solid rgba(167, 243, 208, 0.14);
            box-shadow: 0 18px 45px rgba(0,0,0,0.45), inset 0 1px 0 rgba(255,255,255,0.06);
            margin-bottom: 0.8rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
        }
        .hero-brand {
            display: flex;
            align-items: center;
            gap: 0.8rem;
        }
        .brand-mark {
            width: 42px;
            height: 42px;
            display: grid;
            place-items: center;
            border-radius: 14px;
            color: #fff4d8;
            font-size: 1.35rem;
            background: linear-gradient(145deg, rgba(224,189,121,0.28), rgba(52,211,153,0.14));
            border: 1px solid rgba(224,189,121,0.42);
            box-shadow: 0 0 22px rgba(224,189,121,0.12);
        }
        .hero-meta {
            display: flex;
            align-items: center;
            justify-content: flex-end;
            flex-wrap: wrap;
            gap: 0.65rem;
        }
        .hero-title {
            font-size: 1.25rem !important;
            line-height: 1.2;
            font-weight: 700;
            background: linear-gradient(110deg, #fff4d8 0%, #e0bd79 55%, #a9e3ce 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 0;
            padding-bottom: 0.1rem;
        }
        .hero-subtitle {
            color: var(--text-muted);
            font-size: 0.72rem;
            margin: 0;
            font-weight: 400;
        }
        .dataset-pill {
            display: inline-flex;
            align-items: center;
            max-width: 290px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            padding: 0.38rem 0.75rem;
            border-radius: 999px;
            background: rgba(255,255,255,0.035);
            border: 1px solid rgba(167,243,208,0.13);
            color: #c4d2cd;
            font-size: 0.76rem;
        }
        .status-pill {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 0.4rem 0.8rem;
            border-radius: 9999px;
            font-size: 0.74rem;
            font-weight: 600;
            background: rgba(52, 211, 153, 0.10);
            color: var(--mint);
            border: 1px solid rgba(52, 211, 153, 0.28);
            box-shadow: 0 0 16px rgba(52, 211, 153, 0.10);
        }
        .status-pill-alert {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 0.4rem 0.9rem;
            border-radius: 9999px;
            font-size: 0.8rem;
            font-weight: 600;
            background: rgba(224, 189, 121, 0.10);
            color: var(--champagne);
            border: 1px solid rgba(224, 189, 121, 0.34);
        }

        .kpi-card {
            background: rgba(11, 21, 20, 0.72);
            backdrop-filter: blur(22px) saturate(140%);
            -webkit-backdrop-filter: blur(22px) saturate(140%);
            border: 1px solid rgba(167, 243, 208, 0.11);
            border-radius: 18px;
            padding: 0.78rem 0.9rem;
            margin-bottom: 0.55rem;
            box-shadow: 0 18px 45px rgba(0,0,0,0.45), inset 0 1px 0 rgba(255,255,255,0.06);
            transition: transform 0.2s ease, border-color 0.2s ease;
        }
        .evaluation-kpi-card {
            min-height: 128px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }
        .evaluation-kpi-caption {
            color: #9aa9a5;
            font-size: 0.58rem;
            line-height: 1.2;
            margin-top: 0.45rem;
            white-space: normal;
        }
        .kpi-card:hover {
            transform: translateY(-2px);
            border-color: rgba(224, 189, 121, 0.45);
            box-shadow: 0 0 22px rgba(224,189,121,0.10), 0 18px 45px rgba(0,0,0,0.42);
        }
        .kpi-label {
            font-size: 0.68rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #9eada8;
            margin-bottom: 0.2rem;
        }
        .kpi-value {
            font-size: 1.25rem;
            font-weight: 700;
            color: #f1f5f4;
        }
        .metric-tile .kpi-label {
            font-size: 0.58rem;
            letter-spacing: 0.04em;
        }
        .metric-tile .kpi-value {
            font-size: 1.05rem;
        }
        .metric-tile {
            min-height: 82px;
            padding: 0.75rem;
            border-radius: 14px;
            background: rgba(5, 9, 8, 0.42);
            border: 1px solid rgba(167,243,208,0.13);
        }
        .panel-heading {
            color: #f5e5c1;
            font-size: 0.92rem;
            font-weight: 650;
            margin: 0.15rem 0 0.65rem;
        }
        [data-testid="stAlert"] {
            background: rgba(11, 21, 20, 0.84);
            backdrop-filter: blur(16px);
            border: 1px solid rgba(224, 189, 121, 0.22);
            border-radius: 16px;
        }
        [data-testid="stVerticalBlockBorderWrapper"] {
            background: rgba(11, 21, 20, 0.72);
            backdrop-filter: blur(22px) saturate(140%);
            -webkit-backdrop-filter: blur(22px) saturate(140%);
            border: 1px solid rgba(167, 243, 208, 0.11);
            border-radius: 18px;
            box-shadow: 0 18px 45px rgba(0,0,0,0.45), inset 0 1px 0 rgba(255,255,255,0.06);
            transition: border-color 0.2s ease, box-shadow 0.2s ease;
        }
        [data-testid="stVerticalBlockBorderWrapper"]:hover {
            border-color: rgba(224, 189, 121, 0.45);
            box-shadow: 0 0 20px rgba(224,189,121,0.08), 0 18px 45px rgba(0,0,0,0.45);
        }
        .empty-state {
            padding: 2.2rem 1.5rem;
            border-radius: 18px;
            border: 1px dashed rgba(167, 243, 208, 0.20);
            background: rgba(11, 21, 20, 0.48);
            backdrop-filter: blur(18px);
            text-align: center;
            margin: 1rem 0;
        }
        .route-tag {
            font-size: 0.72rem;
            font-weight: 600;
            color: var(--champagne);
            padding: 0.15rem 0.5rem;
            border-radius: 999px;
            background: rgba(224, 189, 121, 0.10);
            border: 1px solid rgba(224, 189, 121, 0.25);
            display: inline-block;
            margin-top: 0.4rem;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 5px;
            background: rgba(11, 21, 20, 0.72);
            padding: 5px;
            border-radius: 15px;
            border: 1px solid rgba(167, 243, 208, 0.11);
            margin-bottom: 0.8rem;
            box-shadow: 0 10px 24px rgba(0,0,0,0.25);
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 11px;
            color: #a7b3af;
            padding: 8px 14px;
            font-weight: 600;
            transition: all 0.2s ease;
        }
        .stTabs [aria-selected="true"] {
            background: rgba(224, 189, 121, 0.12) !important;
            color: #fff4d8 !important;
            border: 1px solid rgba(224, 189, 121, 0.45) !important;
            box-shadow: 0 0 16px rgba(224, 189, 121, 0.12);
        }
        .stTabs [data-baseweb="tab"]:hover {
            color: #f3dfb2;
        }
        [data-testid="stRadio"] [role="radiogroup"] {
            gap: 0.35rem;
            flex-wrap: wrap;
        }
        [data-testid="stRadio"] label[data-baseweb="radio"] {
            border: 1px solid rgba(167,243,208,0.12);
            border-radius: 999px;
            padding: 0.32rem 0.65rem;
            background: rgba(5,9,8,0.4);
        }
        [data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) {
            border-color: rgba(224,189,121,0.48);
            background: rgba(224,189,121,0.10);
        }
        .stButton > button, .stDownloadButton > button {
            border-radius: 999px;
            border: 1px solid rgba(167, 243, 208, 0.16);
            background: rgba(11, 21, 20, 0.86);
            color: #e9f1ed;
            transition: all 0.2s ease;
        }
        .stButton > button:hover, .stDownloadButton > button:hover {
            border-color: rgba(224, 189, 121, 0.55);
            color: #fff4d8;
            box-shadow: 0 0 18px rgba(224, 189, 121, 0.13);
        }
        [data-testid="stChatInput"] {
            border: 1px solid rgba(224, 189, 121, 0.38);
            border-radius: 999px;
            background: rgba(11, 21, 20, 0.92);
            box-shadow: none;
            overflow: hidden;
        }
        [data-testid="stChatInput"] > div {
            border: 0 !important;
            border-radius: inherit !important;
            background: transparent !important;
            box-shadow: none !important;
        }
        [data-testid="stChatInput"] [data-testid="stChatInputTextArea"] {
            border: 0 !important;
            border-radius: inherit !important;
            background: transparent !important;
        }
        [data-testid="stChatInput"] textarea {
            color: #f1f5f4;
        }
        [data-testid="stChatMessage"] {
            background: rgba(11, 21, 20, 0.45);
            border: 1px solid rgba(167, 243, 208, 0.08);
            border-radius: 15px;
            padding: 0.7rem 0.85rem;
            margin-bottom: 0.55rem;
        }
        [data-testid="stExpander"] {
            border-color: rgba(167, 243, 208, 0.12);
            border-radius: 13px;
        }
        [data-testid="stDataFrame"], [data-testid="stTable"] {
            border: 1px solid rgba(167, 243, 208, 0.11);
            border-radius: 12px;
            overflow: hidden;
        }
        [data-testid="stElementToolbar"] {
            background: rgba(11, 21, 20, 0.92);
            border: 1px solid rgba(224, 189, 121, 0.35);
            color: #f1f5f4;
        }
        @media (max-width: 900px) {
            .hero-container { align-items: flex-start; flex-direction: column; }
            .hero-meta { justify-content: flex-start; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # --- Session State Initialization ---
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "df" not in st.session_state:
        st.session_state.df = None
    if "raw_df" not in st.session_state:
        st.session_state.raw_df = None
    if "preprocessing_logs" not in st.session_state:
        st.session_state.preprocessing_logs = []
    if "df_name" not in st.session_state:
        st.session_state.df_name = ""
    if "data_source" not in st.session_state:
        st.session_state.data_source = ""
    if "last_uploaded_file_signature" not in st.session_state:
        st.session_state.last_uploaded_file_signature = None
    if "visual_stage_mode_v2" not in st.session_state:
        st.session_state.visual_stage_mode_v2 = "📋 Dataset Overview"
    if "pending_prompt" not in st.session_state:
        st.session_state.pending_prompt = None
    if "evaluation_result" not in st.session_state:
        st.session_state.evaluation_result = None
    if "eval_table_rev" not in st.session_state:
        st.session_state.eval_table_rev = 0
    if "offline_security_metrics" not in st.session_state:
        st.session_state.offline_security_metrics = {}
    if "live_benchmark_metrics" not in st.session_state:
        st.session_state.live_benchmark_metrics = {}
    pending_visual_mode = st.session_state.pop("pending_visual_stage_mode_v2", None)
    if pending_visual_mode:
        st.session_state.visual_stage_mode_v2 = pending_visual_mode

    # Discover active AI backend routes from .env
    active_routes = get_available_routes()
    route_labels = [r["provider"] for r in active_routes]
    has_active_engine = len(active_routes) > 0

    # --- Minimal Translucent Sidebar ---
    with st.sidebar:
        st.markdown("### 📊 Chat-with-your-CSV")
        st.caption("Zero-Config Natural-Language Tabular Analyst")
        st.divider()

        # 1. Dataset Management
        st.markdown("#### 📁 Dataset")
        uploaded_file = st.file_uploader(
            "Upload CSV File",
            type=["csv"],
            help="Upload any tabular CSV file to query in natural language.",
        )
        uploaded_file_signature = None
        if uploaded_file is not None:
            upload_file_id = getattr(uploaded_file, "file_id", None) or uploaded_file.name
            uploaded_file_signature = (upload_file_id, uploaded_file.size)

        demo_path = os.path.join(os.path.dirname(__file__), "data.csv")
        if os.path.exists(demo_path):
            if st.button("🎵 Load Sample Dataset (Spotify)", **width_kwargs):
                with open(demo_path, "rb") as f:
                    demo_bytes = f.read()
                loaded_df = load_csv_data_cached(demo_bytes, "data.csv")
                st.session_state.raw_df = loaded_df.copy()
                st.session_state.df = loaded_df.copy()
                st.session_state.df_name = "data.csv (Spotify Tracks)"
                st.session_state.data_source = "sample"
                st.session_state.preprocessing_logs = []
                st.session_state.messages = []
                st.session_state["ai_eda_insight"] = None
                st.session_state.last_uploaded_file_signature = uploaded_file_signature
                st.session_state.visual_stage_mode_v2 = "📋 Dataset Overview"
                st.toast("Loaded Spotify sample dataset!", icon="✅")

        # Handle uploaded CSV without overwriting sample dataset on subsequent rerun
        if (
            uploaded_file is not None
            and uploaded_file_signature != st.session_state.last_uploaded_file_signature
        ):
            try:
                file_bytes = uploaded_file.getvalue()
                loaded_df = load_csv_data_cached(file_bytes, uploaded_file.name)
            except Exception as e:
                st.error(f"Could not load CSV: {e}")
            else:
                st.session_state.raw_df = loaded_df.copy()
                st.session_state.df = loaded_df.copy()
                st.session_state.df_name = uploaded_file.name
                st.session_state.data_source = "upload"
                st.session_state.last_uploaded_file_signature = uploaded_file_signature
                st.session_state.preprocessing_logs = []
                st.session_state.messages = []
                st.session_state["ai_eda_insight"] = None
                st.session_state.visual_stage_mode_v2 = "📋 Dataset Overview"
                st.toast(f"Loaded {uploaded_file.name}", icon="📁")

        # Keep raw_df synchronized if not yet set
        if st.session_state.df is not None and st.session_state.raw_df is None:
            st.session_state.raw_df = st.session_state.df.copy()
        profile = (
            compute_data_profile(st.session_state.df)
            if st.session_state.df is not None
            else None
        )

        st.divider()

        # 2. AI Engine Status (Zero-Config .env Indicator)
        st.markdown("#### ⚡ AI Engine Status")
        if has_active_engine:
            st.markdown(
                f"""
                <div class="status-pill">
                    ● Connected: {' + '.join(route_labels)}
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.caption("Auto-failover: Groq (Primary) ➔ Gemini (Backup)")
        else:
            st.markdown(
                """
                <div class="status-pill-alert">
                    ⚠️ No API Keys Detected
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.caption("Please configure `GROQ_API_KEY` or `GEMINI_API_KEY` in your `.env` file.")

        st.divider()

        # 3. Settings & History Controls
        st.markdown("#### 🛠️ Controls")
        show_code = st.checkbox("Show generated Python code", value=True)
        if st.button("🗑️ Clear Conversation", **width_kwargs):
            st.session_state.messages = []
            st.session_state.visual_stage_mode_v2 = "📋 Dataset Overview"
            st.rerun()

        # 4. Session Report Export (.md)
        if st.session_state.df is not None:
            st.divider()
            st.markdown("#### 📄 Session Report")
            report_md_str = export_session_report_markdown(
                df_name=st.session_state.df_name,
                df=st.session_state.df,
                profile=profile,
                preprocessing_logs=st.session_state.preprocessing_logs,
                messages=st.session_state.messages,
            )
            safe_rep_name = re.sub(r"[^\w\-.]", "_", st.session_state.df_name)
            st.download_button(
                label="📄 Export Session Report (.md)",
                data=report_md_str.encode("utf-8"),
                file_name=f"Session_Report_{safe_rep_name}.md",
                mime="text/markdown",
                key="sidebar_dl_report_btn",
                **width_kwargs,
            )

    # --- Compact Glass Command Bar ---
    display_routes = [
        "Gemini" if route["provider"] == "Google Gemini" else route["provider"]
        for route in active_routes
    ]
    status_html = (
        f'<span class="status-pill">● {html.escape(" + ".join(display_routes))} Active</span>'
        if has_active_engine
        else '<span class="status-pill-alert">⚠️ AI keys not configured</span>'
    )
    dataset_label = html.escape(st.session_state.df_name or "No dataset loaded")

    st.markdown(
        f"""
        <div class="hero-container">
            <div class="hero-brand">
                <span class="brand-mark">✦</span>
                <div>
                    <h1 class="hero-title">Chat-with-your-CSV</h1>
                    <p class="hero-subtitle">NATURAL LANGUAGE ANALYTICS · GROUNDED AI</p>
                </div>
            </div>
            <div class="hero-meta">
                <span class="dataset-pill">◈ {dataset_label}</span>
                {status_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    df = st.session_state.df

    # --- State 1: No Data Uploaded Yet ---
    if df is None:
        st.markdown(
            """
            <div class="empty-state">
                <h3 style="color: #cbd5e1; margin-bottom: 0.5rem;">📂 No Dataset Loaded</h3>
                <p style="color: #64748b; max-width: 500px; margin: 0 auto 1.2rem auto;">
                    Upload any CSV file using the sidebar, or click below to launch instant queries on the built-in Spotify track dataset.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if os.path.exists(demo_path):
            if st.button("🎵 Load Sample Dataset (Spotify)", key="main_demo_btn", **width_kwargs):
                with open(demo_path, "rb") as f:
                    demo_bytes = f.read()
                loaded_df = load_csv_data_cached(demo_bytes, "data.csv")
                st.session_state.raw_df = loaded_df.copy()
                st.session_state.df = loaded_df.copy()
                st.session_state.df_name = "data.csv (Spotify Tracks)"
                st.session_state.data_source = "sample"
                st.session_state.preprocessing_logs = []
                st.session_state.messages = []
                st.session_state["ai_eda_insight"] = None
                st.session_state.last_uploaded_file_signature = uploaded_file_signature
                st.session_state.visual_stage_mode_v2 = "📋 Dataset Overview"
                st.rerun()
        st.stop()
        return

    # --- State 2: Data loaded ---
    missing_count = int(df.isnull().sum().sum())

    # --- Multi-Tab Workspace ---
    tab_chat, tab_profiler, tab_eval = st.tabs([
        "💬 AI Data Chat",
        "🧹 Data Profiler & Preprocessing",
        "🧪 Evaluation & Benchmark",
    ])

    # ==========================================================================
    # TAB 1: AI Data Chat
    # ==========================================================================
    with tab_chat:
        col_left, col_right = st.columns([1.45, 1.0], gap="large")
        latest_assistant_idx = next(
            (
                idx for idx in range(len(st.session_state.messages) - 1, -1, -1)
                if st.session_state.messages[idx].get("role") == "assistant"
            ),
            None,
        )
        latest_assistant = (
            st.session_state.messages[latest_assistant_idx]
            if latest_assistant_idx is not None
            else None
        )
        has_active_visual = bool(
            latest_assistant and any(
                latest_assistant.get(key) is not None
                for key in ("plotly_json", "figure_png", "result_df", "result")
            )
        )

        with col_left:
            st.markdown("#### ◈ Visual Stage")
            with st.container(border=True):
                stage_mode = st.radio(
                    "Visual stage",
                    ["📊 Active Query Visual", "📋 Dataset Overview"],
                    horizontal=True,
                    label_visibility="collapsed",
                    index=0 if has_active_visual else 1,
                    key="visual_stage_mode_v2",
                )
                if stage_mode == "📊 Active Query Visual":
                    if has_active_visual:
                        if latest_assistant.get("plotly_json"):
                            active_figure = pio.from_json(latest_assistant["plotly_json"])
                            st.plotly_chart(
                                active_figure,
                                key="stage_active_plotly",
                                **width_kwargs,
                            )
                            st.download_button(
                                "🌐 Download Interactive Chart (HTML)",
                                data=active_figure.to_html(full_html=True, include_plotlyjs=True),
                                file_name="interactive_chart.html",
                                mime="text/html",
                                key="stage_active_html",
                                **width_kwargs,
                            )
                        if latest_assistant.get("figure_png") is not None:
                            st.image(latest_assistant["figure_png"], **width_kwargs)
                            st.download_button(
                                "📥 Download Chart (PNG)",
                                data=latest_assistant["figure_png"],
                                file_name="chart.png",
                                mime="image/png",
                                key="stage_active_png",
                                **width_kwargs,
                            )
                        if latest_assistant.get("result_df") is not None:
                            st.dataframe(latest_assistant["result_df"], **width_kwargs)
                            st.download_button(
                                "📥 Download Result Table as CSV",
                                data=latest_assistant["result_df"].to_csv(index=False).encode("utf-8"),
                                file_name="query_result.csv",
                                mime="text/csv",
                                key="stage_active_csv",
                                **width_kwargs,
                            )
                        if latest_assistant.get("result") is not None:
                            st.metric("Verified Result", str(latest_assistant["result"]))
                    else:
                        st.info("Your latest chart or result table will appear here.")
                else:
                    overview = build_dataset_overview_figure(df)
                    st.plotly_chart(
                        overview,
                        key="dataset_overview",
                        **width_kwargs,
                    )
                    st.markdown("##### 📊 Data Preview")
                    st.dataframe(df.head(8), hide_index=True, **width_kwargs)

            with st.expander("🔍 Inspect Dataset · Preview, Schema & Statistics", expanded=False):
                preview_tab, schema_tab, stats_tab = st.tabs(
                    ["📋 Preview", "🏷️ Schema", "📊 Statistics"]
                )
                with preview_tab:
                    st.dataframe(df.head(10), **width_kwargs)
                with schema_tab:
                    row_count = max(len(df), 1)
                    col_summary = pd.DataFrame({
                        "Column Name": df.columns,
                        "Data Type": [str(dtype) for dtype in df.dtypes],
                        "Non-Null Count": df.notnull().sum().values,
                        "Null Count": df.isnull().sum().values,
                        "Null Percentage (%)": (
                            (df.isnull().sum() / row_count) * 100
                        ).round(2).values,
                        "Unique Values": [df[column].nunique() for column in df.columns],
                    })
                    st.dataframe(col_summary, hide_index=True, **width_kwargs)
                with stats_tab:
                    if len(df) == 0:
                        st.info("Statistics are unavailable because this dataset has no rows.")
                    else:
                        st.dataframe(df.describe(include="all").T, **width_kwargs)

        with col_right:
            with st.container(border=True):
                status_text = "Active" if has_active_engine else "Unavailable"
                status_class = "status-pill" if has_active_engine else "status-pill-alert"
                st.markdown(
                    f"""
                    <div style="display:flex;justify-content:space-between;align-items:center;gap:0.6rem;margin-bottom:0.7rem;">
                        <div>
                            <div class="panel-heading" style="margin-bottom:0.12rem;">Active Dataset</div>
                            <div style="color:#d5e0dc;font-size:0.84rem;word-break:break-word;">{html.escape(st.session_state.df_name)}</div>
                        </div>
                        <span class="{status_class}">● {status_text}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f"""
                    <div class="kpi-card" style="margin-bottom:0.7rem;">
                        <div style="display:flex;justify-content:space-between;align-items:center;gap:0.8rem;">
                            <div>
                                <div class="kpi-label">Data Health Score</div>
                                <div class="kpi-value" style="color:#e0bd79;">{profile['health_score']}%</div>
                            </div>
                            <div style="width:58%;max-width:250px;">{build_column_sparkline(df)}</div>
                        </div>
                        <div style="color:#9aa9a5;font-size:0.68rem;margin-top:0.15rem;">{profile['cols']} columns · {profile['missing_cells']} missing cells · {profile['duplicate_rows']} duplicates</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                metric_values = [
                    ("Total Rows", f"{df.shape[0]:,}"),
                    ("Columns", f"{df.shape[1]:,}"),
                    ("Missing Cells", f"{missing_count:,}"),
                    (
                        "Outliers / Duplicates",
                        f"{profile['total_outliers']:,} / {profile['duplicate_rows']:,}",
                    ),
                ]
                metric_col1, metric_col2 = st.columns(2)
                for index, (label, value) in enumerate(metric_values):
                    with (metric_col1 if index % 2 == 0 else metric_col2):
                        st.markdown(
                            f"""
                            <div class="metric-tile">
                                <div class="kpi-label">{label}</div>
                                <div class="kpi-value">{value}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                st.markdown('<div class="panel-heading">🧠 AI Analysis Stream</div>', unsafe_allow_html=True)
                with st.container(height=270, border=False):
                    for idx, message in enumerate(st.session_state.messages):
                        role = message.get("role", "user")
                        with st.chat_message(role, avatar="🧑‍💻" if role == "user" else "🤖"):
                            status = message.get("status", "success")
                            if status == "error":
                                category = message.get("failure_category", "Error")
                                st.error(f"⚠️ **{category}**\n\n{message.get('content', '')}")
                            elif status == "declined":
                                st.warning(f"🛡️ **Request Declined**\n\n{message.get('content', '')}")
                            else:
                                st.markdown(message.get("content", ""))

                            if message.get("route_label"):
                                st.markdown(
                                    f'<span class="route-tag">Engine: {html.escape(message["route_label"])}</span>',
                                    unsafe_allow_html=True,
                                )
                            if message.get("code") and show_code:
                                with st.expander("💻 View Code", expanded=False):
                                    st.code(message["code"], language="python")
                            if message.get("stdout"):
                                st.caption(f"Console Output:\n{message['stdout']}")
                            if message.get("result") is not None:
                                st.write(message["result"])
                            has_message_visual = any(
                                message.get(key) is not None
                                for key in ("result_df", "figure_png", "plotly_json")
                            )
                            if has_message_visual and role == "assistant":
                                if idx == latest_assistant_idx:
                                    st.caption("📌 Visual / table displayed on Visual Stage")
                                else:
                                    with st.expander("📊 View Visual / Table", expanded=False):
                                        if message.get("result_df") is not None:
                                            st.dataframe(message["result_df"], **width_kwargs)
                                            st.download_button(
                                                "📥 Download Result Table as CSV",
                                                data=message["result_df"].to_csv(index=False).encode("utf-8"),
                                                file_name="query_result.csv",
                                                mime="text/csv",
                                                key=f"dl_csv_{idx}",
                                            )
                                        if message.get("figure_png") is not None:
                                            st.image(message["figure_png"], **width_kwargs)
                                            st.download_button(
                                                "📥 Download Chart (PNG)",
                                                data=message["figure_png"],
                                                file_name="chart.png",
                                                mime="image/png",
                                                key=f"dl_png_{idx}",
                                            )
                                        if message.get("plotly_json"):
                                            render_plotly_chart(
                                                message["plotly_json"],
                                                str(idx),
                                                width_kwargs,
                                            )
                            if role == "assistant" and message.get("pipeline_trace"):
                                render_pipeline_trace(message["pipeline_trace"])

                st.markdown('<div class="panel-heading">⚡ Suggested Commands</div>', unsafe_allow_html=True)
                suggestions = generate_suggested_prompts(df)
                prompt_cols = st.columns(2)
                for index, suggestion in enumerate(suggestions):
                    with prompt_cols[index % len(prompt_cols)]:
                        if st.button(
                            suggestion,
                            key=f"sugg_{index}",
                            **width_kwargs,
                        ):
                            st.session_state.pending_prompt = suggestion
                            st.rerun()

                user_input = st.chat_input("Ask a question about your CSV...")
                if active_routes:
                    agent_route = active_routes[0]
                    agent_model = agent_route["primary_model"].replace("openai/", "")
                    st.markdown(
                        f'<span class="route-tag">⚡ {html.escape(agent_route["provider"])} {html.escape(agent_model)}</span>',
                        unsafe_allow_html=True,
                    )
                if st.session_state.pending_prompt:
                    user_input = st.session_state.pending_prompt
                    st.session_state.pending_prompt = None

                if user_input:
                    if not has_active_engine:
                        st.error("⚠️ No active AI engine found. Please configure an API key in `.env`.")
                        st.stop()
                        return

                    st.session_state.messages.append({
                        "role": "user",
                        "content": user_input,
                    })
                    with st.chat_message("user", avatar="🧑‍💻"):
                        st.markdown(user_input)

                    with st.chat_message("assistant", avatar="🤖"):
                        with st.spinner("Analyzing data and generating query via auto-pipeline..."):
                            schema_context = get_df_schema_prompt(df)
                            recent_history: List[str] = []
                            for message in st.session_state.messages[-4:]:
                                role = message.get("role")
                                content = message.get("content", "")
                                code = message.get("code")
                                if role == "user":
                                    recent_history.append(f"User: {content}")
                                elif role == "assistant":
                                    if code:
                                        recent_history.append(
                                            f"Assistant Query:\n```python\n{code}\n```\nAnswer: {content}"
                                        )
                                    else:
                                        recent_history.append(f"Assistant: {content}")

                            (
                                code,
                                exec_res,
                                decline,
                                active_label,
                                pipeline_err,
                                pipeline_trace,
                            ) = execute_query_pipeline(
                                prompt=user_input,
                                df_schema=schema_context,
                                df=df,
                                conversation_history=recent_history,
                            )

                            if pipeline_err:
                                pipeline_trace["grounded_status"] = "Not run (pipeline failed)"
                                st.error(f"⚠️ **Pipeline Error**\n\n{pipeline_err}")
                                st.session_state.messages.append({
                                    "role": "assistant",
                                    "content": pipeline_err,
                                    "status": "error",
                                    "failure_category": "Provider/API Error",
                                    "pipeline_trace": pipeline_trace,
                                })
                            elif decline:
                                st.warning(f"🛡️ **Request Declined**\n\n{decline}")
                                st.session_state.messages.append({
                                    "role": "assistant",
                                    "content": decline,
                                    "status": "declined",
                                    "route_label": active_label,
                                    "pipeline_trace": pipeline_trace,
                                })
                            elif exec_res and exec_res.get("error"):
                                pipeline_trace["grounded_status"] = "Not run (execution failed)"
                                category, friendly_message = classify_failure(exec_res["error"], df)
                                st.error(f"⚠️ **{category}**\n\n{friendly_message}")
                                if code and show_code:
                                    with st.expander("💻 View Attempted Code"):
                                        st.code(code, language="python")
                                st.session_state.messages.append({
                                    "role": "assistant",
                                    "content": friendly_message,
                                    "code": code,
                                    "status": "error",
                                    "failure_category": category,
                                    "route_label": active_label,
                                    "pipeline_trace": pipeline_trace,
                                })
                            elif exec_res:
                                grounded_answer = generate_grounded_answer(
                                    prompt=user_input,
                                    code=code,
                                    exec_res=exec_res,
                                )
                                pipeline_trace["grounded_status"] = "Verified against computed output"
                                st.markdown(grounded_answer)
                                if active_label:
                                    st.markdown(
                                        f'<span class="route-tag">Engine: {html.escape(active_label)}</span>',
                                        unsafe_allow_html=True,
                                    )
                                if exec_res.get("stdout"):
                                    st.caption(f"Console Output:\n{exec_res['stdout']}")
                                if code and show_code:
                                    with st.expander("💻 View Code", expanded=False):
                                        st.code(code, language="python")
                                if exec_res.get("result") is not None:
                                    st.write(exec_res["result"])
                                if any(
                                    exec_res.get(key) is not None
                                    for key in ("result_df", "figure_png", "plotly_json")
                                ):
                                    st.caption("📌 Visual / table displayed on Visual Stage")
                                st.session_state.messages.append({
                                    "role": "assistant",
                                    "content": grounded_answer,
                                    "code": code,
                                    "result": exec_res.get("result"),
                                    "result_df": exec_res.get("result_df"),
                                    "figure_png": exec_res.get("figure_png"),
                                    "plotly_json": exec_res.get("plotly_json"),
                                    "stdout": exec_res.get("stdout"),
                                    "route_label": active_label,
                                    "status": "success",
                                    "pipeline_trace": pipeline_trace,
                                })

                        render_pipeline_trace(pipeline_trace)
                    if exec_res and any(
                        exec_res.get(key) is not None
                        for key in ("result_df", "figure_png", "plotly_json")
                    ):
                        st.session_state["pending_visual_stage_mode_v2"] = (
                            "📊 Active Query Visual"
                        )
                    st.rerun()

    # ==========================================================================
    # TAB 2: Data Profiler & Preprocessing
    # ==========================================================================
    with tab_profiler:
        # 4 Profiler Metric Cards
        p_col1, p_col2, p_col3, p_col4 = st.columns(4)
        health_score = profile["health_score"]
        if health_score >= 85:
            score_color = "#34d399"
            score_badge = "Excellent"
        elif health_score >= 70:
            score_color = "#fbbf24"
            score_badge = "Moderate"
        else:
            score_color = "#f87171"
            score_badge = "Needs Review"

        with p_col1:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">Data Health Score</div>
                    <div class="kpi-value" style="color: {score_color};">{health_score}% <span style="font-size: 0.8rem; font-weight: 500; color: #94a3b8;">({score_badge})</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with p_col2:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">Duplicate Rows</div>
                    <div class="kpi-value">{profile['duplicate_rows']:,} <span style="font-size: 0.8rem; font-weight: 500; color: #94a3b8;">({profile['duplicate_pct']}%)</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with p_col3:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">IQR Outliers</div>
                    <div class="kpi-value">{profile['total_outliers']:,}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with p_col4:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">Constant Columns</div>
                    <div class="kpi-value">{profile['constant_cols']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("##### ⚙️ Interactive Preprocessing Pipeline")
        st.caption("Apply transformations to the active workspace. AI Chat will query the cleaned dataset.")

        prep_c1, prep_c2 = st.columns(2)
        with prep_c1:
            prep_drop_dups = st.checkbox("Drop Exact Duplicate Rows", value=False)
            prep_strip_ws = st.checkbox("Trim Text Whitespace & Convert Empty Strings to Null", value=False)
            prep_drop_const = st.checkbox("Drop Constant Columns (0 variance)", value=False)

        with prep_c2:
            prep_impute = st.selectbox(
                "Missing Value Strategy",
                ["none", "median_mode", "mean_mode", "drop_rows"],
                format_func=lambda x: {
                    "none": "None (Preserve missing values)",
                    "median_mode": "Impute (Median for numeric, Mode for categorical)",
                    "mean_mode": "Impute (Mean for numeric, Mode for categorical)",
                    "drop_rows": "Drop Rows with any missing cells",
                }[x],
            )
            prep_outliers = st.selectbox(
                "IQR Outlier Strategy",
                ["none", "clip_iqr", "remove_iqr"],
                format_func=lambda x: {
                    "none": "None (Preserve outliers)",
                    "clip_iqr": "Clip/Winsorize outliers to [Q1 - 1.5*IQR, Q3 + 1.5*IQR] (skip binary)",
                    "remove_iqr": "Remove rows with IQR outliers (skip binary)",
                }[x],
            )

        act_col1, act_col2, act_col3 = st.columns([1.5, 1.2, 1.3])
        with act_col1:
            if st.button("⚡ Apply Preprocessing to Workspace", key="btn_apply_prep", **width_kwargs):
                cleaned_df, logs = apply_preprocessing(
                    st.session_state.df,
                    drop_duplicates=prep_drop_dups,
                    strip_whitespace=prep_strip_ws,
                    impute_strategy=prep_impute,
                    outlier_strategy=prep_outliers,
                    drop_constant_cols=prep_drop_const,
                )
                st.session_state.df = cleaned_df
                st.session_state.preprocessing_logs.extend(logs)
                st.toast("Preprocessing applied! AI Chat will now query the cleaned dataset.", icon="⚡")
                st.rerun()

        with act_col2:
            if st.button("↩️ Reset to Original CSV", key="btn_reset_prep", **width_kwargs):
                if st.session_state.raw_df is not None:
                    st.session_state.df = st.session_state.raw_df.copy()
                    st.session_state.preprocessing_logs = []
                    st.toast("Reset workspace to original raw CSV!", icon="↩️")
                    st.rerun()

        with act_col3:
            cleaned_csv_bytes = st.session_state.df.to_csv(index=False).encode("utf-8")
            cleaned_source_name = re.sub(
                r"\s*\(Spotify Tracks\)\s*$",
                "",
                str(st.session_state.df_name),
            )
            cleaned_source_name = os.path.basename(cleaned_source_name)
            cleaned_stem = os.path.splitext(cleaned_source_name)[0] or "dataset"
            st.download_button(
                label="📥 Download Cleaned CSV",
                data=cleaned_csv_bytes,
                file_name=f"cleaned_{cleaned_stem}.csv",
                mime="text/csv",
                key="btn_dl_cleaned_csv",
                **width_kwargs,
            )

        # Preprocessing Audit Trail
        st.markdown("##### 📜 Preprocessing Audit Trail")
        if st.session_state.preprocessing_logs:
            for idx, entry in enumerate(st.session_state.preprocessing_logs, 1):
                st.markdown(f"**Step {idx}:** {entry}")
        else:
            st.info("No preprocessing transformations applied yet. Dataset is in original raw state.")

        st.divider()

        # Column Diagnostics Table
        st.markdown("##### 📋 Per-Column Diagnostics & Quality Report")
        st.dataframe(profile["column_report"], hide_index=True, **width_kwargs)

        # Top Correlations Table
        if profile.get("top_correlations"):
            st.markdown("##### 🔗 Top Feature Correlations (Pearson)")
            corr_display_df = pd.DataFrame(profile["top_correlations"])[["feature_1", "feature_2", "correlation"]]
            corr_display_df.columns = ["Feature 1", "Feature 2", "Pearson Correlation (r)"]
            st.dataframe(corr_display_df, hide_index=True, **width_kwargs)

        st.divider()

        # Auto-EDA Executive Insights
        st.markdown("##### ✨ Auto-EDA Executive Insights")
        eda_bullets = build_deterministic_eda_insights(st.session_state.df, profile)
        for bullet in eda_bullets:
            st.markdown(f"- {bullet}")

        # Optional AI Executive Briefing
        if st.button("🤖 Enrich Insights with AI", key="btn_enrich_ai", **width_kwargs):
            with st.spinner("Generating AI executive briefing grounded in dataset profile..."):
                deterministic_insights = "\n".join(f"- {bullet}" for bullet in eda_bullets)
                prompt = f"""Based STRICTLY on the computed data profile metrics below, provide a high-level executive briefing (3 concise paragraphs) detailing key statistical takeaways, potential data risks, and recommended modeling strategies. DO NOT hallucinate.

Dataset Name: {st.session_state.df_name}
Rows: {profile['rows']:,}, Columns: {profile['cols']}
Health Score: {profile['health_score']}/100
Missing Cells: {profile['missing_cells']:,} ({profile['missing_pct']}%)
Duplicate Rows: {profile['duplicate_rows']:,} ({profile['duplicate_pct']}%)
Outliers Detected: {profile['total_outliers']:,}
Top Correlations: {profile['top_correlations']}
Columns: {list(st.session_state.df.columns)}

Deterministic EDA Insights (cite these exact computed findings, including class percentages):
{deterministic_insights}
"""
                st.session_state["ai_eda_insight"] = generate_executive_briefing(prompt)

        if st.session_state.get("ai_eda_insight"):
            with st.container(border=True):
                st.markdown("#### 🤖 AI Executive Commentary")
                st.markdown(st.session_state["ai_eda_insight"])

    # ==========================================================================
    # TAB 3: Evaluation & Benchmark
    # ==========================================================================
    with tab_eval:
        st.markdown("### 🧪 Gen-AI Benchmark & Guardrail Evaluation")
        st.caption(
            "The offline audit uses no API calls. Live runs call the selected provider; "
            "the fast subset uses six of the 16 curated cases."
        )

        provider_choice = st.selectbox(
            "Live benchmark provider",
            ["Auto Router", "Groq", "Google Gemini", "Groq vs Google Gemini"],
            key="eval_provider_choice",
        )
        suite_mode = st.radio(
            "Benchmark size",
            ["Fast Subset (6 queries)", "Full Suite"],
            horizontal=True,
            key="eval_suite_mode",
        )

        offline_col, live_col = st.columns(2)
        with offline_col:
            if st.button(
                "🛡️ Run Offline Security & Sandbox Audit (Instant)",
                key="run_offline_eval",
                **width_kwargs,
            ):
                with st.spinner("Running sandbox security and reference-query checks..."):
                    offline_result = run_offline_security_eval(df)
                    st.session_state.evaluation_result = offline_result
                    saved_metrics = dict(st.session_state.offline_security_metrics)
                    for key in (
                        "security_block_rate_pct",
                        "false_positive_rate_pct",
                        "average_sandbox_latency_ms",
                    ):
                        value = offline_result["metrics"].get(key)
                        if value is not None:
                            saved_metrics[key] = value
                    st.session_state.offline_security_metrics = saved_metrics

        with live_col:
            if st.button(
                "⚡ Run Live Gen-AI Benchmark (Groq vs Gemini / Fast Subset)",
                key="run_live_eval",
                **width_kwargs,
            ):
                routes = get_available_routes()
                progress_bar = st.progress(0.0)
                progress_caption = st.empty()

                def update_live_progress(
                    current: int,
                    total: int,
                    case_id: str,
                    status_msg: str,
                ) -> None:
                    progress_bar.progress(current / total if total else 0.0)
                    progress_caption.caption(
                        f"{current}/{total} · {case_id} · {status_msg}"
                    )

                try:
                    with st.spinner("Running live provider benchmark; this may use API quota..."):
                        if provider_choice == "Groq vs Google Gemini":
                            result = compare_providers_eval(
                                df=df,
                                routes=routes,
                                generate_pandas_code=call_llm_code_generator_single,
                                suite_mode=suite_mode,
                                progress_callback=update_live_progress,
                            )
                        else:
                            result = run_live_llm_eval(
                                df=df,
                                routes=routes,
                                generate_pandas_code=call_llm_code_generator_single,
                                provider=provider_choice,
                                suite_mode=suite_mode,
                                progress_callback=update_live_progress,
                            )
                    st.session_state.evaluation_result = result
                    saved_metrics = dict(st.session_state.live_benchmark_metrics)
                    for key in (
                        "pass_at_1_accuracy_pct",
                        "self_healed_pass_at_3_accuracy_pct",
                        "average_latency_ms",
                    ):
                        value = result["metrics"].get(key)
                        if value is not None:
                            saved_metrics[key] = value
                    st.session_state.live_benchmark_metrics = saved_metrics
                except ValueError as ex:
                    st.error(str(ex))

        evaluation = st.session_state.get("evaluation_result")
        metrics = evaluation["metrics"] if evaluation else {}
        is_offline = bool(evaluation and evaluation["mode"].startswith("Offline"))
        offline_metrics = st.session_state.offline_security_metrics
        live_metrics = st.session_state.live_benchmark_metrics

        security_block_rate = metrics.get("security_block_rate_pct")
        if security_block_rate is None:
            security_block_rate = offline_metrics.get("security_block_rate_pct")
        false_positive_rate = metrics.get("false_positive_rate_pct")
        if false_positive_rate is None:
            false_positive_rate = offline_metrics.get("false_positive_rate_pct")
        pass_at_1 = metrics.get("pass_at_1_accuracy_pct")
        if pass_at_1 is None:
            pass_at_1 = live_metrics.get("pass_at_1_accuracy_pct")
        pass_at_3 = metrics.get("self_healed_pass_at_3_accuracy_pct")
        if pass_at_3 is None:
            pass_at_3 = live_metrics.get("self_healed_pass_at_3_accuracy_pct")
        average_latency = (
            metrics.get("average_sandbox_latency_ms")
            if is_offline
            else metrics.get("average_latency_ms")
        )
        if average_latency is None:
            average_latency = live_metrics.get("average_latency_ms")
        if average_latency is None:
            average_latency = offline_metrics.get("average_sandbox_latency_ms")

        def format_kpi(value: Optional[float], suffix: str = "%") -> str:
            return "Not Run" if value is None else f"{value:,.1f}{suffix}"

        with st.expander("📖 Metric Definitions & Benchmark Methodology", expanded=False):
            st.markdown(
                """
                - **Security Block Rate:** Share of malicious security payloads stopped by the sandbox; the goal is **100%**.
                - **False-Positive Rate:** Share of safe reference queries wrongly blocked; the goal is **0%**.
                - **Pass@1 Accuracy:** Share of live prompts solved on the first LLM attempt, with no retries.
                - **Self-Healed Pass@3:** Share of live prompts solved within three attempts, including traceback-guided corrections.
                - **Average Latency:** Mean query-to-result time plus sandbox execution; offline runs report sandbox execution time only.

                **Fast Subset (6 queries)** checks live functional, visualization, and recovery prompts. The **Offline Audit** makes no LLM calls and measures security blocking and false positives against the security/reference-query suite.
                """
            )

        metric_cards = [
            (
                "Security Block Rate",
                format_kpi(security_block_rate),
                "Malicious payloads stopped (Goal: 100%)",
            ),
            (
                "False-Positive Rate",
                format_kpi(false_positive_rate),
                "Safe queries wrongly blocked (Goal: 0%)",
            ),
            (
                "Pass@1 Accuracy",
                format_kpi(pass_at_1),
                "Solved on 1st LLM attempt (0 retries)",
            ),
            (
                "Self-Healed Pass@3",
                format_kpi(pass_at_3),
                "Solved within 3 tries via traceback fix",
            ),
            (
                "Avg Latency (ms)",
                format_kpi(average_latency, " ms"),
                "Mean end-to-end query + sandbox time",
            ),
        ]
        metric_columns = st.columns(5)
        for column, (label, value, caption) in zip(metric_columns, metric_cards):
            with column:
                st.markdown(
                    f"""
                    <div class="kpi-card evaluation-kpi-card">
                        <div>
                            <div class="kpi-label">{label}</div>
                            <div class="kpi-value">{value}</div>
                        </div>
                        <div class="evaluation-kpi-caption">{caption}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        if evaluation:
            st.markdown(f"#### Latest Run: {evaluation['mode']}")
            if not is_offline:
                comparison = evaluation.get("provider_comparison")
                if isinstance(comparison, pd.DataFrame) and not comparison.empty:
                    chart_col1, chart_col2 = st.columns(2)
                    with chart_col1:
                        st.markdown("##### Pass@1 vs Self-Healed Pass@3")
                        pass_chart = px.bar(
                            comparison,
                            x="Provider",
                            y=[
                                "Pass@1 Accuracy (%)",
                                "Self-Healed Pass@3 Accuracy (%)",
                            ],
                            barmode="group",
                            text_auto=True,
                            color_discrete_sequence=["#e0bd79", "#34d399"],
                        )
                        st.plotly_chart(
                            _style_evaluation_figure(pass_chart),
                            key="eval_provider_pass_chart",
                            **width_kwargs,
                        )
                    with chart_col2:
                        st.markdown("##### Provider Latency Comparison")
                        latency_chart = px.bar(
                            comparison,
                            x="Provider",
                            y="Average Latency (ms)",
                            color="Provider",
                            text_auto=True,
                            color_discrete_map={
                                "Groq": "#e0bd79",
                                "Google Gemini": "#34d399",
                            },
                        )
                        st.plotly_chart(
                            _style_evaluation_figure(latency_chart),
                            key="eval_provider_latency_chart",
                            **width_kwargs,
                        )
                else:
                    chart_col1, chart_col2 = st.columns(2)
                    with chart_col1:
                        st.markdown("##### Pass@1 vs Self-Healed Pass@3")
                        pass_chart = pd.DataFrame(
                            [
                                {
                                    "Metric": "Pass@1",
                                    "Accuracy (%)": metrics.get("pass_at_1_accuracy_pct"),
                                },
                                {
                                    "Metric": "Self-Healed Pass@3",
                                    "Accuracy (%)": metrics.get(
                                        "self_healed_pass_at_3_accuracy_pct"
                                    ),
                                },
                            ]
                        ).dropna()
                        if not pass_chart.empty:
                            pass_figure = px.bar(
                                pass_chart,
                                x="Metric",
                                y="Accuracy (%)",
                                color="Metric",
                                text_auto=True,
                                color_discrete_map={
                                    "Pass@1": "#e0bd79",
                                    "Self-Healed Pass@3": "#34d399",
                                },
                            )
                            st.plotly_chart(
                                _style_evaluation_figure(pass_figure),
                                key="eval_pass_chart",
                                **width_kwargs,
                            )
                    with chart_col2:
                        st.markdown("##### Average End-to-End Latency")
                        if average_latency is not None:
                            latency_chart = pd.DataFrame(
                                {
                                    "Provider": [
                                        metrics.get("provider", provider_choice)
                                    ],
                                    "Average Latency (ms)": [average_latency],
                                }
                            )
                            latency_figure = px.bar(
                                latency_chart,
                                x="Provider",
                                y="Average Latency (ms)",
                                text_auto=True,
                                color_discrete_sequence=["#34d399"],
                            )
                            st.plotly_chart(
                                _style_evaluation_figure(latency_figure),
                                key="eval_latency_chart",
                                **width_kwargs,
                            )
            else:
                st.markdown("##### Offline Case Outcomes")
                outcome_counts = (
                    evaluation["results"]["Final Status"]
                    .value_counts()
                    .rename_axis("Final Status")
                    .reset_index(name="Cases")
                )
                if not outcome_counts.empty:
                    outcome_chart = px.bar(
                        outcome_counts,
                        x="Cases",
                        y="Final Status",
                        orientation="h",
                        color="Final Status",
                        text_auto=True,
                        color_discrete_sequence=["#34d399", "#e0bd79"],
                    )
                    st.plotly_chart(
                        _style_evaluation_figure(outcome_chart),
                        key="eval_offline_outcome_chart",
                        **width_kwargs,
                    )

            title_col, reset_col = st.columns([1.55, 1.0])
            with title_col:
                st.markdown("##### Detailed Per-Case Results")
            with reset_col:
                if st.button(
                    "↻ Reset Hidden Columns",
                    key="reset_eval_hidden_columns",
                    **width_kwargs,
                ):
                    st.session_state.eval_table_rev += 1
                    st.rerun()
            st.dataframe(
                evaluation["results"],
                hide_index=True,
                key=f"eval_per_case_df_{st.session_state.eval_table_rev}",
                **width_kwargs,
            )
            export_col1, export_col2 = st.columns(2)
            with export_col1:
                st.download_button(
                    "📥 Download Evaluation Report (CSV)",
                    data=export_results_csv(evaluation),
                    file_name="evaluation_report.csv",
                    mime="text/csv",
                    key="download_eval_csv",
                    **width_kwargs,
                )
            with export_col2:
                st.download_button(
                    "📥 Download Evaluation Metrics (JSON)",
                    data=export_metrics_json(
                        evaluation,
                        offline_security_metrics=st.session_state.offline_security_metrics,
                    ),
                    file_name="evaluation_metrics.json",
                    mime="application/json",
                    key="download_eval_json",
                    **width_kwargs,
                )


if __name__ == "__main__":
    main()
