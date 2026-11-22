"""
Chat-with-your-CSV: Enterprise Natural-Language Tabular Data Analyst
Upgraded with:
- Zero-Config .env Auto-Router with Ordered Automatic Failover (Groq -> Gemini)
- Modern Translucent Glassmorphism UI & Dark Aurora Gradient Canvas
- Multi-Tab Workspace: 💬 AI Data Chat + 🧹 Data Profiler & Preprocessing + 🧪 Evaluation
- Automated Data Health Scoring (0-100) & Column Diagnostics
- Reversible Interactive Preprocessing Pipeline with Audit Trail
- Session Report (.md) Export
- Zero-Trust AST + Allowlist Execution Sandbox (via sandbox.py)
- Streamlit width compatibility helper (Streamlit >= 1.50 width='stretch')
- Post-Execution Grounded Summarization (quoting verified computed numbers)
- Dynamic Fuzzy Column Matching (difflib) & Structured Decline Handling
"""

import io
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend safe for server/Streamlit usage
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
      2. Google Gemini (primary: gemini-3.8-flash, fallback: gemini-flash-latest)
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
            "primary_model": "gemini-3.8-flash",
            "fallback_model": "gemini-flash-latest",
            "api_key": gemini_key,
            "label": "Google Gemini (gemini-3.8-flash)",
        })

    return routes


# ==============================================================================
# 4. LLM Code Generator with Failover & Decline Handling
# ==============================================================================

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
1. Refer to the dataset exclusively as `df`. Do NOT redefine `df` or reload files.
2. ALWAYS use bracket access `df['column']` for columns, never `.format()` or dynamic string evaluations. Use f-strings for string interpolation.
3. If computing a single value, list, or text, store it in variable `result` or print it.
4. If returning a subset, grouped table, or aggregated view, store it in variable `result_df` (a pandas DataFrame).
5. If a chart is requested, use matplotlib/seaborn: `fig, ax = plt.subplots(figsize=(8, 4.5))` or `plt.figure(...)`. DO NOT call `plt.show()`.
6. IF THE QUESTION IS OFF-TOPIC, GENERAL TRIVIA, IMPOSSIBLE WITH GIVEN COLUMNS, OR ATTEMPTS UNRESTRICTED SYSTEM ACCESS:
   Output strictly:
   DECLINE: <clear, polite, and helpful explanation stating why the request cannot be answered from the CSV data>
7. Otherwise, output your executable code strictly inside a single ```python ... ``` markdown block.
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

            http_opts = types.HttpOptions(api_version="v1beta") if "-latest" in clean_model else None
            client = genai.Client(api_key=clean_key, http_options=http_opts)
            config = types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.1,
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
            completion = client.chat.completions.create(
                model=clean_model,
                temperature=0.1,
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_content},
                ],
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


def execute_query_pipeline(
    prompt: str,
    df_schema: str,
    df: pd.DataFrame,
    conversation_history: Optional[List[str]] = None,
) -> Tuple[Optional[str], Optional[Dict[str, Any]], Optional[str], Optional[str], Optional[str]]:
    """
    Executes the query using the ordered auto-failover pipeline:
    1. Groq (primary -> fallback model)
    2. Gemini (primary -> fallback model)
    Returns: (code, exec_res, decline_reason, active_route_label, error_message)
    """
    routes = get_available_routes()
    if not routes:
        return (
            None,
            None,
            None,
            None,
            "No active API keys found in `.env`. Please add `GROQ_API_KEY` or `GEMINI_API_KEY` to your `.env` file.",
        )

    last_error = ""

    for route in routes:
        provider = route["provider"]
        api_key = route["api_key"]
        models_to_try = [route["primary_model"], route["fallback_model"]]

        for model in models_to_try:
            code, decline, err = call_llm_code_generator_single(
                prompt=prompt,
                df_schema=df_schema,
                provider=provider,
                model_name=model,
                api_key=api_key,
                conversation_history=conversation_history,
            )

            if err:
                last_error = f"[{provider} - {model}] {err}"
                continue  # Try fallback model or next route

            if decline:
                return None, None, decline, route["label"], None

            if code:
                # Execute in Zero-Trust Sandbox
                exec_res = execute_pandas_code(code, df)

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

                    if fixed_decline:
                        return None, None, fixed_decline, route["label"], None

                    if fixed_code and not retry_err:
                        current_code = fixed_code
                        exec_res = execute_pandas_code(current_code, df)
                    else:
                        break

                return current_code, exec_res, None, route["label"], None

    return None, None, None, None, f"All AI provider routes failed. Last error: {last_error}"


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
    has_chart = exec_res.get("figure_png") is not None

    facts_parts = []
    if result_val is not None:
        facts_parts.append(f"Result Value: {result_val}")
    if result_df is not None:
        facts_parts.append(f"Result Table Preview ({len(result_df)} total rows):\n{result_df.head(5).to_string(index=False)}")
    if stdout:
        facts_parts.append(f"Console Output:\n{stdout}")
    if has_chart:
        facts_parts.append("Matplotlib Chart: Successfully rendered.")

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
Cite the exact computed numbers or findings from the ACTUAL COMPUTED RESULTS above.
Do NOT guess, extrapolate, or invent any numbers not present in the results.
"""

    routes = get_available_routes()
    for route in routes:
        clean_key = route["api_key"]
        model = route["primary_model"]
        try:
            if route["provider"] == "Groq":
                from openai import OpenAI

                client = OpenAI(api_key=clean_key, base_url="https://api.groq.com/openai/v1")
                completion = client.chat.completions.create(
                    model=model,
                    temperature=0.2,
                    messages=[{"role": "user", "content": summary_prompt}],
                    max_tokens=250,
                )
                grounded_text = completion.choices[0].message.content or ""
                if grounded_text.strip():
                    return grounded_text.strip()

            elif route["provider"] == "Google Gemini":
                from google import genai
                from google.genai import types

                http_opts = types.HttpOptions(api_version="v1beta") if "-latest" in model else None
                client = genai.Client(api_key=clean_key, http_options=http_opts)
                response = client.models.generate_content(
                    model=model,
                    contents=summary_prompt,
                    config=types.GenerateContentConfig(temperature=0.2),
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

        /* Dark Aurora Radial Mesh Canvas */
        .stApp {
            background-color: #070b14;
            background-image: 
                radial-gradient(circle at 12% 12%, rgba(99, 102, 241, 0.12) 0%, transparent 40%),
                radial-gradient(circle at 88% 18%, rgba(168, 85, 247, 0.10) 0%, transparent 42%),
                radial-gradient(circle at 50% 88%, rgba(6, 182, 212, 0.08) 0%, transparent 45%);
            background-attachment: fixed;
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
            color: #f1f5f9;
        }

        /* Sidebar Styling (Frosted Glass) */
        [data-testid="stSidebar"] {
            background-color: rgba(15, 23, 42, 0.70) !important;
            backdrop-filter: blur(20px) !important;
            -webkit-backdrop-filter: blur(20px) !important;
            border-right: 1px solid rgba(255, 255, 255, 0.07) !important;
        }

        /* Header Hero Container */
        .hero-container {
            padding: 1.2rem 1.6rem;
            border-radius: 16px;
            background: rgba(15, 23, 42, 0.60);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.36);
            margin-bottom: 1.2rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .hero-title {
            font-size: 2.1rem;
            font-weight: 700;
            background: linear-gradient(135deg, #a5b4fc 0%, #c084fc 50%, #38bdf8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 0;
            padding-bottom: 0.2rem;
        }
        .hero-subtitle {
            color: #94a3b8;
            font-size: 0.95rem;
            margin: 0;
            font-weight: 400;
        }

        /* Backend Engine Status Pill */
        .status-pill {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 0.4rem 0.9rem;
            border-radius: 9999px;
            font-size: 0.8rem;
            font-weight: 600;
            background: rgba(16, 185, 129, 0.12);
            color: #34d399;
            border: 1px solid rgba(52, 211, 153, 0.3);
            box-shadow: 0 0 12px rgba(16, 185, 129, 0.15);
        }
        .status-pill-alert {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 0.4rem 0.9rem;
            border-radius: 9999px;
            font-size: 0.8rem;
            font-weight: 600;
            background: rgba(239, 68, 68, 0.12);
            color: #f87171;
            border: 1px solid rgba(248, 113, 113, 0.3);
        }

        /* 4 Glassmorphism KPI Metric Cards */
        .kpi-card {
            background: rgba(30, 41, 59, 0.50);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 0.9rem 1.1rem;
            margin-bottom: 0.8rem;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
            transition: transform 0.2s ease, border-color 0.2s ease;
        }
        .kpi-card:hover {
            transform: translateY(-2px);
            border-color: rgba(129, 140, 248, 0.35);
        }
        .kpi-label {
            font-size: 0.78rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #94a3b8;
            margin-bottom: 0.2rem;
        }
        .kpi-value {
            font-size: 1.45rem;
            font-weight: 700;
            color: #f8fafc;
        }

        /* Chat Message Bubbles */
        .chat-bubble-user {
            background: rgba(99, 102, 241, 0.15);
            border: 1px solid rgba(129, 140, 248, 0.3);
            border-radius: 14px 14px 2px 14px;
            padding: 0.9rem 1.2rem;
            margin: 0.5rem 0;
            color: #f1f5f9;
        }
        .chat-bubble-assistant {
            background: rgba(15, 23, 42, 0.65);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 14px 14px 14px 2px;
            padding: 0.9rem 1.2rem;
            margin: 0.5rem 0;
            color: #e2e8f0;
        }

        /* Frosted native Streamlit alerts and bordered commentary panels */
        [data-testid="stAlert"] {
            background: rgba(30, 41, 59, 0.72);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 12px;
        }
        [data-testid="stVerticalBlockBorderWrapper"] {
            background: rgba(15, 23, 42, 0.45);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border-color: rgba(255, 255, 255, 0.12);
            border-radius: 14px;
        }

        /* Empty State */
        .empty-state {
            padding: 3rem 1.5rem;
            border-radius: 16px;
            border: 1px dashed rgba(255, 255, 255, 0.15);
            background: rgba(15, 23, 42, 0.40);
            backdrop-filter: blur(12px);
            text-align: center;
            margin: 1.5rem 0;
        }

        /* Route Tag */
        .route-tag {
            font-size: 0.72rem;
            font-weight: 600;
            color: #818cf8;
            padding: 0.15rem 0.5rem;
            border-radius: 6px;
            background: rgba(99, 102, 241, 0.15);
            border: 1px solid rgba(99, 102, 241, 0.25);
            display: inline-block;
            margin-top: 0.4rem;
        }

        /* Custom Tabs Styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background-color: rgba(15, 23, 42, 0.45);
            padding: 6px;
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.08);
            margin-bottom: 1rem;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px;
            color: #94a3b8;
            padding: 8px 18px;
            font-weight: 600;
        }
        .stTabs [aria-selected="true"] {
            background-color: rgba(99, 102, 241, 0.25) !important;
            color: #f1f5f9 !important;
            border: 1px solid rgba(129, 140, 248, 0.4) !important;
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
    if "last_uploaded_file_id" not in st.session_state:
        st.session_state.last_uploaded_file_id = None
    if "pending_prompt" not in st.session_state:
        st.session_state.pending_prompt = None
    if "evaluation_result" not in st.session_state:
        st.session_state.evaluation_result = None

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
                st.toast("Loaded Spotify sample dataset!", icon="✅")

        # Handle uploaded CSV without overwriting sample dataset on subsequent rerun
        if uploaded_file is not None:
            file_id = getattr(uploaded_file, "file_id", uploaded_file.name)
            if st.session_state.last_uploaded_file_id != file_id or st.session_state.data_source == "upload":
                if st.session_state.df_name != uploaded_file.name:
                    file_bytes = uploaded_file.getvalue()
                    loaded_df = load_csv_data_cached(file_bytes, uploaded_file.name)
                    st.session_state.raw_df = loaded_df.copy()
                    st.session_state.df = loaded_df.copy()
                    st.session_state.df_name = uploaded_file.name
                    st.session_state.data_source = "upload"
                    st.session_state.last_uploaded_file_id = file_id
                    st.session_state.preprocessing_logs = []
                    st.session_state.messages = []
                    st.session_state["ai_eda_insight"] = None
                    st.toast(f"Loaded {uploaded_file.name}", icon="📁")

        # Keep raw_df synchronized if not yet set
        if st.session_state.df is not None and st.session_state.raw_df is None:
            st.session_state.raw_df = st.session_state.df.copy()

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
            st.rerun()

        # 4. Session Report Export (.md)
        if st.session_state.df is not None:
            st.divider()
            st.markdown("#### 📄 Session Report")
            profile_for_rep = compute_data_profile(st.session_state.df)
            report_md_str = export_session_report_markdown(
                df_name=st.session_state.df_name,
                df=st.session_state.df,
                profile=profile_for_rep,
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

    # --- Main Hero Header ---
    status_html = (
        f'<span class="status-pill">● {" + ".join(route_labels)} Active</span>'
        if has_active_engine
        else '<span class="status-pill-alert">⚠️ Set keys in .env</span>'
    )

    st.markdown(
        f"""
        <div class="hero-container">
            <div>
                <h1 class="hero-title">Chat-with-your-CSV</h1>
                <p class="hero-subtitle">Zero-Trust Tabular Intelligence • Sandboxed Execution • Grounded Answers</p>
            </div>
            <div>
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
                st.rerun()
        st.stop()
        return

    # --- State 2: Data Loaded - Display 4 Frosted Glass KPI Cards ---
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Active Dataset</div>
                <div class="kpi-value" style="font-size: 1.15rem; word-break: break-all;">{st.session_state.df_name}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Total Rows</div>
                <div class="kpi-value">{df.shape[0]:,}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Columns</div>
                <div class="kpi-value">{df.shape[1]:,}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col4:
        missing_count = int(df.isnull().sum().sum())
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Missing Cells</div>
                <div class="kpi-value">{missing_count:,}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

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
        # Collapsible Data Inspector
        with st.expander("🔍 Inspect Uploaded Data (Preview, Schema & Statistics)", expanded=False):
            t_tab1, t_tab2, t_tab3 = st.tabs(["📋 Data Preview", "🏷️ Column Types & Nulls", "📊 Summary Statistics"])

            with t_tab1:
                st.dataframe(df.head(10), **width_kwargs)

            with t_tab2:
                row_count = max(len(df), 1)  # Guard against 0-row DataFrames
                col_summary = pd.DataFrame({
                    "Column Name": df.columns,
                    "Data Type": [str(t) for t in df.dtypes],
                    "Non-Null Count": df.notnull().sum().values,
                    "Null Count": df.isnull().sum().values,
                    "Null Percentage (%)": ((df.isnull().sum() / row_count) * 100).round(2).values,
                    "Unique Values": [df[c].nunique() for c in df.columns],
                })
                st.dataframe(col_summary, hide_index=True, **width_kwargs)

            with t_tab3:
                st.dataframe(df.describe(include="all").T, **width_kwargs)

        # Suggested Prompts Chips
        st.markdown("##### 💡 Suggested Questions:")
        prompt_cols = st.columns(4)
        suggestions = generate_suggested_prompts(df)
        for i, suggestion in enumerate(suggestions):
            with prompt_cols[i % len(prompt_cols)]:
                if st.button(suggestion, key=f"sugg_{i}", **width_kwargs):
                    st.session_state.pending_prompt = suggestion
                    st.rerun()

        st.divider()

        # Render Message History
        for idx, msg in enumerate(st.session_state.messages):
            role = msg.get("role", "user")
            with st.chat_message(role, avatar="🧑‍💻" if role == "user" else "🤖"):
                status = msg.get("status", "success")

                if status == "error":
                    cat = msg.get("failure_category", "Error")
                    st.error(f"⚠️ **{cat}**\n\n{msg.get('content', '')}")
                elif status == "declined":
                    st.warning(f"🛡️ **Request Declined**\n\n{msg.get('content', '')}")
                else:
                    st.markdown(msg.get("content", ""))

                # Route badge
                if msg.get("route_label"):
                    st.markdown(f'<span class="route-tag">Engine: {msg["route_label"]}</span>', unsafe_allow_html=True)

                # View Generated Python Query
                if msg.get("code") and show_code:
                    with st.expander("💻 View Generated Python / Pandas Query"):
                        st.code(msg["code"], language="python")

                # Console Output
                if msg.get("stdout"):
                    st.caption(f"Console Output:\n{msg['stdout']}")

                # Result Value
                if msg.get("result") is not None:
                    st.write(msg["result"])

                # Result DataFrame Table
                if msg.get("result_df") is not None:
                    st.dataframe(msg["result_df"], **width_kwargs)
                    csv_data = msg["result_df"].to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="📥 Download Result Table as CSV",
                        data=csv_data,
                        file_name="query_result.csv",
                        mime="text/csv",
                        key=f"dl_csv_{idx}",
                    )

                # Matplotlib Chart PNG
                if msg.get("figure_png") is not None:
                    st.image(msg["figure_png"], **width_kwargs)
                    st.download_button(
                        label="📥 Download Chart (PNG)",
                        data=msg["figure_png"],
                        file_name="chart.png",
                        mime="image/png",
                        key=f"dl_png_{idx}",
                    )

        # Handle New User Input
        user_input = st.chat_input("Ask a question about your data in plain English...")

        # Handle clicked suggestion prompt
        if st.session_state.pending_prompt:
            user_input = st.session_state.pending_prompt
            st.session_state.pending_prompt = None

        if user_input:
            if not has_active_engine:
                st.error("⚠️ No active AI engine found. Please add `GROQ_API_KEY` or `GEMINI_API_KEY` to your `.env` file.")
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

                    # Collect last 4 turns of context
                    recent_history: List[str] = []
                    for m in st.session_state.messages[-4:]:
                        r = m.get("role")
                        c = m.get("content", "")
                        cd = m.get("code")
                        if r == "user":
                            recent_history.append(f"User: {c}")
                        elif r == "assistant":
                            if cd:
                                recent_history.append(f"Assistant Query:\n```python\n{cd}\n```\nAnswer: {c}")
                            else:
                                recent_history.append(f"Assistant: {c}")

                    # Execute via Ordered Auto-Failover Pipeline
                    code, exec_res, decline, active_label, pipeline_err = execute_query_pipeline(
                        prompt=user_input,
                        df_schema=schema_context,
                        df=df,
                        conversation_history=recent_history,
                    )

                    if pipeline_err:
                        st.error(f"⚠️ **Pipeline Error**\n\n{pipeline_err}")
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": pipeline_err,
                            "status": "error",
                            "failure_category": "Provider/API Error",
                        })

                    elif decline:
                        st.warning(f"🛡️ **Request Declined**\n\n{decline}")
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": decline,
                            "status": "declined",
                            "route_label": active_label,
                        })

                    elif exec_res and exec_res.get("error"):
                        cat, friendly_msg = classify_failure(exec_res["error"], df)
                        st.error(f"⚠️ **{cat}**\n\n{friendly_msg}")
                        if code and show_code:
                            with st.expander("💻 View Attempted Code"):
                                st.code(code, language="python")

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": friendly_msg,
                            "code": code,
                            "status": "error",
                            "failure_category": cat,
                            "route_label": active_label,
                        })

                    elif exec_res:
                        # Grounded answer quoting real computed facts
                        grounded_answer = generate_grounded_answer(
                            prompt=user_input,
                            code=code,
                            exec_res=exec_res,
                        )

                        st.markdown(grounded_answer)

                        if active_label:
                            st.markdown(f'<span class="route-tag">Engine: {active_label}</span>', unsafe_allow_html=True)

                        if exec_res.get("stdout"):
                            st.caption(f"Console Output:\n{exec_res['stdout']}")

                        if code and show_code:
                            with st.expander("💻 View Generated Python / Pandas Query"):
                                st.code(code, language="python")

                        if exec_res.get("result") is not None:
                            st.write(exec_res["result"])

                        if exec_res.get("result_df") is not None:
                            st.dataframe(exec_res["result_df"], **width_kwargs)
                            csv_data = exec_res["result_df"].to_csv(index=False).encode("utf-8")
                            st.download_button(
                                label="📥 Download Result Table as CSV",
                                data=csv_data,
                                file_name="query_result.csv",
                                mime="text/csv",
                                key=f"dl_csv_{len(st.session_state.messages)}",
                            )

                        if exec_res.get("figure_png") is not None:
                            st.image(exec_res["figure_png"], **width_kwargs)
                            st.download_button(
                                label="📥 Download Chart (PNG)",
                                data=exec_res["figure_png"],
                                file_name="chart.png",
                                mime="image/png",
                                key=f"dl_png_{len(st.session_state.messages)}",
                            )

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": grounded_answer,
                            "code": code,
                            "result": exec_res.get("result"),
                            "result_df": exec_res.get("result_df"),
                            "figure_png": exec_res.get("figure_png"),
                            "stdout": exec_res.get("stdout"),
                            "route_label": active_label,
                            "status": "success",
                        })

    # ==========================================================================
    # TAB 2: Data Profiler & Preprocessing
    # ==========================================================================
    with tab_profiler:
        profile = compute_data_profile(df)

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
            st.download_button(
                label="📥 Download Cleaned CSV",
                data=cleaned_csv_bytes,
                file_name=f"cleaned_{st.session_state.df_name}",
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
                routes = get_available_routes()
                if routes:
                    route = routes[0]
                    deterministic_insights = "\n".join(f"- {bullet}" for bullet in eda_bullets)
                    prompt = f"""You are a principal data scientist. Based STRICTLY on the computed data profile metrics below, provide a high-level executive briefing (3 concise paragraphs) detailing key statistical takeaways, potential data risks, and recommended modeling strategies. DO NOT hallucinate.

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
                    try:
                        code, ai_insight, err = call_llm_code_generator_single(
                            prompt=prompt,
                            df_schema="",
                            provider=route["provider"],
                            model_name=route["primary_model"],
                            api_key=route["api_key"],
                        )
                        st.session_state["ai_eda_insight"] = ai_insight or code or err
                    except Exception as ex:
                        st.session_state["ai_eda_insight"] = f"AI briefing unavailable: {ex}"
                else:
                    st.session_state["ai_eda_insight"] = "No API keys configured in .env for AI briefing."

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
                    st.session_state.evaluation_result = run_offline_security_eval(df)

        with live_col:
            if st.button(
                "⚡ Run Live Gen-AI Benchmark (Groq vs Gemini / Fast Subset)",
                key="run_live_eval",
                **width_kwargs,
            ):
                routes = get_available_routes()
                try:
                    with st.spinner("Running live provider benchmark; this may use API quota..."):
                        if provider_choice == "Groq vs Google Gemini":
                            result = compare_providers_eval(
                                df=df,
                                routes=routes,
                                generate_pandas_code=call_llm_code_generator_single,
                                suite_mode=suite_mode,
                            )
                        else:
                            result = run_live_llm_eval(
                                df=df,
                                routes=routes,
                                generate_pandas_code=call_llm_code_generator_single,
                                provider=provider_choice,
                                suite_mode=suite_mode,
                            )
                    st.session_state.evaluation_result = result
                except ValueError as ex:
                    st.error(str(ex))

        evaluation = st.session_state.get("evaluation_result")
        if evaluation:
            metrics = evaluation["metrics"]
            is_offline = evaluation["mode"].startswith("Offline")
            average_latency = (
                metrics.get("average_sandbox_latency_ms")
                if is_offline
                else metrics.get("average_latency_ms")
            )

            def format_kpi(value: Optional[float], suffix: str = "%") -> str:
                return "N/A" if value is None else f"{value:,.1f}{suffix}"

            st.markdown(f"#### Latest Run: {evaluation['mode']}")
            metric_cards = [
                ("Security Block Rate", format_kpi(metrics.get("security_block_rate_pct"))),
                ("False-Positive Rate", format_kpi(metrics.get("false_positive_rate_pct"))),
                ("Pass@1 Accuracy", format_kpi(metrics.get("pass_at_1_accuracy_pct"))),
                (
                    "Self-Healed Pass@3",
                    format_kpi(metrics.get("self_healed_pass_at_3_accuracy_pct")),
                ),
                ("Avg Latency (ms)", format_kpi(average_latency, " ms")),
            ]
            metric_columns = st.columns(5)
            for column, (label, value) in zip(metric_columns, metric_cards):
                with column:
                    st.markdown(
                        f"""
                        <div class="kpi-card">
                            <div class="kpi-label">{label}</div>
                            <div class="kpi-value">{value}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            if not is_offline:
                comparison = evaluation.get("provider_comparison")
                if isinstance(comparison, pd.DataFrame) and not comparison.empty:
                    chart_col1, chart_col2 = st.columns(2)
                    with chart_col1:
                        st.markdown("##### Pass@1 vs Self-Healed Pass@3")
                        pass_chart = comparison.set_index("Provider")[
                            ["Pass@1 Accuracy (%)", "Self-Healed Pass@3 Accuracy (%)"]
                        ]
                        st.bar_chart(pass_chart, **width_kwargs)
                    with chart_col2:
                        st.markdown("##### Provider Latency Comparison")
                        latency_chart = comparison.set_index("Provider")[["Average Latency (ms)"]]
                        st.bar_chart(latency_chart, **width_kwargs)
                else:
                    chart_col1, chart_col2 = st.columns(2)
                    with chart_col1:
                        st.markdown("##### Pass@1 vs Self-Healed Pass@3")
                        pass_chart = pd.DataFrame(
                            {
                                "Accuracy (%)": [
                                    metrics.get("pass_at_1_accuracy_pct"),
                                    metrics.get("self_healed_pass_at_3_accuracy_pct"),
                                ]
                            },
                            index=["Pass@1", "Self-Healed Pass@3"],
                        ).dropna()
                        if not pass_chart.empty:
                            st.bar_chart(pass_chart, **width_kwargs)
                    with chart_col2:
                        st.markdown("##### Average End-to-End Latency")
                        if average_latency is not None:
                            latency_chart = pd.DataFrame(
                                {"Average Latency (ms)": [average_latency]},
                                index=[metrics.get("provider", provider_choice)],
                            )
                            st.bar_chart(latency_chart, **width_kwargs)
            else:
                st.markdown("##### Offline Case Outcomes")
                outcome_counts = evaluation["results"]["Final Status"].value_counts()
                if not outcome_counts.empty:
                    st.bar_chart(outcome_counts, **width_kwargs)

            st.markdown("##### Detailed Per-Case Results")
            st.dataframe(evaluation["results"], hide_index=True, **width_kwargs)
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
                    data=export_metrics_json(evaluation),
                    file_name="evaluation_metrics.json",
                    mime="application/json",
                    key="download_eval_json",
                    **width_kwargs,
                )


if __name__ == "__main__":
    main()
