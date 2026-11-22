"""
Data Profiler & Preprocessing Pipeline for Chat-with-your-CSV.
Provides automated data health scoring, column diagnostics, IQR outlier detection,
deterministic EDA insights, reversible data cleaning transformations, and Session Report generation.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ==============================================================================
# 1. Comprehensive Data Profiling & Health Scoring
# ==============================================================================

def compute_data_profile(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes an automated data health score (0-100), summary statistics,
    per-column diagnostics, and top Pearson correlation pairs.
    Handles 0-row and empty DataFrames safely without ZeroDivisionError.
    """
    n_rows = int(df.shape[0])
    n_cols = int(df.shape[1])
    total_cells = n_rows * n_cols

    # Handle 0-row or empty DataFrames safely
    if n_rows == 0 or n_cols == 0:
        return {
            "health_score": 100.0 if n_cols > 0 else 0.0,
            "rows": n_rows,
            "cols": n_cols,
            "missing_cells": 0,
            "missing_pct": 0.0,
            "duplicate_rows": 0,
            "duplicate_pct": 0.0,
            "constant_cols": 0,
            "numeric_cols": 0,
            "categorical_cols": 0,
            "total_outliers": 0,
            "column_report": pd.DataFrame(columns=[
                "Column", "Type", "Non-Null", "Null %", "Unique", "Skewness", "IQR Outliers", "Detected Issues"
            ]),
            "top_correlations": [],
        }

    # Summary counts
    missing_cells = int(df.isnull().sum().sum())
    missing_pct = round((missing_cells / total_cells) * 100, 2) if total_cells > 0 else 0.0
    duplicate_rows = int(df.duplicated().sum())
    duplicate_pct = round((duplicate_rows / n_rows) * 100, 2) if n_rows > 0 else 0.0

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols]
    numeric_cols_count = len(num_cols)
    categorical_cols_count = len(cat_cols)

    constant_cols_list = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    constant_cols_count = len(constant_cols_list)
    constant_pct = (constant_cols_count / n_cols) * 100 if n_cols > 0 else 0.0

    # Column-level diagnostics & Outlier calculation
    column_records: List[Dict[str, Any]] = []
    total_outliers = 0
    total_numeric_values = n_rows * len(num_cols)

    for col in df.columns:
        dtype_str = str(df[col].dtype)
        non_null_count = int(df[col].notnull().sum())
        null_count = int(df[col].isnull().sum())
        null_pct = round((null_count / n_rows) * 100, 2)
        unique_count = int(df[col].nunique())

        is_num = col in num_cols
        skew_val: Optional[float] = None
        col_outliers: Optional[int] = None
        issues: List[str] = []

        if null_count > 0:
            issues.append(f"Missing ({null_pct}%)")

        if unique_count <= 1:
            issues.append("Constant")

        if is_num:
            # Skewness
            if non_null_count >= 3:
                try:
                    s_val = float(df[col].skew())
                    if not np.isnan(s_val):
                        skew_val = round(s_val, 2)
                        if abs(s_val) > 1.5:
                            issues.append(f"High Skew ({skew_val})")
                except Exception:
                    pass

            # IQR Outliers (excluding binary 0/1 indicator columns)
            unique_vals = df[col].dropna().unique()
            is_binary = len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0})

            if not is_binary and non_null_count >= 4:
                q25 = float(df[col].quantile(0.25))
                q75 = float(df[col].quantile(0.75))
                iqr = q75 - q25
                if iqr > 0:
                    lower_bound = q25 - 1.5 * iqr
                    upper_bound = q75 + 1.5 * iqr
                    outliers_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                    col_outliers = int(outliers_mask.sum())
                    if col_outliers > 0:
                        total_outliers += col_outliers
                        issues.append(f"{col_outliers} Outliers")
                else:
                    col_outliers = 0
            elif is_binary:
                col_outliers = 0

        detected_issues_str = ", ".join(issues) if issues else "Clean"

        column_records.append({
            "Column": str(col),
            "Type": dtype_str,
            "Non-Null": non_null_count,
            "Null %": null_pct,
            "Unique": unique_count,
            "Skewness": skew_val if skew_val is not None else "-",
            "IQR Outliers": col_outliers if col_outliers is not None else "-",
            "Detected Issues": detected_issues_str,
        })

    # Data Health Score calculation (0 - 100)
    # Penalties for missing data, duplicates, constant columns, and outliers
    penalty_missing = missing_pct * 0.45
    penalty_duplicates = duplicate_pct * 0.30
    penalty_constants = constant_pct * 0.15
    outlier_ratio = (total_outliers / max(total_numeric_values, 1)) * 100
    penalty_outliers = min(outlier_ratio * 1.5, 15.0)

    raw_score = 100.0 - (penalty_missing + penalty_duplicates + penalty_constants + penalty_outliers)
    health_score = round(max(0.0, min(100.0, raw_score)), 1)

    # Top 5 strongest Pearson correlations (excluding self-correlations)
    top_correlations: List[Dict[str, Any]] = []
    if len(num_cols) >= 2:
        try:
            corr_df = df[num_cols].corr(method="pearson")
            pairs = []
            for i in range(len(num_cols)):
                for j in range(i + 1, len(num_cols)):
                    c1, c2 = num_cols[i], num_cols[j]
                    val = corr_df.loc[c1, c2]
                    if not np.isnan(val):
                        pairs.append({
                            "feature_1": c1,
                            "feature_2": c2,
                            "correlation": round(float(val), 3),
                            "abs_correlation": round(abs(float(val)), 3),
                        })
            pairs.sort(key=lambda x: x["abs_correlation"], reverse=True)
            top_correlations = pairs[:5]
        except Exception:
            top_correlations = []

    return {
        "health_score": health_score,
        "rows": n_rows,
        "cols": n_cols,
        "missing_cells": missing_cells,
        "missing_pct": missing_pct,
        "duplicate_rows": duplicate_rows,
        "duplicate_pct": duplicate_pct,
        "constant_cols": constant_cols_count,
        "numeric_cols": numeric_cols_count,
        "categorical_cols": categorical_cols_count,
        "total_outliers": total_outliers,
        "column_report": pd.DataFrame(column_records),
        "top_correlations": top_correlations,
    }


# ==============================================================================
# 2. Reversible Data Preprocessing Pipeline
# ==============================================================================

def apply_preprocessing(
    df: pd.DataFrame,
    drop_duplicates: bool = False,
    strip_whitespace: bool = False,
    impute_strategy: str = "none",
    outlier_strategy: str = "none",
    drop_constant_cols: bool = False,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Applies configurable cleaning and preprocessing steps to a copy of df.
    Returns: (cleaned_df, audit_logs)
    """
    df_clean = df.copy()
    audit_logs: List[str] = []

    # 1. Drop exact duplicate rows
    if drop_duplicates:
        dup_count = int(df_clean.duplicated().sum())
        if dup_count > 0:
            df_clean = df_clean.drop_duplicates().reset_index(drop=True)
            audit_logs.append(f"Dropped {dup_count} exact duplicate rows ({len(df_clean)} rows remaining).")
        else:
            audit_logs.append("Checked for duplicate rows (0 duplicates found).")

    # 2. Trim whitespace on text columns & convert empty strings to NaN
    if strip_whitespace:
        text_cols = df_clean.select_dtypes(include=["object", "string"]).columns
        trimmed_count = 0
        for col in text_cols:
            original = df_clean[col]
            # Strip string values while keeping actual NaNs intact
            stripped = original.astype(str).str.strip().replace("", np.nan)
            df_clean[col] = stripped.where(original.notnull(), np.nan)
            trimmed_count += 1
        if trimmed_count > 0:
            audit_logs.append(f"Trimmed whitespace across {trimmed_count} text column(s) and converted empty strings to null.")

    # 3. Missing Value Imputation
    if impute_strategy == "drop_rows":
        null_rows = int(df_clean.isnull().any(axis=1).sum())
        if null_rows > 0:
            df_clean = df_clean.dropna().reset_index(drop=True)
            audit_logs.append(f"Dropped {null_rows} row(s) with missing cells ({len(df_clean)} rows remaining).")
        else:
            audit_logs.append("Drop missing rows checked (0 null rows found).")

    elif impute_strategy in ("median_mode", "mean_mode"):
        imputed_details: List[str] = []
        for col in df_clean.columns:
            if df_clean[col].isnull().sum() > 0:
                if np.issubdtype(df_clean[col].dtype, np.number):
                    fill_val = df_clean[col].median() if impute_strategy == "median_mode" else df_clean[col].mean()
                    if not np.isnan(fill_val):
                        df_clean[col] = df_clean[col].fillna(fill_val)
                        stat_name = "median" if impute_strategy == "median_mode" else "mean"
                        imputed_details.append(f"'{col}' ({stat_name}: {round(float(fill_val), 3)})")
                else:
                    mode_series = df_clean[col].mode()
                    if not mode_series.empty:
                        mode_val = mode_series.iloc[0]
                        df_clean[col] = df_clean[col].fillna(mode_val)
                        imputed_details.append(f"'{col}' (mode: '{mode_val}')")
        if imputed_details:
            strat_label = "median/mode" if impute_strategy == "median_mode" else "mean/mode"
            audit_logs.append(f"Imputed missing values via {strat_label} in: {', '.join(imputed_details)}.")

    # 4. Outlier Handling (Clip or Remove IQR, excluding binary 0/1 columns)
    if outlier_strategy == "clip_iqr":
        clipped_details: List[str] = []
        num_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            unique_vals = df_clean[col].dropna().unique()
            # Skip binary 0/1 flags
            if len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0}):
                continue

            q25 = df_clean[col].quantile(0.25)
            q75 = df_clean[col].quantile(0.75)
            iqr = q75 - q25
            if iqr > 0:
                lower_b = q25 - 1.5 * iqr
                upper_b = q75 + 1.5 * iqr
                outliers_count = int(((df_clean[col] < lower_b) | (df_clean[col] > upper_b)).sum())
                if outliers_count > 0:
                    df_clean[col] = df_clean[col].clip(lower=lower_b, upper=upper_b)
                    clipped_details.append(f"'{col}' ({outliers_count} clipped to [{round(lower_b, 2)}, {round(upper_b, 2)}])")

        if clipped_details:
            audit_logs.append(f"Winsorized/clipped IQR outliers in: {', '.join(clipped_details)}.")

    elif outlier_strategy == "remove_iqr":
        num_cols = df_clean.select_dtypes(include=[np.number]).columns
        outlier_mask = pd.Series(False, index=df_clean.index)
        for col in num_cols:
            unique_vals = df_clean[col].dropna().unique()
            if len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0}):
                continue
            q25 = df_clean[col].quantile(0.25)
            q75 = df_clean[col].quantile(0.75)
            iqr = q75 - q25
            if iqr > 0:
                lower_b = q25 - 1.5 * iqr
                upper_b = q75 + 1.5 * iqr
                outlier_mask = outlier_mask | (df_clean[col] < lower_b) | (df_clean[col] > upper_b)

        removed_count = int(outlier_mask.sum())
        if removed_count > 0:
            df_clean = df_clean[~outlier_mask].reset_index(drop=True)
            audit_logs.append(f"Removed {removed_count} row(s) containing IQR outliers ({len(df_clean)} rows remaining).")

    # 5. Drop Constant Columns
    if drop_constant_cols:
        const_cols = [c for c in df_clean.columns if df_clean[c].nunique(dropna=False) <= 1]
        if const_cols:
            df_clean = df_clean.drop(columns=const_cols)
            col_list_str = ", ".join([f"'{c}'" for c in const_cols])
            audit_logs.append(f"Dropped {len(const_cols)} constant column(s): {col_list_str}.")

    if not audit_logs:
        audit_logs.append("No transformations applied; dataset remains unchanged.")

    return df_clean, audit_logs


# ==============================================================================
# 3. Deterministic Auto-EDA Insights
# ==============================================================================

def build_deterministic_eda_insights(df: pd.DataFrame, profile: Dict[str, Any]) -> List[str]:
    """
    Generates 4-5 crisp, factual observations about the dataset without requiring an LLM call.
    """
    insights: List[str] = []

    # 1. Dimensions & Overall Health
    insights.append(
        f"**Dataset Scale & Quality:** The dataset contains **{profile['rows']:,} rows** and **{profile['cols']} columns** with an overall Data Health Score of **{profile['health_score']}/100**."
    )

    # 2. Missing & Duplication Status
    if profile["missing_cells"] == 0 and profile["duplicate_rows"] == 0:
        insights.append(
            "**Data Completeness:** **Zero missing values** and **zero duplicate rows** detected — the dataset is structurally complete."
        )
    else:
        insights.append(
            f"**Data Cleanliness:** Identified **{profile['missing_cells']:,} missing cells** ({profile['missing_pct']}%) and **{profile['duplicate_rows']:,} duplicate rows** requiring preprocessing attention."
        )

    # 3. Top Pearson Correlation
    if profile.get("top_correlations"):
        top_corr = profile["top_correlations"][0]
        direction = "positive" if top_corr["correlation"] > 0 else "negative"
        insights.append(
            f"**Primary Correlation:** Strongest relationship exists between **`{top_corr['feature_1']}`** and **`{top_corr['feature_2']}`** with a {direction} Pearson correlation of **r = {top_corr['correlation']:+.3f}**."
        )

    # 4. Binary Target Distribution (e.g. liked, target, mode)
    num_cols = df.select_dtypes(include=[np.number]).columns
    binary_cols = [c for c in num_cols if df[c].dropna().nunique() == 2]
    target_col = "liked" if "liked" in binary_cols else (binary_cols[0] if binary_cols else None)

    if target_col is not None and profile["rows"] > 0:
        vc = df[target_col].value_counts(normalize=True) * 100
        val_0 = vc.get(0, 0.0)
        val_1 = vc.get(1, 0.0)
        insights.append(
            f"**Target Distribution (`{target_col}`):** Binary target split is **{val_1:.1f}% class 1** vs **{val_0:.1f}% class 0**, providing a solid foundation for classification/comparative analysis."
        )

    # 5. Widest-spread or skewed feature
    if len(num_cols) > 0 and profile["rows"] >= 3:
        skews = {}
        for c in num_cols:
            try:
                s = df[c].skew()
                if not np.isnan(s):
                    skews[c] = float(s)
            except Exception:
                pass
        if skews:
            max_skew_col = max(skews, key=lambda k: abs(skews[k]))
            val_s = skews[max_skew_col]
            insights.append(
                f"**Distribution Asymmetry:** **`{max_skew_col}`** displays the highest skewness (**{val_s:+.2f}**), indicating an asymmetric distribution with potential tail anomalies."
            )

    return insights


# ==============================================================================
# 4. Session Report (.md) Generator
# ==============================================================================

def export_session_report_markdown(
    df_name: str,
    df: pd.DataFrame,
    profile: Dict[str, Any],
    preprocessing_logs: List[str],
    messages: List[Dict[str, Any]],
) -> str:
    """
    Generates a comprehensive Markdown report summarizing dataset metrics,
    applied preprocessing transformations, and complete Q&A session history.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    md_lines: List[str] = [
        f"# 📊 Data Analysis & Q&A Session Report: {df_name}",
        f"**Generated:** {timestamp}  ",
        f"**Active Dataset:** `{df_name}`  ",
        f"**Current Dimensions:** {df.shape[0]:,} rows × {df.shape[1]:,} columns  ",
        f"**Data Health Score:** **{profile.get('health_score', 'N/A')}/100**  ",
        "",
        "---",
        "",
        "## 1. Executive Summary & Health Metrics",
        "",
        "| Metric | Value |",
        "| :--- | :--- |",
        f"| **Total Rows** | {profile.get('rows', len(df)):,} |",
        f"| **Total Columns** | {profile.get('cols', len(df.columns)):,} |",
        f"| **Missing Cells** | {profile.get('missing_cells', 0):,} ({profile.get('missing_pct', 0.0)}%) |",
        f"| **Duplicate Rows** | {profile.get('duplicate_rows', 0):,} ({profile.get('duplicate_pct', 0.0)}%) |",
        f"| **Constant Columns** | {profile.get('constant_cols', 0)} |",
        f"| **Numeric Features** | {profile.get('numeric_cols', 0)} |",
        f"| **Categorical Features** | {profile.get('categorical_cols', 0)} |",
        f"| **Total Outliers Detected** | {profile.get('total_outliers', 0):,} |",
        "",
        "### Top Feature Correlations",
        "",
    ]

    top_corrs = profile.get("top_correlations", [])
    if top_corrs:
        md_lines.extend([
            "| Feature 1 | Feature 2 | Pearson r | Direction |",
            "| :--- | :--- | :--- | :--- |",
        ])
        for tc in top_corrs:
            dir_str = "Positive (+)" if tc["correlation"] > 0 else "Negative (-)"
            md_lines.append(f"| `{tc['feature_1']}` | `{tc['feature_2']}` | **{tc['correlation']:+.3f}** | {dir_str} |")
    else:
        md_lines.append("*No numeric correlation pairs available.*")

    md_lines.extend([
        "",
        "---",
        "",
        "## 2. Preprocessing & Data Cleaning Audit Trail",
        "",
    ])

    if preprocessing_logs:
        for idx, log in enumerate(preprocessing_logs, 1):
            md_lines.append(f"{idx}. {log}")
    else:
        md_lines.append("*No preprocessing transformations applied. Dataset is in original raw state.*")

    md_lines.extend([
        "",
        "---",
        "",
        "## 3. Column Diagnostics & Detected Issues",
        "",
    ])

    col_rep = profile.get("column_report")
    if isinstance(col_rep, pd.DataFrame) and not col_rep.empty:
        md_lines.append(col_rep.to_markdown(index=False))
    else:
        md_lines.append("*No column report data available.*")

    md_lines.extend([
        "",
        "---",
        "",
        "## 4. Q&A Session History & Verified Code",
        "",
    ])

    qa_turns = [m for m in messages if m.get("role") in ("user", "assistant")]
    if qa_turns:
        turn_num = 1
        i = 0
        while i < len(qa_turns):
            user_msg = qa_turns[i]
            if user_msg.get("role") == "user":
                q_text = user_msg.get("content", "").strip()
                ans_text = ""
                route_str = "Auto-Engine"
                code_snippet = None

                # Find matching assistant message
                if i + 1 < len(qa_turns) and qa_turns[i + 1].get("role") == "assistant":
                    asst_msg = qa_turns[i + 1]
                    ans_text = asst_msg.get("content", "").strip()
                    route_str = asst_msg.get("route_label") or "Auto-Engine"
                    code_snippet = asst_msg.get("code")
                    i += 2
                else:
                    i += 1

                md_lines.append(f"### Query {turn_num}: \"{q_text}\"")
                md_lines.append(f"*Engine Route: `{route_str}`*  ")
                md_lines.append("")
                md_lines.append(f"**Grounded Answer:**  \n{ans_text}")
                md_lines.append("")

                if code_snippet:
                    md_lines.append("```python")
                    md_lines.append(code_snippet.strip())
                    md_lines.append("```")
                    md_lines.append("")

                turn_num += 1
            else:
                i += 1
    else:
        md_lines.append("*No questions were asked during this session.*")

    md_lines.extend([
        "",
        "---",
        "*Report generated automatically by Chat-with-your-CSV (Phase 1 Workspace).*  ",
    ])

    return "\n".join(md_lines)
