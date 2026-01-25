#!/usr/bin/env python3
"""
Normalize IMC Filled Abstract Forms (wide Excel) into:
1) question_catalog.csv  (one row per question/column)
2) responses_long.csv    (one row per contract/form response per question)

Runs with NO CLI arguments.

Assumptions:
- Excel file is at: data/raw/forms/Filled Abstract Forms.xlsx
  (change DEFAULT_INPUT below if needed)
- A unique row identifier column is named "ID"
  (change DEFAULT_ROW_ID_COL below if needed)

Outputs:
- data/processed/forms_normalized/question_catalog.csv
- data/processed/forms_normalized/responses_long.csv
"""

import hashlib
import re
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd

# ----------------------------
# Defaults (edit if needed)
# ----------------------------
DEFAULT_INPUT = Path("data/raw/forms/Filled Abstract Forms.xlsx")
DEFAULT_OUTDIR = Path("data/processed/forms_normalized")
DEFAULT_ROW_ID_COL = "ID"

META_COL_CANDIDATES = {"id", "start time", "completion time", "email", "name"}


# ----------------------------
# Helpers
# ----------------------------
def clean_text(s: str) -> str:
    s = str(s)
    s = s.replace("\u00a0", " ")  # non-breaking spaces
    s = re.sub(r"\s+", " ", s).strip()
    return s

def slugify(s: str, max_len: int = 60) -> str:
    s = clean_text(s).lower()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"[\s_-]+", "_", s).strip("_")
    return s[:max_len] if len(s) > max_len else s

def make_question_id(question_text: str) -> str:
    base = clean_text(question_text)
    h = hashlib.sha1(base.encode("utf-8")).hexdigest()[:8]
    slug = slugify(base, max_len=50) or "question"
    return f"Q_{slug}_{h}"

def infer_answer_type(series: pd.Series) -> str:
    """
    Light heuristics:
    - boolean: values mostly Yes/No/True/False
    - date: parseable dates in majority
    - number: numeric in majority
    - text: default
    """
    vals = series.dropna().astype(str).map(clean_text)
    if len(vals) == 0:
        return "text"

    # boolean check
    bool_set = {"yes", "no", "true", "false", "y", "n"}
    bool_ratio = vals.map(lambda x: x.lower() in bool_set).mean()
    if bool_ratio >= 0.7:
        return "boolean"

    # date check (suppress pandas "could not infer format" warnings)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        parsed_dates = pd.to_datetime(vals, errors="coerce")
    date_ratio = parsed_dates.notna().mean()
    if date_ratio >= 0.7:
        return "date"

    # number check
    def to_num(x: str):
        x2 = re.sub(r"[$,%]", "", x)
        x2 = x2.replace(",", "").strip()
        return pd.to_numeric(x2, errors="coerce")

    nums = vals.map(to_num)
    num_ratio = nums.notna().mean()
    if num_ratio >= 0.7:
        return "number"

    return "text"

def is_meta_column(colname: str) -> bool:
    return clean_text(colname).lower() in META_COL_CANDIDATES

def normalize_answer(x):
    if pd.isna(x):
        return None
    s = clean_text(x)
    return s if s != "" else None


# ----------------------------
# Main
# ----------------------------
def main():
    input_path = DEFAULT_INPUT
    outdir = DEFAULT_OUTDIR
    row_id_col = DEFAULT_ROW_ID_COL

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input Excel not found at:\n  {input_path}\n\n"
            f"Fix by either:\n"
            f"1) placing the file at that path, OR\n"
            f"2) editing DEFAULT_INPUT at the top of this script."
        )

    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(input_path)
    df.columns = [clean_text(c) for c in df.columns]

    if row_id_col not in df.columns:
        # helpful fallback: if there is no ID column, create one
        df.insert(0, row_id_col, [f"ROW-{i+1:04d}" for i in range(len(df))])
        print(f"⚠️  '{row_id_col}' column not found; created a synthetic row id column.")

    meta_cols = [c for c in df.columns if is_meta_column(c)]
    question_cols = [c for c in df.columns if c not in meta_cols]

    # Build question catalog
    catalog_rows = []
    for q in question_cols:
        q_id = make_question_id(q)
        ans_type = infer_answer_type(df[q])
        catalog_rows.append(
            {
                "question_id": q_id,
                "question_text": q,
                "answer_type": ans_type,
                "source_column": q,
            }
        )

    catalog = pd.DataFrame(catalog_rows).sort_values("question_id")
    catalog_path = outdir / "question_catalog.csv"
    catalog.to_csv(catalog_path, index=False, encoding="utf-8")

    # Melt to long format
    id_vars = [row_id_col] + [c for c in meta_cols if c != row_id_col]
    long_df = df.melt(
        id_vars=id_vars,
        value_vars=question_cols,
        var_name="question_text",
        value_name="answer_raw",
    )

    q_map = dict(zip(catalog["question_text"], catalog["question_id"]))
    t_map = dict(zip(catalog["question_text"], catalog["answer_type"]))

    long_df["question_id"] = long_df["question_text"].map(q_map)
    long_df["answer_type"] = long_df["question_text"].map(t_map)
    long_df["answer"] = long_df["answer_raw"].apply(normalize_answer)

    # Drop blanks (keep only answered fields)
    long_df = long_df[long_df["answer"].notna()].copy()

    long_df["normalized_at"] = datetime.now().isoformat(timespec="seconds")
    long_df.rename(columns={row_id_col: "form_row_id"}, inplace=True)

    # Column order
    front_cols = ["form_row_id"]
    preferred_meta = ["Start time", "Completion time", "Email", "Name"]
    for c in preferred_meta:
        if c in long_df.columns:
            front_cols.append(c)

    final_cols = front_cols + ["question_id", "question_text", "answer_type", "answer"]
    remaining = [c for c in long_df.columns if c not in final_cols and c not in ["answer_raw"]]
    final_cols += remaining
    long_df = long_df[final_cols]

    responses_path = outdir / "responses_long.csv"
    long_df.to_csv(responses_path, index=False, encoding="utf-8")

    print("✅ Normalization complete")
    print(f"- Input:  {input_path}")
    print(f"- Forms (rows):     {len(df)}")
    print(f"- Questions (cols): {len(question_cols)}")
    print(f"- Output catalog:   {catalog_path}")
    print(f"- Output responses: {responses_path}")
    print("\nNext recommended step:")
    print("- Create a mapping from form_row_id → contract_id (and contract filename/path).")


if __name__ == "__main__":
    main()
