#!/usr/bin/env python3
"""
Normalize IMC Filled Abstract Forms (wide Excel) into:
1) question_catalog.csv  (one row per question/column)
2) responses_long.csv    (one row per contract/form response per question)
3) form_contract_map.csv (one row per form -> contract filename/id)

Runs with NO CLI arguments.

Assumptions:
- Excel file is at: data/raw/forms/Filled Abstract Forms.xlsx
  (change DEFAULT_INPUT below if needed)
- A unique row identifier column is named "ID"
  (change DEFAULT_ROW_ID_COL below if needed)
- The Excel includes a first column with contract file names (e.g., Contract__1.pdf).
  The script auto-detects it; you can force it via DEFAULT_CONTRACT_FILENAME_COL.

Outputs:
- data/processed/forms_normalized/question_catalog.csv
- data/processed/forms_normalized/responses_long.csv
- data/processed/forms_normalized/form_contract_map.csv
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

# If your column name is exactly this, it will be picked up immediately.
DEFAULT_CONTRACT_FILENAME_COL = "Contract file names"

# "Meta" columns that should NOT become questions (we'll also force-add contract filename)
META_COL_CANDIDATES = {
    "id",
    "start time",
    "completion time",
    "email",
    "name",
    "contract file names",
    "contract filename",
    "contract file name",
    "contract",
}


# ----------------------------
# Helpers
# ----------------------------
def clean_text(s: str) -> str:
    s = "" if s is None else str(s)
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
    if parsed_dates.notna().mean() >= 0.7:
        return "date"

    # number check
    def to_num(x: str):
        x2 = re.sub(r"[$,%]", "", x)
        x2 = x2.replace(",", "").strip()
        return pd.to_numeric(x2, errors="coerce")

    nums = vals.map(to_num)
    if nums.notna().mean() >= 0.7:
        return "number"

    return "text"


def is_meta_column(colname: str) -> bool:
    return clean_text(colname).lower() in META_COL_CANDIDATES


def normalize_answer(x):
    if pd.isna(x):
        return None
    s = clean_text(x)
    return s if s != "" else None


def find_contract_filename_col(columns) -> str | None:
    """
    Prefer DEFAULT_CONTRACT_FILENAME_COL exact match, else fuzzy match:
    any column containing 'contract' and ('file' or 'filename' or 'name').
    """
    cols = [clean_text(c) for c in columns]

    # exact match first
    for c in cols:
        if c.lower() == DEFAULT_CONTRACT_FILENAME_COL.lower():
            return c

    # fuzzy match
    for c in cols:
        cl = c.lower()
        if "contract" in cl and ("file" in cl or "filename" in cl or "name" in cl):
            return c

    return None


def contract_id_from_filename(filename: str | None) -> str | None:
    """
    Stable contract_id from filename stem.
    Example: 'Contract__1.pdf' -> 'contract__1'
    """
    if filename is None:
        return None
    fn = clean_text(filename)
    if fn == "":
        return None
    fn = fn.replace("\\", "/").split("/")[-1]  # basename
    stem = re.sub(r"\.[A-Za-z0-9]{1,6}$", "", fn)  # drop extension
    stem = stem.strip()
    return stem.lower() if stem else None


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

    # Ensure row id exists
    if row_id_col not in df.columns:
        df.insert(0, row_id_col, [f"ROW-{i+1:04d}" for i in range(len(df))])
        print(f"⚠️  '{row_id_col}' column not found; created a synthetic row id column.")

    # Detect contract filename column and force it into meta
    contract_col = find_contract_filename_col(df.columns)
    if contract_col is None:
        raise ValueError(
            "Could not find a contract filename column.\n"
            f"Expected something like '{DEFAULT_CONTRACT_FILENAME_COL}' or a column containing "
            "'contract' + ('file'/'filename'/'name')."
        )

    meta_cols = [c for c in df.columns if is_meta_column(c)]
    if contract_col not in meta_cols:
        meta_cols.append(contract_col)
    if row_id_col not in meta_cols:
        meta_cols.append(row_id_col)

    # Only non-meta columns become questions
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

    # Standardize key columns
    long_df["normalized_at"] = datetime.now().isoformat(timespec="seconds")
    long_df.rename(columns={row_id_col: "form_row_id", contract_col: "contract_filename"}, inplace=True)
    long_df["contract_id"] = long_df["contract_filename"].apply(contract_id_from_filename)

    # Column order
    front_cols = ["form_row_id", "contract_filename", "contract_id"]
    preferred_meta = ["Start time", "Completion time", "Email", "Name"]
    for c in preferred_meta:
        if c in long_df.columns:
            front_cols.append(c)

    final_cols = front_cols + ["question_id", "question_text", "answer_type", "answer", "normalized_at"]
    remaining = [c for c in long_df.columns if c not in final_cols and c not in ["answer_raw"]]
    final_cols += remaining
    long_df = long_df[final_cols]

    responses_path = outdir / "responses_long.csv"
    long_df.to_csv(responses_path, index=False, encoding="utf-8")

    # Also write a simple form -> contract mapping table
    form_map = df[[row_id_col, contract_col]].copy()
    form_map.rename(columns={row_id_col: "form_row_id", contract_col: "contract_filename"}, inplace=True)
    form_map["contract_id"] = form_map["contract_filename"].apply(contract_id_from_filename)
    form_map_path = outdir / "form_contract_map.csv"
    form_map.to_csv(form_map_path, index=False, encoding="utf-8")

    print("✅ Normalization complete")
    print(f"- Input:              {input_path}")
    print(f"- Forms (rows):        {len(df)}")
    print(f"- Questions (cols):    {len(question_cols)}")
    print(f"- Contract col:        {contract_col}")
    print(f"- Output catalog:      {catalog_path}")
    print(f"- Output responses:    {responses_path}")
    print(f"- Output form mapping: {form_map_path}")
    print("\nNext recommended step:")
    print("- Start contract text extraction + clause segmentation for the mapped contract PDFs.")


if __name__ == "__main__":
    main()
