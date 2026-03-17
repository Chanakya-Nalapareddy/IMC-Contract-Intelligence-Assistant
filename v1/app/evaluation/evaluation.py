from pathlib import Path
import json
import math
import os
import re
from typing import Any, Dict, List, Optional

import pandas as pd
from openai import AzureOpenAI


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"

EXCEL_PATH = DATA_DIR / "Filled Abstract Forms.xlsx"
QUESTIONS_PATH = DATA_DIR / "questions.jsonl"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

SHEET_NAME = 0


# ------------------------------------------------------------
# Azure OpenAI judge config
# ------------------------------------------------------------
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
AZURE_OPENAI_CHAT_DEPLOYMENT = (
    os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
    or os.getenv("AZURE_DEPLOYMENT_NAME")
)

CATEGORICAL_TYPES = {
    "boolean",
    "enum",
    "identifier",
    "date",
    "number",
    "percentage",
    "currency",
    "currency_or_number",
    "article_reference",
    "article",
}
DESCRIPTIVE_TYPES = {
    "text",
    "long_text",
    "list",
    "list[string]",
    "string_list",
}


# ------------------------------------------------------------
# Path helpers
# ------------------------------------------------------------
def get_contract_file_name() -> str:
    files = [f for f in RAW_DIR.iterdir() if f.is_file() and f.suffix.lower() == ".pdf"]

    if not files:
        raise FileNotFoundError(f"No PDF files found in {RAW_DIR}")

    if len(files) > 1:
        raise RuntimeError(
            "Keep only ONE PDF in raw for evaluation:\n"
            + "\n".join(f" - {f.name}" for f in files)
        )

    return files[0].name


def get_contract_id(contract_file_name: str) -> str:
    return Path(contract_file_name).stem


def get_contract_out_dir(contract_file_name: str) -> Path:
    contract_id = get_contract_id(contract_file_name)
    out_dir = PROCESSED_DIR / contract_id
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def get_results_path(contract_file_name: str) -> Path:
    contract_id = get_contract_id(contract_file_name)
    results_path = PROCESSED_DIR / contract_id / "results.jsonl"
    if not results_path.exists():
        raise FileNotFoundError(f"results.jsonl not found at: {results_path}")
    return results_path


def get_output_paths(contract_file_name: str) -> Dict[str, Path]:
    out_dir = get_contract_out_dir(contract_file_name)
    return {
        "details": out_dir / "evaluation_details.jsonl",
        "summary": out_dir / "evaluation_summary.json",
        "failures": out_dir / "evaluation_failures.csv",
        "gold_json": out_dir / "gold_answers.json",
        "gold_jsonl": out_dir / "gold_answers.jsonl",
        "metrics_table": out_dir / "evaluation_metrics_table.csv",
        "categorical_table": out_dir / "categorical_llm_results.csv",
        "descriptive_table": out_dir / "descriptive_llm_results.csv",
        "categorical_combined_table": out_dir / "categorical_llm_results_combined.csv",
        "descriptive_combined_table": out_dir / "descriptive_llm_results_combined.csv",
        "markdown_report": out_dir / "evaluation_report.md",
        "html_report": out_dir / "evaluation_report.html",
    }


# ------------------------------------------------------------
# Utility helpers
# ------------------------------------------------------------
def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_question(text: str) -> str:
    text = str(text or "")
    text = text.replace("\xa0", " ")
    text = text.replace("“", '"').replace("”", '"').replace("’", "'")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_filename(name: str) -> str:
    name = str(name or "").strip().lower()
    name = name.replace("\\", "/")
    return name.split("/")[-1]


def is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    s = str(value).strip()
    return s == "" or s.lower() in {"nan", "none", "null", "n/a", "na"}


def safe_mean(values: List[float]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def safe_rate_bools(values: List[bool]) -> Optional[float]:
    if not values:
        return None
    return sum(1 for v in values if v) / len(values)


def round_or_none(value: Optional[float], digits: int = 4) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), digits)


def extract_first_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"No JSON object found in response: {text[:500]}")
    return json.loads(text[start:end + 1])


def clean_cell(value: Any) -> Optional[str]:
    if is_blank(value):
        return None
    return str(value).strip()


def format_html_value(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value).replace("\n", " ").strip()


def df_to_html_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "<p><em>No data</em></p>"

    lines = []
    lines.append('<table class="report-table">')
    lines.append("  <thead>")
    lines.append("    <tr>")
    for col in df.columns:
        lines.append(f"      <th>{col}</th>")
    lines.append("    </tr>")
    lines.append("  </thead>")
    lines.append("  <tbody>")

    for _, row in df.iterrows():
        lines.append("    <tr>")
        for col in df.columns:
            lines.append(f"      <td>{format_html_value(row[col])}</td>")
        lines.append("    </tr>")

    lines.append("  </tbody>")
    lines.append("</table>")
    return "\n".join(lines)


# ------------------------------------------------------------
# Gold answer extraction
# ------------------------------------------------------------
def extract_gold_answers(
    df: pd.DataFrame,
    gold_row: pd.Series,
    questions: List[Dict[str, Any]],
    contract_file_name: str,
    contract_id: str,
    gold_json_out: Path,
    gold_jsonl_out: Path,
) -> List[Dict[str, Any]]:
    gold_records: List[Dict[str, Any]] = []

    for item in questions:
        question = normalize_question(item["question"])
        expected_type = item.get("expected_type", "text")

        if question not in df.columns:
            gold_value = None
            status = "question_not_in_excel"
        else:
            val = gold_row[question]
            gold_value = clean_cell(val)
            status = "ok" if gold_value is not None else "blank_gold"

        gold_records.append({
            "contract_file_name": contract_file_name,
            "contract_id": contract_id,
            "question": question,
            "expected_type": expected_type,
            "gold_value": gold_value,
            "status": status,
        })

    with gold_json_out.open("w", encoding="utf-8") as f:
        json.dump(gold_records, f, indent=2, ensure_ascii=False)

    with gold_jsonl_out.open("w", encoding="utf-8") as f:
        for row in gold_records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return gold_records


# ------------------------------------------------------------
# LLM Judge
# ------------------------------------------------------------
class LLMJudge:
    def __init__(self) -> None:
        missing = []
        if not AZURE_OPENAI_ENDPOINT:
            missing.append("AZURE_OPENAI_ENDPOINT")
        if not AZURE_OPENAI_API_KEY:
            missing.append("AZURE_OPENAI_API_KEY")
        if not AZURE_OPENAI_CHAT_DEPLOYMENT:
            missing.append("AZURE_OPENAI_CHAT_DEPLOYMENT or AZURE_DEPLOYMENT_NAME")

        if missing:
            raise RuntimeError(
                "Missing required Azure OpenAI environment variables: "
                + ", ".join(missing)
            )

        self.client = AzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
        )
        self.model = AZURE_OPENAI_CHAT_DEPLOYMENT

    def _call(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        text = resp.choices[0].message.content or ""
        return extract_first_json_object(text)

    def judge(
        self,
        question: str,
        expected_type: str,
        gold_answer: str,
        predicted_answer: Optional[str],
    ) -> Dict[str, Any]:
        predicted_answer = "" if predicted_answer is None else str(predicted_answer)
        et = (expected_type or "text").lower().strip()

        if et in CATEGORICAL_TYPES:
            return self.judge_categorical(question, expected_type, gold_answer, predicted_answer)

        return self.judge_descriptive(question, expected_type, gold_answer, predicted_answer)

    def judge_categorical(
        self,
        question: str,
        expected_type: str,
        gold_answer: str,
        predicted_answer: str,
    ) -> Dict[str, Any]:
        system_prompt = """
You are an evaluation judge for contract extraction.
Compare a GOLD answer and a PREDICTED answer for a single question.

Rules:
- Do NOT require exact string equality.
- Allow harmless formatting differences.
- Focus on whether the predicted answer is substantively the same as the gold answer.
- For structured/categorical values such as boolean, enum, identifier, date, number, percentage, currency, or article reference:
  - correct = same meaning/value
  - partial = close but incomplete, ambiguous, or missing a meaningful part
  - incorrect = wrong, contradictory, or unsupported by the gold answer
- Return strict JSON only.

Return schema:
{
  "group": "categorical_accuracy",
  "verdict": "correct|partial|incorrect",
  "score": 1.0,
  "is_correct": true,
  "confidence": 0.0,
  "explanation": "short reason"
}
""".strip()

        user_prompt = f"""
Question:
{question}

Expected type:
{expected_type}

Gold answer:
{gold_answer}

Predicted answer:
{predicted_answer}
""".strip()

        data = self._call(system_prompt, user_prompt)

        verdict = str(data.get("verdict", "incorrect")).lower().strip()
        score = data.get("score")
        if score is None:
            score = 1.0 if verdict == "correct" else 0.5 if verdict == "partial" else 0.0

        return {
            "group": "categorical_accuracy",
            "verdict": verdict,
            "score": float(score),
            "is_correct": bool(data.get("is_correct", verdict == "correct")),
            "confidence": float(data.get("confidence", 0.0)),
            "explanation": str(data.get("explanation", "")).strip(),
            "precision": None,
            "recall": None,
            "f1": None,
            "cosine_similarity": None,
            "token_f1": None,
            "hybrid_text_score": None,
        }

    def judge_descriptive(
        self,
        question: str,
        expected_type: str,
        gold_answer: str,
        predicted_answer: str,
    ) -> Dict[str, Any]:
        system_prompt = """
You are an evaluation judge for contract extraction.
Compare a GOLD answer and a PREDICTED answer for a single question.

Rules:
- Evaluate semantic agreement, not literal wording.
- For text/long_text/list answers, score the prediction against the gold answer on:
  1. semantic_similarity: same meaning/content
  2. coverage: how much of the important gold content is captured
  3. contradiction: whether the prediction conflicts with the gold answer
- final_score should be between 0 and 1.
- A good answer can be paraphrased.
- Return strict JSON only.

Return schema:
{
  "group": "descriptive_metrics",
  "semantic_similarity": 0.0,
  "coverage": 0.0,
  "contradiction": 0.0,
  "token_f1_like": 0.0,
  "hybrid_text_score": 0.0,
  "score": 0.0,
  "verdict": "strong|partial|weak|incorrect",
  "confidence": 0.0,
  "explanation": "short reason"
}
""".strip()

        user_prompt = f"""
Question:
{question}

Expected type:
{expected_type}

Gold answer:
{gold_answer}

Predicted answer:
{predicted_answer}
""".strip()

        data = self._call(system_prompt, user_prompt)

        semantic = float(data.get("semantic_similarity", 0.0))
        coverage = float(data.get("coverage", 0.0))
        contradiction = float(data.get("contradiction", 0.0))
        token_like = float(data.get("token_f1_like", 0.0))

        hybrid = data.get("hybrid_text_score")
        if hybrid is None:
            hybrid = max(
                0.0,
                (0.55 * semantic) + (0.25 * coverage) + (0.20 * token_like) - (0.30 * contradiction)
            )

        score = data.get("score")
        if score is None:
            score = hybrid

        return {
            "group": "descriptive_metrics",
            "verdict": str(data.get("verdict", "partial")).strip().lower(),
            "score": float(score),
            "is_correct": None,
            "confidence": float(data.get("confidence", 0.0)),
            "explanation": str(data.get("explanation", "")).strip(),
            "precision": None,
            "recall": None,
            "f1": None if expected_type.lower() not in {"list", "list[string]", "string_list"} else coverage,
            "cosine_similarity": semantic,
            "token_f1": token_like,
            "hybrid_text_score": float(hybrid),
        }


# ------------------------------------------------------------
# Reporting helpers
# ------------------------------------------------------------
def build_categorical_table(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    order = [
        "boolean", "enum", "identifier", "date",
        "number", "percentage", "currency", "currency_or_number",
        "article_reference", "article"
    ]
    result_rows = []

    for dtype in order:
        subset = [r for r in rows if r["expected_type"].lower() == dtype]
        if not subset:
            continue

        score_vals = [r["score"] for r in subset if r.get("score") is not None]
        correct_vals = [bool(r["is_correct"]) for r in subset if r.get("is_correct") is not None]

        result_rows.append({
            "data_type": dtype,
            "count": len(subset),
            "metric": "judge_score_mean",
            "value": round_or_none(safe_mean(score_vals)),
        })

        if correct_vals:
            result_rows.append({
                "data_type": dtype,
                "count": len(subset),
                "metric": "correct_rate",
                "value": round_or_none(safe_rate_bools(correct_vals)),
            })

    df = pd.DataFrame(result_rows)
    if not df.empty:
        df = df.sort_values(by=["data_type", "metric"]).reset_index(drop=True)
    return df


def build_descriptive_table(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    result_rows = []

    for dtype in ["list", "text", "long_text", "list[string]", "string_list"]:
        subset = [r for r in rows if r["expected_type"].lower() == dtype]
        if not subset:
            continue

        cos_vals = [r["cosine_similarity"] for r in subset if r.get("cosine_similarity") is not None]
        tok_vals = [r["token_f1"] for r in subset if r.get("token_f1") is not None]
        hyb_vals = [r["hybrid_text_score"] for r in subset if r.get("hybrid_text_score") is not None]
        score_vals = [r["score"] for r in subset if r.get("score") is not None]

        if cos_vals:
            result_rows.append({
                "data_type": dtype,
                "count": len(subset),
                "metric": "semantic_similarity_mean",
                "value": round_or_none(safe_mean(cos_vals)),
            })
        if tok_vals:
            result_rows.append({
                "data_type": dtype,
                "count": len(subset),
                "metric": "token_f1_like_mean",
                "value": round_or_none(safe_mean(tok_vals)),
            })
        if hyb_vals:
            result_rows.append({
                "data_type": dtype,
                "count": len(subset),
                "metric": "hybrid_text_score_mean",
                "value": round_or_none(safe_mean(hyb_vals)),
            })
        if score_vals:
            result_rows.append({
                "data_type": dtype,
                "count": len(subset),
                "metric": "judge_score_mean",
                "value": round_or_none(safe_mean(score_vals)),
            })

    df = pd.DataFrame(result_rows)
    if not df.empty:
        df = df.sort_values(by=["data_type", "metric"]).reset_index(drop=True)
    return df


def build_combined_table(table_df: pd.DataFrame, label: str) -> pd.DataFrame:
    if table_df.empty:
        return pd.DataFrame(columns=["data_type", "count", "metric", "value"])

    result_rows = []

    for metric in table_df["metric"].dropna().unique():
        subset = table_df[table_df["metric"] == metric].copy()
        if subset.empty:
            continue

        subset["count"] = pd.to_numeric(subset["count"], errors="coerce")
        subset["value"] = pd.to_numeric(subset["value"], errors="coerce")
        subset = subset.dropna(subset=["count", "value"])

        if subset.empty:
            continue

        metric_total_count = int(subset["count"].sum())
        weighted_mean = float((subset["count"] * subset["value"]).sum() / subset["count"].sum())

        result_rows.append({
            "data_type": label,
            "count": metric_total_count,
            "metric": metric,
            "value": round(weighted_mean, 4),
        })

    df = pd.DataFrame(result_rows)
    if not df.empty:
        df = df.sort_values(by=["metric"]).reset_index(drop=True)
    return df


def write_markdown_report(
    path: Path,
    contract_file_name: str,
    contract_id: str,
    summary: Dict[str, Any],
    categorical_table: pd.DataFrame,
    categorical_combined: pd.DataFrame,
    descriptive_table: pd.DataFrame,
    descriptive_combined: pd.DataFrame,
) -> None:
    summary_rows = pd.DataFrame([
        {"metric": "total_questions", "value": summary.get("total_questions")},
        {"metric": "gold_non_blank_count", "value": summary.get("gold_non_blank_count")},
        {"metric": "predicted_non_blank_count", "value": summary.get("predicted_non_blank_count")},
        {"metric": "scored_questions", "value": summary.get("scored_questions")},
        {"metric": "perfect_matches", "value": summary.get("perfect_matches")},
        {"metric": "overall_judge_score_mean", "value": summary.get("overall_judge_score_mean")},
        {"metric": "judge_model_deployment", "value": summary.get("judge_model_deployment")},
    ])

    parts = [
        "# Contract Evaluation Report",
        "",
        f"**Contract:** {contract_file_name}",
        f"**Contract ID:** {contract_id}",
        "",
        "## Overall Summary",
        "",
        df_to_html_table(summary_rows),
        "",
        "## Categorical Results",
        "",
        df_to_html_table(categorical_table),
        "",
        "## Categorical Combined Results",
        "",
        df_to_html_table(categorical_combined),
        "",
        "## Descriptive Results",
        "",
        df_to_html_table(descriptive_table),
        "",
        "## Descriptive Combined Results",
        "",
        df_to_html_table(descriptive_combined),
        "",
    ]

    path.write_text("\n".join(parts), encoding="utf-8")


def write_html_report(
    path: Path,
    contract_file_name: str,
    contract_id: str,
    summary: Dict[str, Any],
    categorical_table: pd.DataFrame,
    categorical_combined: pd.DataFrame,
    descriptive_table: pd.DataFrame,
    descriptive_combined: pd.DataFrame,
) -> None:
    summary_rows = pd.DataFrame([
        {"metric": "total_questions", "value": summary.get("total_questions")},
        {"metric": "gold_non_blank_count", "value": summary.get("gold_non_blank_count")},
        {"metric": "predicted_non_blank_count", "value": summary.get("predicted_non_blank_count")},
        {"metric": "scored_questions", "value": summary.get("scored_questions")},
        {"metric": "perfect_matches", "value": summary.get("perfect_matches")},
        {"metric": "overall_judge_score_mean", "value": summary.get("overall_judge_score_mean")},
        {"metric": "judge_model_deployment", "value": summary.get("judge_model_deployment")},
    ])

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Contract Evaluation Report</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body {{
      font-family: Arial, sans-serif;
      margin: 40px;
      color: #222;
      background: #fff;
    }}

    .container {{
      max-width: 1200px;
      margin: 0 auto;
    }}

    h1 {{
      color: #1f3b5b;
      margin-bottom: 10px;
    }}

    h2 {{
      margin-top: 36px;
      color: #1f3b5b;
      border-bottom: 2px solid #e6edf5;
      padding-bottom: 8px;
    }}

    .meta {{
      margin-bottom: 24px;
      line-height: 1.8;
    }}

    .report-table {{
      border-collapse: collapse;
      width: 100%;
      margin-top: 12px;
      margin-bottom: 24px;
      font-size: 14px;
    }}

    .report-table th,
    .report-table td {{
      border: 1px solid #d9e2ec;
      padding: 10px 12px;
      text-align: left;
      vertical-align: top;
    }}

    .report-table th {{
      background-color: #f4f7fb;
      font-weight: 700;
    }}

    .report-table tr:nth-child(even) td {{
      background-color: #fafcff;
    }}

    .note {{
      color: #5b6b7a;
      font-size: 13px;
      margin-top: 8px;
    }}
  </style>
</head>
<body>
  <div class="container">
    <h1>Contract Evaluation Report</h1>

    <div class="meta">
      <div><strong>Contract:</strong> {contract_file_name}</div>
      <div><strong>Contract ID:</strong> {contract_id}</div>
    </div>

    <h2>Overall Summary</h2>
    {df_to_html_table(summary_rows)}

    <h2>Categorical Results</h2>
    {df_to_html_table(categorical_table)}

    <h2>Categorical Combined Results</h2>
    {df_to_html_table(categorical_combined)}

    <h2>Descriptive Results</h2>
    {df_to_html_table(descriptive_table)}

    <h2>Descriptive Combined Results</h2>
    {df_to_html_table(descriptive_combined)}

    <div class="note">
      Open this file directly in a browser using a file:/// path.
    </div>
  </div>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main() -> None:
    contract_file_name = get_contract_file_name()
    contract_id = get_contract_id(contract_file_name)
    results_path = get_results_path(contract_file_name)
    paths = get_output_paths(contract_file_name)

    print(f"Evaluating contract: {contract_file_name}")
    print(f"Contract ID        : {contract_id}")
    print(f"Results path       : {results_path}")

    df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME)
    df.columns = [normalize_question(c) for c in df.columns]

    contract_col = "Contract file names"
    if contract_col not in df.columns:
        raise KeyError(f'Missing required Excel column: "{contract_col}"')

    target_name = normalize_filename(contract_file_name)
    mask = df[contract_col].astype(str).map(normalize_filename) == target_name
    matched = df.loc[mask]

    if matched.empty:
        sample_names = df[contract_col].astype(str).dropna().head(20).tolist()
        raise ValueError(
            f'No Excel row found for "{contract_file_name}" in "{contract_col}". '
            f"Sample available names: {sample_names}"
        )

    if len(matched) > 1:
        raise ValueError(
            f'Multiple Excel rows found for "{contract_file_name}". '
            "Please make contract names unique or add ID-based matching."
        )

    gold_row = matched.iloc[0]

    questions = load_jsonl(QUESTIONS_PATH)
    results = load_jsonl(results_path)

    gold_records = extract_gold_answers(
        df=df,
        gold_row=gold_row,
        questions=questions,
        contract_file_name=contract_file_name,
        contract_id=contract_id,
        gold_json_out=paths["gold_json"],
        gold_jsonl_out=paths["gold_jsonl"],
    )

    print(f"Gold answers written to: {paths['gold_json']}")
    print(f"Gold answers JSONL     : {paths['gold_jsonl']}")

    result_map: Dict[str, Dict[str, Any]] = {}
    for item in results:
        q = normalize_question(item.get("question", ""))
        result_map[q] = item

    judge = LLMJudge()
    print(f"LLM judge deployment   : {AZURE_OPENAI_CHAT_DEPLOYMENT}")

    details: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    for gold_item in gold_records:
        question = gold_item["question"]
        expected_type = gold_item["expected_type"]
        gold_value = gold_item["gold_value"]

        result_item = result_map.get(question, {})
        predicted_value = clean_cell(result_item.get("value"))

        if gold_value is None:
            details.append({
                "contract_file_name": contract_file_name,
                "contract_id": contract_id,
                "question": question,
                "expected_type": expected_type,
                "gold_value": None,
                "predicted_value": predicted_value,
                "group": None,
                "score": None,
                "is_correct": None,
                "confidence": None,
                "verdict": None,
                "explanation": None,
                "precision": None,
                "recall": None,
                "f1": None,
                "cosine_similarity": None,
                "token_f1": None,
                "hybrid_text_score": None,
                "status": gold_item["status"],
            })
            continue

        judge_result = judge.judge(
            question=question,
            expected_type=expected_type,
            gold_answer=gold_value,
            predicted_answer=predicted_value,
        )

        is_perfect = False
        if judge_result["group"] == "categorical_accuracy":
            is_perfect = bool(judge_result.get("is_correct"))
        else:
            is_perfect = float(judge_result.get("score", 0.0)) >= 0.95

        record = {
            "contract_file_name": contract_file_name,
            "contract_id": contract_id,
            "question": question,
            "expected_type": expected_type,
            "gold_value": gold_value,
            "predicted_value": predicted_value,
            "group": judge_result["group"],
            "score": round(float(judge_result["score"]), 6) if judge_result.get("score") is not None else None,
            "is_correct": judge_result.get("is_correct"),
            "confidence": round(float(judge_result["confidence"]), 6) if judge_result.get("confidence") is not None else None,
            "verdict": judge_result.get("verdict"),
            "explanation": judge_result.get("explanation"),
            "precision": judge_result.get("precision"),
            "recall": judge_result.get("recall"),
            "f1": round(float(judge_result["f1"]), 6) if judge_result.get("f1") is not None else None,
            "cosine_similarity": round(float(judge_result["cosine_similarity"]), 6) if judge_result.get("cosine_similarity") is not None else None,
            "token_f1": round(float(judge_result["token_f1"]), 6) if judge_result.get("token_f1") is not None else None,
            "hybrid_text_score": round(float(judge_result["hybrid_text_score"]), 6) if judge_result.get("hybrid_text_score") is not None else None,
            "status": "ok" if is_perfect else "mismatch",
        }
        details.append(record)

        if not is_perfect:
            fail = dict(record)
            fail["raw_answer"] = result_item.get("raw_answer")
            fail["citations"] = result_item.get("citations")
            failures.append(fail)

    scored = [d for d in details if d.get("score") is not None]

    categorical_rows = [
        r for r in scored
        if r["expected_type"].lower() in CATEGORICAL_TYPES
    ]
    descriptive_rows = [
        r for r in scored
        if r["expected_type"].lower() in DESCRIPTIVE_TYPES
    ]

    categorical_table = build_categorical_table(categorical_rows)
    descriptive_table = build_descriptive_table(descriptive_rows)

    categorical_combined_table = build_combined_table(
        categorical_table,
        label="categorical_combined",
    )
    descriptive_combined_table = build_combined_table(
        descriptive_table,
        label="descriptive_combined",
    )

    metrics_table = pd.concat(
        [
            categorical_table.assign(result_set="categorical_llm_results"),
            categorical_combined_table.assign(result_set="categorical_llm_results_combined"),
            descriptive_table.assign(result_set="descriptive_llm_results"),
            descriptive_combined_table.assign(result_set="descriptive_llm_results_combined"),
        ],
        ignore_index=True,
    )

    summary = {
        "contract_file_name": contract_file_name,
        "contract_id": contract_id,
        "results_path": str(results_path),
        "gold_answers_path": str(paths["gold_json"]),
        "gold_answers_jsonl_path": str(paths["gold_jsonl"]),
        "excel_path": str(EXCEL_PATH),
        "questions_path": str(QUESTIONS_PATH),
        "judge_model_deployment": AZURE_OPENAI_CHAT_DEPLOYMENT,
        "total_questions": len(questions),
        "gold_non_blank_count": sum(1 for g in gold_records if g["gold_value"] is not None),
        "predicted_non_blank_count": sum(1 for r in results if not is_blank(r.get("value"))),
        "scored_questions": len(scored),
        "perfect_matches": sum(1 for d in scored if d["status"] == "ok"),
        "overall_judge_score_mean": round_or_none(safe_mean([d["score"] for d in scored])),
        "categorical_llm_results": categorical_table.to_dict(orient="records"),
        "categorical_llm_results_combined": categorical_combined_table.to_dict(orient="records"),
        "descriptive_llm_results": descriptive_table.to_dict(orient="records"),
        "descriptive_llm_results_combined": descriptive_combined_table.to_dict(orient="records"),
    }

    with paths["details"].open("w", encoding="utf-8") as f:
        for row in details:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with paths["summary"].open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    pd.DataFrame(failures).to_csv(paths["failures"], index=False)
    metrics_table.to_csv(paths["metrics_table"], index=False)
    categorical_table.to_csv(paths["categorical_table"], index=False)
    descriptive_table.to_csv(paths["descriptive_table"], index=False)
    categorical_combined_table.to_csv(paths["categorical_combined_table"], index=False)
    descriptive_combined_table.to_csv(paths["descriptive_combined_table"], index=False)

    write_markdown_report(
        path=paths["markdown_report"],
        contract_file_name=contract_file_name,
        contract_id=contract_id,
        summary=summary,
        categorical_table=categorical_table,
        categorical_combined=categorical_combined_table,
        descriptive_table=descriptive_table,
        descriptive_combined=descriptive_combined_table,
    )

    write_html_report(
        path=paths["html_report"],
        contract_file_name=contract_file_name,
        contract_id=contract_id,
        summary=summary,
        categorical_table=categorical_table,
        categorical_combined=categorical_combined_table,
        descriptive_table=descriptive_table,
        descriptive_combined=descriptive_combined_table,
    )

    print(f"Details written to              : {paths['details']}")
    print(f"Summary written to              : {paths['summary']}")
    print(f"Failures written to             : {paths['failures']}")
    print(f"Metrics table written to        : {paths['metrics_table']}")
    print(f"Categorical table written       : {paths['categorical_table']}")
    print(f"Categorical combined written    : {paths['categorical_combined_table']}")
    print(f"Descriptive table written       : {paths['descriptive_table']}")
    print(f"Descriptive combined written    : {paths['descriptive_combined_table']}")
    print(f"Markdown report written to      : {paths['markdown_report']}")
    print(f"HTML report written to          : {paths['html_report']}")
    print(f"Browser link                    : file:///{paths['html_report'].resolve().as_posix()}")

    if not categorical_table.empty:
        print("\nCategorical LLM results:")
        print(categorical_table.to_string(index=False))

    if not categorical_combined_table.empty:
        print("\nCategorical LLM combined results:")
        print(categorical_combined_table.to_string(index=False))

    if not descriptive_table.empty:
        print("\nDescriptive LLM results:")
        print(descriptive_table.to_string(index=False))

    if not descriptive_combined_table.empty:
        print("\nDescriptive LLM combined results:")
        print(descriptive_combined_table.to_string(index=False))

    print("\nSummary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()