# v1/app/rag/batch_run.py

import json
from pathlib import Path
from tqdm import tqdm

from v1.app.rag.retrieve import retrieve
from v1.app.rag.answer import answer

# --- Defaults ---
QUESTIONS_PATH = Path("v1/data/questions.jsonl")
OUTPUT_PATH = Path("v1/data/results.jsonl")
TOP_K = 8

SKIP_LINES = {
    "id",
    "start time",
    "completion time",
    "email",
    "name",
}


def load_questions(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing questions file: {path.resolve()}")

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    if text.startswith("["):
        data = json.loads(text)
        return [
            {
                "question": obj.get("question", "").strip(),
                "expected_type": obj.get("expected_type", "text"),
            }
            for obj in data
            if obj.get("question", "").strip()
            and obj.get("question", "").strip().lower() not in SKIP_LINES
        ]

    questions = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            q = line.strip()
            if not q:
                continue
            if q.lower() in SKIP_LINES:
                continue
            try:
                obj = json.loads(q)
                questions.append(
                    {
                        "question": obj.get("question", "").strip(),
                        "expected_type": obj.get("expected_type", "text"),
                    }
                )
            except:
                questions.append({"question": q, "expected_type": "text"})
    return questions


def main(contract_id: str):
    print(f"Using contract_id: {contract_id}")

    questions = load_questions(QUESTIONS_PATH)
    if not questions:
        raise RuntimeError("No questions found.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as out:
        for item in tqdm(questions, desc="Abstracting", unit="q"):
            question = item["question"]
            expected_type = item.get("expected_type", "text")

            excerpts = retrieve(contract_id, question, k=TOP_K)

            if not excerpts:
                result = {
                    "question": question,
                    "expected_type": expected_type,
                    "value": None,
                    "raw_answer": None,
                    "citations": [],
                    "notes": "No search hits returned.",
                }
            else:
                resp = answer(question, expected_type, excerpts)
                result = {
                    "question": question,
                    "expected_type": expected_type,
                    "value": resp.get("value"),
                    "raw_answer": resp.get("raw_answer"),
                    "citations": resp.get("citations", []),
                }
                if "notes" in resp:
                    result["notes"] = resp["notes"]

            out.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"\nDone. Results written to: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    raise RuntimeError(
        "This script now requires a contract_id. "
        "Run through v1.app.pipeline.e2e_run instead."
    )