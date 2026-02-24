import os
import json
import base64
from pathlib import Path
from typing import List, Dict

from dotenv import load_dotenv
from tqdm import tqdm

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from openai import AzureOpenAI

load_dotenv()

# -------- Config --------
INDEX_NAME = os.environ.get("AZURE_SEARCH_INDEX_NAME", "contract-intelligence")
SEARCH_ENDPOINT = os.environ["AZURE_SEARCH_ENDPOINT"]
SEARCH_KEY = os.environ["AZURE_SEARCH_API_KEY"]

EMBED_ENDPOINT = os.environ["AZURE_OPENAI_EMBEDDINGS_ENDPOINT"]
EMBED_KEY = os.environ["AZURE_OPENAI_EMBEDDINGS_API_KEY"]
EMBED_VERSION = os.environ.get("AZURE_OPENAI_EMBEDDINGS_API_VERSION", "2024-02-15-preview")
EMBED_DEPLOYMENT = os.environ["AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT"]

EMBED_BATCH = int(os.environ.get("EMBED_BATCH", 8))
UPLOAD_BATCH = int(os.environ.get("UPLOAD_BATCH", 40))

PROCESSED_DIR = Path("v1/data/processed")
# ------------------------


def get_clients():
    sc = SearchClient(
        endpoint=SEARCH_ENDPOINT,
        index_name=INDEX_NAME,
        credential=AzureKeyCredential(SEARCH_KEY),
    )
    ec = AzureOpenAI(
        api_key=EMBED_KEY,
        api_version=EMBED_VERSION,
        azure_endpoint=EMBED_ENDPOINT,
    )
    return sc, ec


def load_checkpoint(path: Path) -> dict:
    if not path.exists():
        return {"line_num": 0}
    return json.loads(path.read_text(encoding="utf-8"))


def save_checkpoint(path: Path, data: dict):
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def embed_texts(ec: AzureOpenAI, texts: List[str]) -> List[List[float]]:
    resp = ec.embeddings.create(model=EMBED_DEPLOYMENT, input=texts)
    return [d.embedding for d in resp.data]


def safe_upload(sc: SearchClient, docs: List[Dict]):
    resp = sc.upload_documents(documents=docs)
    failures = [r for r in resp if not r.succeeded]
    if failures:
        print(f"\nWARNING: {len(failures)} docs failed to upload. Sample:")
        for f in failures[:5]:
            print(" - key:", f.key, "| error:", getattr(f, "error_message", ""))
    return resp


def make_safe_id(contract_id: str, chunk_id: str) -> str:
    raw = f"{contract_id}:{chunk_id}".encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii")


def main(contract_id: str):
    contract_folder = PROCESSED_DIR / contract_id

    if not contract_folder.exists():
        raise FileNotFoundError(
            f"Processed folder not found: {contract_folder.resolve()}"
        )

    chunks_path = contract_folder / "chunks.jsonl"
    checkpoint_path = contract_folder / "embed_checkpoint.json"

    print(f"Index: {INDEX_NAME}")
    print(f"Contract: {contract_id}")
    print(f"Chunks file: {chunks_path.resolve()}")
    print(f"Checkpoint: {checkpoint_path.resolve()}")

    ckpt = load_checkpoint(checkpoint_path)
    start_line = int(ckpt.get("line_num", 0))

    with chunks_path.open("r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)

    if start_line >= total_lines:
        print("Nothing to do: checkpoint indicates all chunks already processed.")
        return

    sc, ec = get_clients()

    with chunks_path.open("r", encoding="utf-8") as f:
        for _ in range(start_line):
            next(f)

        pbar = tqdm(total=total_lines, initial=start_line, desc="Embedding+Upserting", unit="chunk")
        line_num = start_line
        pending_records: List[Dict] = []

        def flush_upload(batch: List[Dict]):
            if batch:
                safe_upload(sc, batch)

        try:
            while True:
                line = f.readline()
                if not line:
                    break

                rec = json.loads(line)
                pending_records.append(rec)
                line_num += 1

                if len(pending_records) >= EMBED_BATCH:
                    texts = [r["content"] for r in pending_records]
                    vectors = embed_texts(ec, texts)

                    docs = []
                    for r, v in zip(pending_records, vectors):
                        safe_id = make_safe_id(r["contract_id"], r["chunk_id"])
                        docs.append({
                            "id": safe_id,
                            "contract_id": r["contract_id"],
                            "chunk_id": r["chunk_id"],
                            "content": r["content"],
                            "content_vector": v,
                        })

                    for i in range(0, len(docs), UPLOAD_BATCH):
                        flush_upload(docs[i:i + UPLOAD_BATCH])

                    pending_records.clear()
                    save_checkpoint(checkpoint_path, {"line_num": line_num})
                    pbar.update(EMBED_BATCH)

            # Flush remainder
            if pending_records:
                texts = [r["content"] for r in pending_records]
                vectors = embed_texts(ec, texts)

                docs = []
                for r, v in zip(pending_records, vectors):
                    safe_id = make_safe_id(r["contract_id"], r["chunk_id"])
                    docs.append({
                        "id": safe_id,
                        "contract_id": r["contract_id"],
                        "chunk_id": r["chunk_id"],
                        "content": r["content"],
                        "content_vector": v,
                    })

                for i in range(0, len(docs), UPLOAD_BATCH):
                    flush_upload(docs[i:i + UPLOAD_BATCH])

                save_checkpoint(checkpoint_path, {"line_num": line_num})
                pbar.update(len(pending_records))

        finally:
            pbar.close()

    print("Done. All chunks embedded and uploaded.")
    print(f"Final checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    raise RuntimeError(
        "This script now requires a contract_id argument. "
        "Run it through the pipeline (e2e_run.py)."
    )