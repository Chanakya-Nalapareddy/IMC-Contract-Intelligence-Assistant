# v1/app/pipeline/e2e_run.py
# py -m v1.app.pipeline.e2e_run

from pathlib import Path

from v1.app.ingest.run_ingest import main as ingest
from v1.app.ingest.embed_upsert import main as embed
from v1.app.rag.batch_run import main as batch
from v1.app.evaluation.evaluation import main as evaluate
from v1.app.reporting.results_pdf import build_results_pdf

PROCESSED_DIR = Path("v1/data/processed")


def delete_embed_checkpoint(contract_id: str) -> None:
    checkpoint = PROCESSED_DIR / contract_id / "embed_checkpoint.json"

    if checkpoint.exists():
        print(f"Removing old embedding checkpoint: {checkpoint}")
        checkpoint.unlink()


def main() -> None:
    print("Step 1: Ingesting contract...")
    contract_id = ingest()

    # Remove previous embedding checkpoint so embeddings run fresh
    delete_embed_checkpoint(contract_id)

    print("Step 2: Embedding + uploading to Azure Search...")
    embed(contract_id)

    print("Step 3: Running batch abstraction...")
    batch(contract_id)

    print("Step 4: Evaluating results against golden answers...")
    try:
        evaluate()
    except ValueError as e:
        msg = str(e)
        if 'No Excel row found for "' in msg:
            print("Evaluation skipped: no matching gold-answer row found in Excel.")
            print(msg)
        else:
            raise

    print("Step 5: Generating Q&A PDF...")
    results_path = PROCESSED_DIR / contract_id / "results.jsonl"
    pdf_path = build_results_pdf(results_path)
    print(f"PDF written to: {pdf_path}")

    print("\n✅ End-to-end pipeline completed successfully.")


if __name__ == "__main__":
    main()