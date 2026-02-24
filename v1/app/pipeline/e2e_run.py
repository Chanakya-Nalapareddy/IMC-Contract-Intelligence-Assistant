# v1/app/pipeline/e2e_run.py
# py -m v1.app.pipeline.e2e_run

from v1.app.ingest.run_ingest import main as ingest
from v1.app.ingest.embed_upsert import main as embed
from v1.app.rag.batch_run import main as batch


def main():
    print("Step 1: Ingesting contract...")
    contract_id = ingest()   # ← get contract_id from Step 1

    print("Step 2: Embedding + uploading to Azure Search...")
    embed(contract_id)       # ← pass it explicitly

    print("Step 3: Running batch abstraction...")
    batch(contract_id)       # ← pass it explicitly

    print("\n✅ End-to-end pipeline completed successfully.")


if __name__ == "__main__":
    main()