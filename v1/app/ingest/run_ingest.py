from pathlib import Path
import json
from tqdm import tqdm

from v1.app.ingest.extract import extract_text_pdf_iter, extract_text_docx, extract_text_txt
from v1.app.ingest.chunk import chunk_contract_from_pages

RAW_DIR = Path("v1/data/raw")
OUT_DIR = Path("v1/data/processed")
SUPPORTED_EXTS = {".pdf", ".docx", ".txt"}


def main():
    files = [f for f in RAW_DIR.iterdir() if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS]
    if not files:
        raise FileNotFoundError(f"No supported contract files found in {RAW_DIR}")
    if len(files) > 1:
        raise RuntimeError(
            "Keep only ONE file in v1/data/raw for this run:\n"
            + "\n".join(f" - {f.name}" for f in files)
        )

    contract_path = files[0]
    ext = contract_path.suffix.lower()
    contract_id = contract_path.stem

    contract_out = OUT_DIR / contract_id
    contract_out.mkdir(parents=True, exist_ok=True)

    extracted_path = contract_out / "extracted.txt"
    chunks_path = contract_out / "chunks.jsonl"

    print(f"Ingesting contract: {contract_path.name}")
    print(f"Contract ID: {contract_id}")
    print(f"Saving to: {contract_out.resolve()}")

    if ext == ".pdf":
        page_iter = extract_text_pdf_iter(str(contract_path), start_page=0)
    elif ext == ".docx":
        page_iter = iter([extract_text_docx(str(contract_path))])
    else:
        page_iter = iter([extract_text_txt(str(contract_path))])

    def page_stream_and_save():
        with extracted_path.open("w", encoding="utf-8", errors="ignore") as out:
            for page_num, page_text in enumerate(page_iter):
                out.write(f"\n\n===== PAGE {page_num} =====\n")
                out.write(page_text or "")
                yield page_text or ""

    total = 0
    first_two = []

    with chunks_path.open("w", encoding="utf-8") as fout:
        for ch in tqdm(
            chunk_contract_from_pages(page_stream_and_save()),
            desc="Chunking",
            unit="chunk",
        ):
            total += 1
            rec = {
                "contract_id": contract_id,
                "source_file": contract_path.name,
                "chunk_id": ch["chunk_id"],
                "page": ch["page"],
                "content": ch["content"],
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if len(first_two) < 2:
                first_two.append(rec)

    print(f"\nExtracted text saved to: {extracted_path}")
    print(f"Chunks saved to        : {chunks_path}")
    print(f"Chunks created         : {total:,}")

    for idx, ch in enumerate(first_two):
        print("\n" + "=" * 80)
        print(f"Chunk {idx}")
        print(f"chunk_id : {ch['chunk_id']}")
        print(f"page     : {ch['page']}")
        print("-" * 80)
        print(ch["content"][:1200])

    return contract_id


if __name__ == "__main__":
    main()