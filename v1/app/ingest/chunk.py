import re
from typing import Dict, Iterator, List

def _clean(text: str) -> str:
    text = text.replace("\r\n", "\n")
    # collapse excessive whitespace
    text = re.sub(r"[ \t]+", " ", text)
    # remove multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def chunk_contract_from_pages(
    page_text_iter: Iterator[str],
    *,
    target_chars: int = 1800,
    max_chars: int = 2400,
) -> Iterator[Dict]:
    """
    Fast, stable chunker:
    - Works page-by-page
    - Splits into paragraph-ish blocks
    - Accumulates blocks into chunks up to max_chars
    - Yields: {page, content}
    """
    chunk_idx = 0

    for page_num, page_text in enumerate(page_text_iter):
        page_text = _clean(page_text or "")
        if not page_text:
            continue

        # Split into blocks by blank lines (paragraph-ish)
        blocks = [b.strip() for b in page_text.split("\n\n") if b.strip()]

        buf: List[str] = []
        buf_len = 0

        def flush():
            nonlocal chunk_idx, buf, buf_len
            if not buf:
                return None
            content = "\n\n".join(buf).strip()
            buf = []
            buf_len = 0
            rec = {
                "chunk_id": f"c{chunk_idx:05d}",
                "page": page_num,
                "content": content
            }
            chunk_idx += 1
            return rec

        for b in blocks:
            # If a single block is huge, split it hard
            if len(b) > max_chars:
                # flush what we have
                out = flush()
                if out:
                    yield out
                # hard-split big block
                start = 0
                while start < len(b):
                    piece = b[start:start+max_chars].strip()
                    if piece:
                        yield {"chunk_id": f"c{chunk_idx:05d}", "page": page_num, "content": piece}
                        chunk_idx += 1
                    start += max_chars
                continue

            # If adding this block would overflow, flush first
            if buf_len + len(b) + 2 > max_chars:
                out = flush()
                if out:
                    yield out

            buf.append(b)
            buf_len += len(b) + 2

            # If we’re around target size, flush to keep chunks consistent
            if buf_len >= target_chars:
                out = flush()
                if out:
                    yield out

        out = flush()
        if out:
            yield out
