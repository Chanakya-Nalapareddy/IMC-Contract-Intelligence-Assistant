# v1/app/rag/rag_chat.py
import os
from typing import List, Dict, Any, Optional

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from openai import AzureOpenAI


def _make_search_client() -> SearchClient:
    return SearchClient(
        endpoint=os.environ["AZURE_SEARCH_ENDPOINT"],
        index_name=os.environ["AZURE_SEARCH_INDEX_NAME"],
        credential=AzureKeyCredential(os.environ["AZURE_SEARCH_API_KEY"]),
    )


def _make_embed_client() -> AzureOpenAI:
    return AzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_EMBEDDINGS_API_KEY"],
        azure_endpoint=os.environ["AZURE_OPENAI_EMBEDDINGS_ENDPOINT"],
        api_version=os.environ["AZURE_OPENAI_EMBEDDINGS_API_VERSION"],
    )


def _make_chat_client() -> AzureOpenAI:
    return AzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    )


def embed_text(text: str) -> List[float]:
    ec = _make_embed_client()
    resp = ec.embeddings.create(
        model=os.environ["AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT"],
        input=text,
    )
    return resp.data[0].embedding


def retrieve_chunks(
    query: str,
    k: int = 8,
    contract_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Returns a list of evidence chunks:
      [{"chunk_id":..., "content":..., "page":..., "@search.score":...}, ...]
    Adjust select fields to your index schema.
    """
    sc = _make_search_client()
    qvec = embed_text(query)

    vector_query = {"vector": qvec, "k": k, "fields": "content_vector"}

    # Optional filter if you have doc_id or contract_id in the index
    filter_expr = None
    if contract_id:
        # Change 'doc_id' to whatever your index uses
        filter_expr = f"doc_id eq '{contract_id}'"

    results = sc.search(
        search_text=None,
        vector_queries=[vector_query],
        filter=filter_expr,
        select=["chunk_id", "content", "page", "doc_id"],
        top=k,
    )

    out = []
    for r in results:
        out.append({
            "chunk_id": r.get("chunk_id"),
            "content": r.get("content"),
            "page": r.get("page", None),
            "score": r.get("@search.score", None),
            "doc_id": r.get("doc_id", None),
        })
    return out


def answer_question(
    question: str,
    chat_history: List[Dict[str, str]],
    evidence_chunks: List[Dict[str, Any]],
) -> str:
    """
    Builds grounded answer using provided evidence. Does NOT include evidence in the answer unless asked.
    """
    excerpts = "\n\n".join(
        f"[{c.get('chunk_id')}] page={c.get('page')}\n{(c.get('content') or '')[:2000]}"
        for c in evidence_chunks
        if c.get("content")
    )

    system = """You are IMC Contract Intelligence Assistant.
Answer only using the provided contract excerpts.
If the answer is not in the excerpts, say you cannot find it in the contract.
Be concise.
"""

    messages = [{"role": "system", "content": system}]
    for m in chat_history[-10:]:
        if m.get("role") in ("user", "assistant"):
            messages.append({"role": m["role"], "content": m.get("content", "")})

    messages.append({
        "role": "user",
        "content": f"""Question:
{question}

Contract excerpts:
{excerpts}
"""
    })

    cc = _make_chat_client()
    resp = cc.chat.completions.create(
        model=os.environ["AZURE_DEPLOYMENT_NAME"],
        messages=messages,
        temperature=0.0,
    )
    return (resp.choices[0].message.content or "").strip()


def rag_chat_turn(
    question: str,
    chat_history: List[Dict[str, str]],
    k: int = 8,
    contract_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    One RAG chat turn.
    Returns:
      {
        "answer": "...",
        "evidence": [{"chunk_id", "page", "content", "score"}...]
      }
    """
    evidence = retrieve_chunks(query=question, k=k, contract_id=contract_id)
    answer = answer_question(question=question, chat_history=chat_history, evidence_chunks=evidence)
    return {"answer": answer, "evidence": evidence}
