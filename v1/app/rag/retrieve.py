import os
from typing import List, Dict

from dotenv import load_dotenv
from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient

load_dotenv()

SEARCH_ENDPOINT = os.environ["AZURE_SEARCH_ENDPOINT"]
SEARCH_KEY = os.environ["AZURE_SEARCH_API_KEY"]
INDEX_NAME = os.environ.get("AZURE_SEARCH_INDEX_NAME", "contract-intelligence")

EMBED_ENDPOINT = os.environ["AZURE_OPENAI_EMBEDDINGS_ENDPOINT"]
EMBED_KEY = os.environ["AZURE_OPENAI_EMBEDDINGS_API_KEY"]
EMBED_VERSION = os.environ.get("AZURE_OPENAI_EMBEDDINGS_API_VERSION", "2024-02-15-preview")
EMBED_DEPLOYMENT = os.environ["AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT"]

def _search_client() -> SearchClient:
    return SearchClient(
        endpoint=SEARCH_ENDPOINT,
        index_name=INDEX_NAME,
        credential=AzureKeyCredential(SEARCH_KEY),
    )

def _embed_client() -> AzureOpenAI:
    return AzureOpenAI(
        api_key=EMBED_KEY,
        api_version=EMBED_VERSION,
        azure_endpoint=EMBED_ENDPOINT,
    )

def embed_query(text: str) -> List[float]:
    ec = _embed_client()
    resp = ec.embeddings.create(model=EMBED_DEPLOYMENT, input=text)
    return resp.data[0].embedding

def retrieve(contract_id: str, question: str, k: int = 8) -> List[Dict]:
    sc = _search_client()
    qvec = embed_query(question)

    results = sc.search(
        search_text=question,
        filter=f"contract_id eq '{contract_id}'",
        select=["id", "contract_id", "chunk_id", "content"],
        top=k,
        vector_queries=[{
            "kind": "vector",
            "vector": qvec,
            "fields": "content_vector",
            "k": k
        }],
    )

    hits = []
    for r in results:
        hits.append({
            "id": r.get("id"),
            "chunk_id": r.get("chunk_id"),
            "content": r.get("content", ""),
        })
    return hits
