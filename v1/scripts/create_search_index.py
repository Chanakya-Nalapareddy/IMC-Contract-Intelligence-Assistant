import os
from dotenv import load_dotenv

from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    VectorSearchProfile,
    HnswAlgorithmConfiguration,
)

from openai import AzureOpenAI

load_dotenv()

SEARCH_ENDPOINT = os.environ["AZURE_SEARCH_ENDPOINT"]
SEARCH_KEY = os.environ["AZURE_SEARCH_API_KEY"]
INDEX_NAME = os.environ.get("AZURE_SEARCH_INDEX_NAME", "insightmatrix-rag")

EMBED_ENDPOINT = os.environ["AZURE_OPENAI_EMBEDDINGS_ENDPOINT"]
EMBED_KEY = os.environ["AZURE_OPENAI_EMBEDDINGS_API_KEY"]
EMBED_VERSION = os.environ.get("AZURE_OPENAI_EMBEDDINGS_API_VERSION", "2024-02-15-preview")
EMBED_DEPLOYMENT = os.environ["AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT"]


def get_embedding_dim() -> int:
    client = AzureOpenAI(
        api_key=EMBED_KEY,
        api_version=EMBED_VERSION,
        azure_endpoint=EMBED_ENDPOINT,
    )
    resp = client.embeddings.create(
        model=EMBED_DEPLOYMENT,
        input="dimension check"
    )
    return len(resp.data[0].embedding)


def build_index(dim: int) -> SearchIndex:
    # Vector search config (HNSW)
    vector_search = VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name="hnsw")],
        profiles=[VectorSearchProfile(name="vprofile", algorithm_configuration_name="hnsw")],
    )

    fields = [
        # Unique document key: contract_id + chunk_id
        SimpleField(name="id", type=SearchFieldDataType.String, key=True, filterable=True),

        # Core metadata
        SimpleField(name="contract_id", type=SearchFieldDataType.String, filterable=True, sortable=True, facetable=True),
        SimpleField(name="chunk_id", type=SearchFieldDataType.String, filterable=True, sortable=True, facetable=False),
        SimpleField(name="source_file", type=SearchFieldDataType.String, filterable=True, sortable=True, facetable=True),

        # Optional but helpful for citations
        SearchableField(name="section", type=SearchFieldDataType.String, filterable=True, sortable=True, facetable=True),
        SimpleField(name="page_start", type=SearchFieldDataType.Int32, filterable=True, sortable=True, facetable=True),
        SimpleField(name="page_end", type=SearchFieldDataType.Int32, filterable=True, sortable=True, facetable=True),

        # The chunk text
        SearchableField(name="content", type=SearchFieldDataType.String),

        # The vector embedding
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=dim,
            vector_search_profile_name="vprofile",
        ),
    ]

    return SearchIndex(name=INDEX_NAME, fields=fields, vector_search=vector_search)


def main():
    dim = get_embedding_dim()
    print(f"Detected embedding dimension: {dim}")

    index_client = SearchIndexClient(SEARCH_ENDPOINT, AzureKeyCredential(SEARCH_KEY))

    # Create or replace index
    index = build_index(dim)
    index_client.create_or_update_index(index)
    print(f"Index created/updated: {INDEX_NAME}")


if __name__ == "__main__":
    main()
