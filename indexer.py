from azure.search.documents import SearchIndexingBufferedSender
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    VectorSearchAlgorithmConfiguration,
    VectorSearchProfile,
    HnswAlgorithmConfiguration,
    AzureOpenAIVectorizer,
    VectorSearchVectorizer,
    AzureOpenAIVectorizerParameters,
    SemanticSearch,
    SemanticConfiguration,
    SemanticPrioritizedFields,
    SemanticField,
)
from azure.search.documents import SearchClient
import json
from azure.core.credentials import AzureKeyCredential
from main import (
    llm_chunking,
    scrape_website,
    chunk_text,
    generate_embeddings,
    Chunk,
    ChunkResponse,
)
import random
from dotenv import load_dotenv
import os


# Load environment variables from .env file
load_dotenv()

AZURE_SEARCH_SERVICE_ENDPOINT = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
AZURE_SEARCH_ADMIN_KEY = os.getenv("AZURE_SEARCH_ADMIN_KEY")
AZURE_SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME = os.getenv(
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"
)
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "1536"))

# Create a credential object
credential = AzureKeyCredential(AZURE_SEARCH_ADMIN_KEY)

index_client = SearchIndexClient(
    endpoint=AZURE_SEARCH_SERVICE_ENDPOINT, credential=credential
)


# https://github.com/Azure/azure-search-vector-samples/blob/main/demo-python/code/basic-vector-workflow/azure-search-vector-python-sample.ipynb
def create_or_update_search_index(
    index_name, embedding_dimensions, embedding_model_name
):
    # Define the vector search configuration
    vector_search = VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name="myHnsw")],
        profiles=[
            VectorSearchProfile(
                name="myHnswProfile",
                algorithm_configuration_name="myHnsw",
                vectorizer_name="myVectorizer",
            )
        ],
        vectorizers=[
            AzureOpenAIVectorizer(
                vectorizer_name="myVectorizer",
                parameters=AzureOpenAIVectorizerParameters(
                    resource_url=AZURE_OPENAI_ENDPOINT,
                    deployment_name=AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME,
                    model_name=AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME,
                    api_key=AZURE_OPENAI_API_KEY,
                ),
            )
        ],
    )

    fields = [
        SearchField(name="id", type=SearchFieldDataType.String, key=True),
        SearchField(name="text", type=SearchFieldDataType.String, searchable=True),
        SearchField(name="topic", type=SearchFieldDataType.String, filterable=True),
        SearchField(
            name="description", type=SearchFieldDataType.String, filterable=True
        ),
        SearchField(
            name="textVector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            vector_search_profile_name="myHnswProfile",
            vector_search_dimensions=embedding_dimensions,
            searchable=True,
            retrievable=True,
        ),
        SearchField(
            name="topicVector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            vector_search_profile_name="myHnswProfile",
            vector_search_dimensions=embedding_dimensions,
            searchable=True,
            retrievable=True,
        ),
        SearchField(
            name="descriptionVector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            vector_search_profile_name="myHnswProfile",
            vector_search_dimensions=embedding_dimensions,
            searchable=True,
            retrievable=True,
        ),
        # Add other metadata fields if needed, e.g., title, last_modified, etc.
        SearchField(
            name="keywords",
            type=SearchFieldDataType.Collection(SearchFieldDataType.String),
            searchable=True,
        ),
        SearchField(
            name="links",
            type=SearchFieldDataType.Collection(SearchFieldDataType.String),
            searchable=True,
        ),
    ]

    """semantic_search = SemanticSearch(
        default_configuration_name="semantic_config",
        configurations=[
            SemanticConfiguration(
                name="semantic_config",
                prioritized_fields=SemanticPrioritizedFields(
                    content_fields=[SemanticField(field_name="content")],
                    title_field=SemanticField(field_name="title"),
                    keywords_fields=[SemanticField(field_name="url")],
                ),
            )
        ],
    )"""

    index = SearchIndex(
        name=index_name,
        fields=fields,
        vector_search=vector_search,
    )
    print(f"Creating or updating index '{index_name}'...")
    index_client.create_or_update_index(index)
    print(f"Index '{index_name}' created/updated successfully.")


def upload_documents_to_azure_ai_search(index_name, documents):
    search_client = SearchClient(
        endpoint=AZURE_SEARCH_SERVICE_ENDPOINT,
        index_name=index_name,
        credential=credential,
    )

    print(f"Uploading {len(documents)} documents to index '{index_name}'...")
    try:
        search_client.upload_documents(documents=documents)
        print("Documents uploaded successfully.")
    except Exception as e:
        print(f"Error uploading documents: {e}")
    finally:
        search_client.close()


if __name__ == "__main__":

    embedding_model_name = (
        "text-embedding-3-large"  # Match this to your deployed embedding model name
    )
    create_or_update_search_index(AZURE_SEARCH_INDEX_NAME, 3072, embedding_model_name)

    # 2. Example: Scrape, Chunk, Embed, and Prepare Documents for Indexing
    website_urls = [
        "https://help.csod.com/lxp/Content/Analytics/EdData/DataAccess_Export.htm",
        # Add more URLs as needed
    ]

    documents_to_upload = []
    doc_id_counter = 0
    prompt = """You are an expert text analysis and summarization assistant. Your task is to process a given long document and break it down into semantically coherent, concise chunks of max 100 words. For each generated chunk, you must also extract specific metadata.

**Strict Output Format:**
Output ONLY a JSON array. Do not include any introductory text, concluding remarks, or explanations outside the JSON. The JSON must be valid and directly parsable.

**Chunking Rules:**
1.  **Semantic Coherence:** Each chunk must represent a single, complete idea, topic, or logical unit. Prioritize semantic completeness over strict length.
2.  **No Meaningful Overlap:** Chunks should ideally not overlap in content, unless absolutely necessary to keep a complete thought that spans a natural semantic break.
3.  **Target Length Guideline:** Aim for each chunk's `text` content to be approximately **200 to 500 words (or 800-2000 characters)**. Adjust as needed to maintain semantic coherence. If a natural semantic unit is slightly larger or smaller, keep it together.
4.  **No Broken Sentences:** Do not split a sentence across chunks.
5.  **Conciseness:** Ensure each chunk is as concise as possible while retaining its full meaning.

**Metadata to Extract for Each Chunk:**
For each chunk you create, provide the following fields:

* `text` (string): The full, raw text content of the semantically coherent chunk.
* `topic` (string): The overarching main subject or theme of *this specific chunk*, articulated in 3-7 words. Example: "Latest advancements in AI research".
* `description` (string): A single, concise sentence (max 25 words) that summarizes the core content of this chunk.
* `keywords` (array of strings): A list of 3 to 7 highly relevant key terms or phrases extracted directly from this chunk's content. Do not include generic stop words. Example: `["semantic chunking", "LLM applications", "RAG systems"]`.
* `links` (array of strings): A list of any full URLs (e.g., starting with "http://" or "https://", "www.") that are explicitly mentioned and fully visible *within the text content of this specific chunk*. Do not infer links. If no links are found, provide an empty array `[]`.

**Output JSON Schema:**
```json
[
  {
    "text": "The full text content of the first chunk.",
    "topic": "Concise topic of chunk 1",
    "description": "A single sentence describing the content of chunk 1.",
    "keywords": ["keyword1", "keyword2", "keyword3"],
    "links": ["http://example.com/link1", "https://another.link/"]
  },
  {
    "text": "The full text content of the second chunk.",
    "topic": "Concise topic of chunk 2",
    "description": "A single sentence describing the content of chunk 2.",
    "keywords": ["keywordA", "keywordB"],
    "links": []
  }
]"""

    for url in website_urls:
        print(f"\nProcessing URL: {url}")
        scraped_content = scrape_website(url)
        if scraped_content:
            resp: ChunkResponse = llm_chunking(scraped_content, prompt=prompt)

            for i, part in enumerate(resp.chunks):
                document_id = f"{random.randint(1000, 9999)}"
                documents_to_upload.append(
                    {
                        "id": document_id,
                        "text": part.text,
                        "topic": part.topic,
                        "description": part.description,
                        "keywords": part.keywords,
                        "links": part.links,
                        "textVector": generate_embeddings(part.text),
                        "topicVector": generate_embeddings(part.topic),
                        "descriptionVector": generate_embeddings(part.description),
                    }
                )

        else:
            print(f"No content scraped from {url}")

    if documents_to_upload:
        upload_documents_to_azure_ai_search(
            AZURE_SEARCH_INDEX_NAME, documents_to_upload
        )
    else:
        print("No documents to upload.")
