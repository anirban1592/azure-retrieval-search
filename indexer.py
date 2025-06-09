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
from main import scrape_website, chunk_text, generate_embeddings
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
        SearchField(name="content", type=SearchFieldDataType.String, searchable=True),
        SearchField(name="url", type=SearchFieldDataType.String, filterable=True),
        SearchField(
            name="contentVector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            vector_search_profile_name="myHnswProfile",
            vector_search_dimensions=embedding_dimensions,
            searchable=True,
            retrievable=True,
        ),
        # Add other metadata fields if needed, e.g., title, last_modified, etc.
        SearchField(name="title", type=SearchFieldDataType.String, searchable=True),
    ]

    semantic_search = SemanticSearch(
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
    )

    index = SearchIndex(
        name=index_name,
        fields=fields,
        vector_search=vector_search,
        semantic_search=semantic_search,
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
        "https://help.csod.com/myguide/Content/MyGuide/SolutionsLibrary.htm",
        "https://help.csod.com/myguide/Content/MyGuide/Insights/Insights.htm",
        "https://help.csod.com/myguide/Content/MyGuide/Creator/Creator.htm",
        # Add more URLs as needed
    ]

    documents_to_upload = []
    doc_id_counter = 0

    for url in website_urls:
        print(f"\nProcessing URL: {url}")
        scraped_content = scrape_website(url)
        if scraped_content:
            text_chunks = chunk_text(scraped_content)
            print(text_chunks)
            embeddings = generate_embeddings(text_chunks, use_azure_openai=True)
            print(f"***************{len(embeddings)}")
            for i, chunk in enumerate(text_chunks):

                if i < len(embeddings):
                    document_id = f"{random.randint(1000, 9999)}"
                    documents_to_upload.append(
                        {
                            "id": document_id,
                            "content": chunk,
                            "url": url,
                            "contentVector": embeddings[i],
                            "title": f"Content from {url} - Part {i+1}",  # Example title
                        }
                    )

                else:
                    print(f"Warning: No embedding for chunk {i} of {url}. Skipping.")
        else:
            print(f"No content scraped from {url}")

    if documents_to_upload:
        upload_documents_to_azure_ai_search(
            AZURE_SEARCH_INDEX_NAME, documents_to_upload
        )
    else:
        print("No documents to upload.")
