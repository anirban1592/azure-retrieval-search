from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
import json
from azure.search.documents.models import VectorizableTextQuery
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchItemPaged
from dotenv import load_dotenv
import os


# Load environment variables from .env file
load_dotenv()

OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL")
AZURE_SEARCH_ADMIN_KEY = os.getenv("AZURE_SEARCH_ADMIN_KEY")
AZURE_SEARCH_SERVICE_ENDPOINT = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
AZURE_SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME")
credential = AzureKeyCredential(AZURE_SEARCH_ADMIN_KEY)


def print_results(results: SearchItemPaged[dict]):
    semantic_answers = results.get_answers()
    if semantic_answers:
        for answer in semantic_answers:
            if answer.highlights:
                print(f"Semantic Answer: {answer.highlights}")
            else:
                print(f"Semantic Answer: {answer.text}")
            print(f"Semantic Answer Score: {answer.score}\n")

    for result in results:
        print(f"Title: {result['title']}")
        print(f"Score: {result['@search.score']}")
        if result.get("@search.reranker_score"):
            print(f"Reranker Score: {result['@search.reranker_score']}")
        print(f"Content: {result['content']}")
        print(f"URL: {result['url']}\n")


search_client = SearchClient(
    endpoint=AZURE_SEARCH_SERVICE_ENDPOINT,
    index_name=AZURE_SEARCH_INDEX_NAME,
    credential=credential,
)

query = "Which enhancements are available"

vector_query = VectorizableTextQuery(
    text=query, k_nearest_neighbors=50, fields="contentVector"
)
results = search_client.search(
    search_text=None,
    vector_queries=[vector_query],
    select=["content", "url", "title"],
    top=3,
)

print_results(results)
