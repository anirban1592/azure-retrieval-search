from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
import json
from azure.search.documents.models import VectorizableTextQuery
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchItemPaged
from dotenv import load_dotenv
import os
from openai import AzureOpenAI


# Load environment variables from .env file
load_dotenv()

OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL")
AZURE_SEARCH_ADMIN_KEY = os.getenv("AZURE_SEARCH_ADMIN_KEY")
AZURE_SEARCH_SERVICE_ENDPOINT = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
AZURE_SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
credential = AzureKeyCredential(AZURE_SEARCH_ADMIN_KEY)


def search_index(query: str) -> list[str]:
    """Searches the Azure Search index using a vector query and returns the top results."""
    search_client = SearchClient(
        endpoint=AZURE_SEARCH_SERVICE_ENDPOINT,
        index_name=AZURE_SEARCH_INDEX_NAME,
        credential=credential,
    )
    vector_query = VectorizableTextQuery(
        text=query,
        k_nearest_neighbors=50,
        fields="textVector,topicVector,descriptionVector",
    )

    results = search_client.search(
        search_text=None,
        vector_queries=[vector_query],
        select=["text", "topic", "description"],
        top=3,
    )

    search_response = [result["text"] for result in results]

    return search_response


def generate_llm_response(search_response: list[str], user_query: str) -> None:

    endpoint = os.getenv(
        "AZURE_OPENAI_ENDPOINT",
        "https://azopenai1592.openai.azure.com/openai/deployments/gpt-4.1-nano",
    )
    model_name = os.getenv("AZURE_OPENAI_GPT_DEPLOYMENT", "gpt-4.1-nano")
    deployment = model_name
    subscription_key = AZURE_OPENAI_API_KEY
    api_version = os.getenv("OPENAI_API_VERSION", "2024-02-01")
    dynamic_search_response = json.dumps(search_response, indent=2)
    SYSTEM_PROMPT = (
        "You are a helpful assistant. Use the following information to answer the user's question. If you don't know the answer, say 'I don't know'.\n\n"
        + dynamic_search_response
        + "\n\n"
        "Be precise and concise in your answers. Do not include any additional information that is not relevant to the user's question.\n\n"
    )

    client = AzureOpenAI(
        api_version=api_version,
        azure_endpoint=endpoint,
        api_key=subscription_key,
    )

    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_query,
            },
        ],
        max_completion_tokens=800,
        temperature=1.0,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        model=deployment,
    )

    print(
        "A: " + response.choices[0].message.content
    )  # Placeholder for LLM response generation logic


if __name__ == "__main__":

    query = input("Q: ")
    generate_llm_response(search_index(query), query)
    while query.lower() != "exit":
        query = input("Q: ")
        if query.lower() == "exit":
            break
        generate_llm_response(search_index(query), query)
