import json
from typing import List
import requests
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from openai import AzureOpenAI
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv
from pydantic import BaseModel
from azure.ai.inference.models import SystemMessage, UserMessage
import os

# This script scrapes a webpage and prints the text content.
# It uses the requests library to fetch the page and BeautifulSoup to parse the HTML.


class Chunk(BaseModel):
    text: str
    topic: str
    description: str
    keywords: list[str]
    links: list[str]


class ChunkResponse(BaseModel):
    chunks: list[Chunk]


def scrape_website(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check for HTTP errors
        soup = BeautifulSoup(response.text, "html.parser")
        text = soup.get_text(separator=" ", strip=True)
        return text
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None


website_url = "https://help.csod.com/skills-studio/ss-help-whats-new/topics/ss-whats-new-enhancements-LXTabFix.html"

# scraped_text = scrape_website(website_url)
# if scraped_text:
#    print(scraped_text) # Print first 500 characters


def chunk_text(text, chunk_size=200, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_text(text)
    print(f"Total chunks created: {len(chunks)}")
    return chunks


# Example usage of chunk_text function
# print("Chunking text...")
# chunks = chunk_text(scrape_website(website_url))
# for i, chunk in enumerate(chunks):
# print(f"Chunk {i+1}:")
# print(chunk)
# print("-" * 40)  # Separator for readability

# Load environment variables from .env file

# Load .env file
load_dotenv()

# Get environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME = os.getenv(
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME", "text-embedding-3-large"
)
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "1536"))


def generate_embeddings(text: str) -> str:
    embeddings_list = []

    client = AzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_version="2024-02-01",  # Or your specific API version
    )

    response = client.embeddings.create(
        input=[text], model=AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME
    )

    return response.data[0].embedding


endpoint = os.getenv(
    "AZURE_OPENAINANO_ENDPOINT",
    "https://azopenai1592.openai.azure.com/openai/deployments/gpt-4.1-nano",
)
model_name = os.getenv("AZURE_OPENAI_GPT_MODEL", "gpt-4.1-nano")


def llm_chunking(text: str, prompt: str) -> ChunkResponse:
    """Generates text chunks using a language model."""

    client = ChatCompletionsClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(
            os.getenv("AZURE_OPENAI_API_KEY", "random-key-1234567890")
        ),
        api_version="2024-05-01-preview",
    )

    response = client.complete(
        messages=[
            SystemMessage(content=prompt),
            UserMessage(content="Please chunk the following text: " + text),
        ],
        max_tokens=2048,
        model=model_name,
    )
    json_str = response.choices[0].message.content
    print(f"JSON response from LLM: {json_str}")
    chunks_data: List[Chunk] = [Chunk(**item) for item in json.loads(json_str)]
    chunks = ChunkResponse(chunks=chunks_data)

    return chunks


if __name__ == "__main__":

    print(scrape_website(website_url))
