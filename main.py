import requests
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from openai import AzureOpenAI
from dotenv import load_dotenv
import os

# This script scrapes a webpage and prints the text content.
# It uses the requests library to fetch the page and BeautifulSoup to parse the HTML.


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


def generate_embeddings(text_chunks, use_azure_openai=False):
    embeddings_list = []
    if use_azure_openai:
        client = AzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_version="2024-02-01",  # Or your specific API version
        )
        for chunk in text_chunks:
            response = client.embeddings.create(
                input=[chunk], model=AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME
            )
            embeddings_list.append(response.data[0].embedding)
    else:
        client = OpenAI(api_key=OPENAI_API_KEY)
        for chunk in text_chunks:
            response = client.embeddings.create(
                input=[chunk], model=OPENAI_EMBEDDING_MODEL
            )
            embeddings_list.append(response.data[0].embedding)
    return embeddings_list


if __name__ == "__main__":

    print(scrape_website(website_url))
