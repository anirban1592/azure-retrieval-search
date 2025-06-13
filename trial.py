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
import json
from typing import List

# This script scrapes a webpage and prints the text content.
# It uses the requests library to fetch the page and BeautifulSoup to parse the HTML.
load_dotenv()

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


class Chunk(BaseModel):
    text: str
    topic: str
    description: str
    keywords: list[str]
    links: list[str]


class ChunkResponse(BaseModel):
    chunks: list[Chunk]


def scrape_website(url) -> str:
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check for HTTP errors
        soup = BeautifulSoup(response.text, "html.parser")
        text = soup.get_text(separator=" ", strip=True)
        return text
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None


endpoint = os.getenv(
    "AZURE_DEEPSEEK_ENDPOINT", "https://deepseekmodel.services.ai.azure.com/models"
)
model_name = os.getenv("AZURE_DEEPSEEK_MODEL_NAME", "DeepSeek-R1-0528")


def llm_chunking(text: str, prompt: str) -> ChunkResponse:
    """Generates text chunks using a language model."""
    credential = AzureKeyCredential(
        os.getenv("AZURE_DEEPSEEK_API_KEY", "DeepSeek-R1-0528")
    )

    client = ChatCompletionsClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(
            os.getenv("AZURE_DEEPSEEK_API_KEY", "DeepSeek-R1-0528")
        ),
        api_version="2024-05-01-preview",
    )

    response = client.complete(
        messages=[
            SystemMessage(content=prompt),
            UserMessage(content="Here is the text I want to chunk: " + text),
        ],
        max_tokens=2048,
        model=model_name,
    )
    json_str = response.choices[0].message.content
    chunks_data: List[Chunk] = [Chunk(**item) for item in json.loads(json_str)]
    chunks = ChunkResponse(chunks=chunks_data)

    return chunks


if __name__ == "__main__":
    url: str = "https://help.csod.com/lxp/Content/Analytics/Analytics_overview.htm"
    scrape_text = scrape_website(url)
    if scrape_text:
        print("Scraped text:", scrape_text)
        chunks = llm_chunking(scrape_text, prompt)
        # pretty print the object as json
        chunks_json = json.dumps(chunks.model_dump(), indent=2)
        print("Generated chunks JSON:", chunks_json)
