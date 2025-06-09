from azure.search.documents.agent import KnowledgeAgentRetrievalClient
from azure.search.documents.agent.models import (
    KnowledgeAgentRetrievalRequest,
    KnowledgeAgentMessage,
    KnowledgeAgentMessageTextContent,
    KnowledgeAgentIndexParams,
)
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv
import os

load_dotenv()

instructions = """
You are a helpful assistant
"""

endpoint = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
agent_name = os.getenv("AGENT_NAME")
AZURE_SEARCH_ADMIN_KEY = os.getenv("AZURE_SEARCH_ADMIN_KEY")
credential = AzureKeyCredential(AZURE_SEARCH_ADMIN_KEY)
AZURE_SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME")

"""messages = [
    {"role": "assistant", "content": instructions},
    {"role": "user", "content": "What enhancements are available?"},
]




retrieval_result = agent_client.retrieve(
    retrieval_request=KnowledgeAgentRetrievalRequest(
        messages=[
            KnowledgeAgentMessage(
                role=msg["role"],
                content=[KnowledgeAgentMessageTextContent(text=msg["content"])],
            )
            for msg in messages
        ],
        target_index_params=[
            KnowledgeAgentIndexParams(
                index_name=AZURE_SEARCH_INDEX_NAME, reranker_threshold=2.5
            )
        ],
    )
)

messages.append(
    {"role": "assistant", "content": retrieval_result.response[0].content[0].text}
)

print("Messages")
for message in messages:
    print(f"{message['role'].capitalize()}: {message['content']}")
    print("-" * 40)  # Separator for readability"""


def retrive(chat_history: list[dict]):
    """A tool that returns the a final response with citations LLM's then need to use this data further to curate the final user friendly response.
    Args:
        chat_history (list[dict]): The chat history to look up."""

    agent_client = KnowledgeAgentRetrievalClient(
        endpoint=endpoint, agent_name=agent_name, credential=credential
    )

    messages = [{"role": "assistant", "content": instructions}, *chat_history]

    retrieval_result = agent_client.retrieve(
        retrieval_request=KnowledgeAgentRetrievalRequest(
            messages=[
                KnowledgeAgentMessage(
                    role=msg["role"],
                    content=[KnowledgeAgentMessageTextContent(text=msg["content"])],
                )
                for msg in messages
            ],
            target_index_params=[
                KnowledgeAgentIndexParams(
                    index_name=AZURE_SEARCH_INDEX_NAME, reranker_threshold=2.5
                )
            ],
        )
    )

    return retrieval_result.response[0].content[0].text
