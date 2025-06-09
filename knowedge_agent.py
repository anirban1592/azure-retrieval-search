from azure.search.documents.indexes.models import (
    KnowledgeAgent,
    KnowledgeAgentAzureOpenAIModel,
    KnowledgeAgentTargetIndex,
    KnowledgeAgentRequestLimits,
    AzureOpenAIVectorizerParameters,
)
from indexer import index_client
from dotenv import load_dotenv
import os

load_dotenv()

AGENT_NAME = os.getenv("AGENT_NAME")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_GPT_DEPLOYMENT = os.getenv("AZURE_OPENAI_GPT_DEPLOYMENT")
AZURE_OPENAI_GPT_MODEL = os.getenv("AZURE_OPENAI_GPT_MODEL")
INDEX_NAME = os.getenv("INDEX_NAME")


agent = KnowledgeAgent(
    name=AGENT_NAME,
    models=[
        KnowledgeAgentAzureOpenAIModel(
            azure_open_ai_parameters=AzureOpenAIVectorizerParameters(
                resource_url=AZURE_OPENAI_ENDPOINT,
                deployment_name=AZURE_OPENAI_GPT_DEPLOYMENT,
                model_name=AZURE_OPENAI_GPT_MODEL,
                api_key=os.getenv("AZURE_OPENAI_API_KEY", "YOUR_AZURE_OPENAI_API_KEY"),
            )
        )
    ],
    target_indexes=[
        KnowledgeAgentTargetIndex(index_name=INDEX_NAME, default_reranker_threshold=2.5)
    ],
)


def create_agent():
    """Create or update the knowledge agent in Azure Search."""
    try:
        index_client.create_or_update_agent(agent)
        print(f"Knowledge agent '{AGENT_NAME}' created or updated successfully")
    except Exception as e:
        print(f"Error creating/updating knowledge agent: {e}")


def delete_agent():
    """Delete the knowledge agent from Azure Search."""
    try:
        index_client.delete_agent(AGENT_NAME)
        print(f"Knowledge agent '{AGENT_NAME}' deleted successfully")
    except Exception as e:
        print(f"Error deleting knowledge agent: {e}")


if __name__ == "__main__":
    # Create or update the knowledge agent
    # create_agent()

    # Uncomment to delete the agent
    delete_agent()

# Note: The agent can be used in your application to handle queries and return results.
# You would typically call the agent with a query and process the response accordingly.
