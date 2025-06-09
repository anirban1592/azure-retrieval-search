from langchain_openai import AzureChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from dotenv import load_dotenv
import os
from retrival_agent import retrive

load_dotenv()


@tool
def az_search_tool(chat_history: list[dict]):
    """A tool that returns the a final response with citations LLM's then need to use this data further to curate the final user friendly response.

    Args:
        chat_history (list[dict]): The chat history to look up. The format of the chat history is a list of dictionaries with `role` and `content` keys.
    """

    return retrive(chat_history)


prompt = """You are a helpful assistant. Use the `az_search_tool` tool to answer the user's question.
The tool will take the chat history as input and return a string with grounded information. You must use the data to build the final response to the user
The tool will return the final response in the format:
***
[{"ref_id": "<reference ID>", "content": "<content from vector db>", "terms":"<citation of why this content was selected, may be a link etc>"}]
***
You must use the data to build the final response to the user.
If the tool does not return any data, you must say "I don't know" in your final response.
As you can see, the tool returns a list of dictionaries with `ref_id` and `content` keys. You must consider all the content from the list and curate the final answer which is most relevant to then users query
Also use the `terms` field to provide additional context or justification for your answer.
"""

memory = MemorySaver()
model = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["AZURE_OPENAI_GPT_DEPLOYMENT_4.1"],
    openai_api_version=os.environ["OPENAI_API_VERSION"],
)

agent = create_react_agent(model=model, tools=[az_search_tool], prompt=prompt)


response = agent.invoke(
    {
        "messages": [
            AIMessage(
                content="You are an helpful assistant. Use the test tool to answer."
            ),
            HumanMessage(content="What is Web Creator?"),
        ],
    }
)
print("Response from agent:")
print(response["messages"][-1].content)
