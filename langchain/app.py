from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

import getpass
import os

if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

from langchain.chat_models import init_chat_model

model = init_chat_model("gpt-4", model_provider="openai")

# Create the agent
memory = MemorySaver()
search = TavilySearchResults(max_results=2)
tools = [search]

agent_executor = create_react_agent(model, tools, checkpointer=memory)

config = {"configurable": {"thread_id": "abc123"}}

user_input = input("Enter your query: ")

for step, metadata in agent_executor.stream(
    {"messages": [HumanMessage(content=user_input)]},
    config,
    stream_mode="messages",
):
    if metadata["langgraph_node"] == "agent" and (text := step.text()):
        print(text, end="|")

user_input_2 = input("Enter your follow up query: ")

for step, metadata in agent_executor.stream(
    {"messages": [HumanMessage(content=user_input_2)]},
    config,
    stream_mode="messages",
):
    if metadata["langgraph_node"] == "agent" and (text := step.text()):
        print(text, end="|")