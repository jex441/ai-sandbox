from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

model = init_chat_model("gpt-4o-mini", model_provider="openai")

messages = [
    SystemMessage("Translate the following from English into Japanese"),
    HumanMessage("hi! how are you?"),
]

response = model.invoke(messages)

print(response.content)