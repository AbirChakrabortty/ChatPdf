from dotenv import load_dotenv

# `ChatGroq` is LangChain's connector to Groq's API
# It handles all the HTTP requests, JSON parsing, etc. for us
from langchain_groq import ChatGroq

# `HumanMessage` represents a message from the user
# (There are also SystemMessage, AIMessage — we'll use those in Module 2)
from langchain_core.messages import HumanMessage

load_dotenv()

llm = ChatGroq(
    model = "qwen/qwen3.8-27b",
    temperature = 0
)

ques = input('enter the question:')
response = llm.invoke([HumanMessage(content=ques)])

print(f'Ai response: {response.content}')