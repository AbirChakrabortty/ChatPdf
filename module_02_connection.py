from dotenv import load_dotenv
from langchain_groq import ChatGroq

# ChatPromptTemplate lets us create reusable prompt templates
# with System + Human message pairs
from langchain_core.prompts import ChatPromptTemplate

# StrOutputParser converts the AI's response object into a plain string
# Without this, we'd get an AIMessage object instead of just text
from langchain_core.output_parsers import StrOutputParser

# Load the API key from .env
load_dotenv()

llm = ChatGroq(
    model = "qwen/qwen3.8-27b",
    temperature = .7
)
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a Senior Research Analyst with 20 years of experience.
Your job is to provide clear, concise, and accurate information.

STRICT RULES:
- Return ONLY the answer without any preamble or introduction
- Do NOT start with "Sure", "Of course", "Great question" etc.
- Be direct and factual
- Use bullet points when listing multiple items
- Keep the response under 100 words"""
    ),
    (
        "human",
        "Explain the following topic: {topic}"
    )
])

#create a rag chain
chain = prompt | llm | StrOutputParser()

topic = input('enter the topic')

response = chain.invoke({'topic' : topic})

print(response)