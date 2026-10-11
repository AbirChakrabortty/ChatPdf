"""
============================================================
 MODULE 5: The Complete RAG Chain (Bringing It All Together!)
============================================================

 WHAT WE'RE DOING:
    This is where everything comes together! We're building a complete
    RAG (Retrieval Augmented Generation) pipeline:

     PDF → ✂️ Chunks → Embeddings → ChromaDB → Retrieve → Answer

    Here's how it works:
      1. User asks a question
      2. We search ChromaDB for the most relevant chunks
      3. We send those chunks + the question to the LLM
      4. The LLM reads the chunks and gives a grounded answer

 WHY RAG?
    Without RAG, the AI can only answer from its training data.
    With RAG, the AI can answer from YOUR documents!

    It's like giving the AI an open-book exam instead of a memory test.

⚠️ THE PROBLEM (HALLUCINATIONS):
    You ask: "What is the company's revenue?"
    The PDF doesn't mention revenue at all.
    But the AI MAKES UP a number!
    This is called a "hallucination."

 THE SOLUTION:
    In our prompt, we add a strict rule:
    "If the answer is NOT in the provided context, strictly say
     'I don't know based on the provided document.'"

▶️ HOW TO RUN:
    python module_05_rag_chain.py

    NOTE: Update `pdf_path` with your PDF file path.
"""

# ============================================================
# STEP 1: Imports
# ============================================================

from dotenv import load_dotenv

# LLM
from langchain_groq import ChatGroq

# Prompt
from langchain_core.prompts import ChatPromptTemplate

# PDF Loading & Chunking
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Embeddings & Vector Store
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# RAG Chain builders
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain

# Warnings
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Load secrets
load_dotenv()


# ============================================================
# STEP 2: Initialize ALL the components
# ============================================================

# --- The Brain (LLM) ---
print("Initializing LLM...")

llm = ChatGroq(
    model="qwen/qwen3.8-27b",
    temperature=0
)

print("LLM ready!")


# --- The Embedding Model ---
print("Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

print("Embedding model ready!\n")


# ============================================================
# STEP 3: Load, Chunk, and Store the PDF
# ============================================================

pdf_path = "TransformerAttenctionMechanism.pdf"

print(f"Loading PDF: {pdf_path}")

loader = PyPDFLoader(pdf_path)

pages = loader.load()

print(f"Loaded {len(pages)} pages")


print("✂️ Chunking the document...")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(pages)

print(f"   → {len(chunks)} chunks created")


print("Storing in ChromaDB...")

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db",
    collection_name="pdf_collection"
)

print("   → Vector store ready!\n")


# ============================================================
# STEP 4: Create the RAG Prompt
# ============================================================

rag_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful assistant that answers questions based ONLY
on the provided context from a PDF document.

RULES:
- Answer the question using ONLY the information in the context below
- Be concise and direct in your response
- If the context contains relevant information, provide a clear answer
- Use bullet points for listing multiple items

IMPORTANT: If the answer is NOT in the provided context, strictly say:
"I don't know based on the provided document."

Do NOT make up or assume any information.

CONTEXT:
{context}"""
    ),
    (
        "human",
        "{input}"
    )
])


# ============================================================
# STEP 5: Build the RAG Chain
# ============================================================

# Step A: Create a chain that processes the retrieved documents

document_chain = create_stuff_documents_chain(
    llm,
    rag_prompt
)


# Step B: Turn our vector store into a retriever

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)


# Step C: Create the final RAG chain

rag_chain = create_retrieval_chain(
    retriever,
    document_chain
)

print("RAG Chain is ready!\n")


# ============================================================
# STEP 6: Ask Questions!
# ============================================================

# Question that should be related to the PDF
question1 = "What is the dimension of the embedding vector?"

print(f"Question 1: {question1}\n")

response1 = rag_chain.invoke({
    "input": question1
})

print(f"Answer: {response1['answer']}\n")


# Let's see which chunks were used to answer
print(f"Sources used ({len(response1['context'])} chunks):")

for i, doc in enumerate(response1["context"], 1):

    page_num = doc.metadata.get("page", "?")

    preview = doc.page_content[:100].replace("\n", " ")

    print(
        f"   {i}. Page {page_num}: {preview}..."
    )


# ============================================================
# STEP 7: Test the Hallucination Guard
# ============================================================

print("\n" + "=" * 50)

print("HALLUCINATION TEST\n")


# This question is probably NOT in your PDF
question2 = "What is the recipe for chocolate cake?"

print(f"Question 2: {question2}\n")

response2 = rag_chain.invoke({
    "input": question2
})

print(f"Answer: {response2['answer']}")

print(
    "\nThe AI should say "
    "'I don't know based on the provided document.' "
    "because this isn't in the PDF!"
)


print("\nRAG Chain working with hallucination protection!")

print(
    "Next: Module 6 — Building the Streamlit UI (Final Product!)"
)

