"""
============================================================
                    CHAT PDF
============================================================

PDF Q&A Chatbot using:

    Streamlit
    LangChain
    Groq
    HuggingFace Embeddings
    ChromaDB

Features:
    - Upload PDF
    - Automatically process PDF
    - Chat with PDF
    - Chat history
    - Source chunks
    - Hallucination protection
    - New PDF completely replaces old PDF
============================================================
"""


# ============================================================
# STEP 1: IMPORTS
# ============================================================

import streamlit as st
from dotenv import load_dotenv

import tempfile
import os
import warnings
import hashlib


# LLM
from langchain_groq import ChatGroq


# Prompt
from langchain_core.prompts import ChatPromptTemplate


# PDF Loading
from langchain_community.document_loaders import PyPDFLoader


# Text Splitting
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Embeddings
from langchain_huggingface import HuggingFaceEmbeddings


# Vector Store
from langchain_chroma import Chroma


# RAG Chain
from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain
)

from langchain_classic.chains.retrieval import (
    create_retrieval_chain
)


# ============================================================
# STEP 2: WARNING SETTINGS
# ============================================================

warnings.filterwarnings(
    "ignore",
    category=FutureWarning
)

warnings.filterwarnings(
    "ignore",
    category=UserWarning
)


# ============================================================
# STEP 3: LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# STEP 4: STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="ChatPDF",
    page_icon="📄",
    layout="wide"
)


# ============================================================
# STEP 5: SESSION STATE
# ============================================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


if "vector_store" not in st.session_state:
    st.session_state.vector_store = None


if "pdf_id" not in st.session_state:
    st.session_state.pdf_id = None


if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None


if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False


# ============================================================
# STEP 6: LOAD LLM
# ============================================================

@st.cache_resource
def get_llm():

    return ChatGroq(
        model="qwen/qwen3.8-27b",
        temperature=0
    )


# ============================================================
# STEP 7: LOAD EMBEDDING MODEL
# ============================================================

@st.cache_resource
def get_embeddings():

    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )


llm = get_llm()

embeddings = get_embeddings()


# ============================================================
# STEP 8: RAG PROMPT
# ============================================================

rag_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a helpful assistant that answers questions based ONLY
on the provided context from the CURRENT PDF.

RULES:

- Answer using ONLY the information in the provided context.
- Do not use outside knowledge.
- Do not use information from any previous PDF.
- Do not use information from previous documents.
- Do not make up information.
- Treat the provided context as the ONLY source of truth.
- Be concise and direct.
- Use bullet points when listing multiple items.

IMPORTANT:

If the answer is NOT present in the provided context,
say exactly:

"I don't know based on the provided document."

Do NOT assume or invent an answer.

CONTEXT:

{context}
"""
        ),
        (
            "human",
            "{input}"
        )
    ]
)


# ============================================================
# STEP 9: CREATE UNIQUE PDF ID
# ============================================================

def get_pdf_id(uploaded_file):

    """
    Creates a unique ID based on the actual PDF content.

    This is better than using id(uploaded_file),
    because Streamlit can recreate the UploadedFile
    object during reruns.
    """

    file_bytes = uploaded_file.getvalue()

    file_hash = hashlib.md5(
        file_bytes
    ).hexdigest()

    return file_hash


# ============================================================
# STEP 10: PROCESS PDF
# ============================================================

def process_pdf(uploaded_file):

    """
    Process the uploaded PDF.

    PDF
      ↓
    Load
      ↓
    Extract text
      ↓
    Split into chunks
      ↓
    Create embeddings
      ↓
    Create NEW in-memory ChromaDB
    """


    # --------------------------------------------------------
    # Save uploaded PDF temporarily
    # --------------------------------------------------------

    with st.spinner("📖 Reading PDF..."):

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as tmp:

            tmp.write(
                uploaded_file.getvalue()
            )

            tmp_path = tmp.name


    try:

        # ====================================================
        # LOAD PDF
        # ====================================================

        with st.spinner(
            "📖 Extracting text from PDF..."
        ):

            loader = PyPDFLoader(
                tmp_path
            )

            pages = loader.load()


        # ====================================================
        # CHECK PAGES
        # ====================================================

        if not pages:

            st.error(
                "❌ Could not read any pages from this PDF."
            )

            return None, 0, 0


        # ====================================================
        # CHECK EXTRACTED TEXT
        # ====================================================

        total_text = "".join(

            page.page_content.strip()

            for page in pages

        )


        if not total_text.strip():

            st.error(
                """
❌ No readable text was found in this PDF.

This may be a scanned/image-only PDF.

Please upload a PDF containing selectable text.
"""
            )

            return None, len(pages), 0


        # ====================================================
        # SPLIT INTO CHUNKS
        # ====================================================

        with st.spinner(
            "✂️ Splitting document into chunks..."
        ):

            text_splitter = RecursiveCharacterTextSplitter(

                chunk_size=1000,

                chunk_overlap=200

            )

            chunks = text_splitter.split_documents(
                pages
            )


        # ====================================================
        # REMOVE EMPTY CHUNKS
        # ====================================================

        chunks = [

            chunk

            for chunk in chunks

            if chunk.page_content.strip()

        ]


        # ====================================================
        # CHECK CHUNKS
        # ====================================================

        if not chunks:

            st.error(
                "❌ No usable text chunks were created."
            )

            return None, len(pages), 0


        # ====================================================
        # CREATE NEW IN-MEMORY CHROMA
        # ====================================================

        with st.spinner(

            f"🧠 Creating embeddings for "
            f"{len(chunks)} chunks..."

        ):

            # IMPORTANT:
            #
            # No persist_directory
            # No fixed collection
            #
            # This creates a fresh in-memory vector store
            # for the current PDF only.

            vector_store = Chroma.from_documents(

                documents=chunks,

                embedding=embeddings

            )


        # ====================================================
        # RETURN
        # ====================================================

        return (

            vector_store,

            len(pages),

            len(chunks)

        )


    finally:

        # ====================================================
        # DELETE TEMPORARY PDF
        # ====================================================

        try:

            os.unlink(
                tmp_path
            )

        except Exception:

            pass


# ============================================================
# STEP 11: SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📤 Upload PDF")


    uploaded_file = st.file_uploader(

        "Drag and drop your PDF here",

        type=["pdf"],

        help=(
            "Upload a PDF file to start "
            "asking questions about it."
        )

    )


    # ========================================================
    # CHECK CURRENT PDF
    # ========================================================

    if uploaded_file:

        current_pdf_id = get_pdf_id(
            uploaded_file
        )


        # ----------------------------------------------------
        # Detect a NEW PDF
        # ----------------------------------------------------

        if current_pdf_id != st.session_state.pdf_id:

            # ------------------------------------------------
            # COMPLETELY REMOVE THE OLD PDF
            # ------------------------------------------------

            # Remove the old vector store reference
            old_vector_store = st.session_state.vector_store
            st.session_state.vector_store = None

            # Clear old conversation
            st.session_state.chat_history = []

            # Clear old PDF information
            st.session_state.pdf_id = None
            st.session_state.pdf_name = None
            st.session_state.pdf_processed = False

            # ------------------------------------------------
            # CREATE A COMPLETELY NEW VECTOR STORE
            # FOR THE NEW PDF
            # ------------------------------------------------

            vector_store, num_pages, num_chunks = process_pdf(
                uploaded_file
            )

            # ------------------------------------------------
            # PROCESSING SUCCEEDED
            # ------------------------------------------------

            if vector_store is not None:

                # Store ONLY the new PDF's vector store
                st.session_state.vector_store = vector_store

                # Store the new PDF identity
                st.session_state.pdf_id = current_pdf_id
                st.session_state.pdf_name = uploaded_file.name
                st.session_state.pdf_processed = True

                st.success(
                    f"✅ Processed successfully!\n\n"
                    f"📄 {uploaded_file.name}\n\n"
                    f"📑 Pages: {num_pages}\n\n"
                    f"🧩 Chunks: {num_chunks}"
                )


    # ========================================================
    # CURRENT PDF STATUS
    # ========================================================

    if (

        st.session_state.pdf_processed

        and st.session_state.vector_store is not None

    ):

        st.info(
            f"📋 Current PDF:\n\n"
            f"**{st.session_state.pdf_name}**"
        )


        # ----------------------------------------------------
        # NEW PDF BUTTON
        # ----------------------------------------------------

        if st.button(
            "🔄 Upload New PDF"
        ):

            # Completely remove current vector store
            st.session_state.vector_store = None

            # Clear chat
            st.session_state.chat_history = []

            # Forget current PDF
            st.session_state.pdf_id = None

            st.session_state.pdf_name = None

            st.session_state.pdf_processed = False


            st.rerun()


    # ========================================================
    # ABOUT
    # ========================================================

    st.divider()


    st.markdown(
        """
### ℹ️ How ChatPDF works

**1️⃣ Upload**

Upload your PDF.

**2️⃣ Process**

The PDF is split into chunks and converted
into embeddings.

**3️⃣ Retrieve**

When you ask a question, ChatPDF searches
for relevant chunks.

**4️⃣ Generate**

The relevant chunks are sent to the Groq LLM.

**5️⃣ Answer**

The LLM answers using information from
your PDF.

---

### 🔥 Technology

- Streamlit
- LangChain
- Groq
- HuggingFace
- ChromaDB
- RAG
"""
    )


# ============================================================
# STEP 12: MAIN HEADER
# ============================================================

st.title("📄 ChatPDF")

st.caption(
    "Upload a PDF and ask questions about it. "
    "Powered by Groq + LangChain + ChromaDB."
)


# ============================================================
# STEP 13: DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.chat_history:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ============================================================
# STEP 14: CHAT INPUT
# ============================================================

user_question = st.chat_input(

    "Ask a question about your PDF...",

    disabled=(
        not st.session_state.pdf_processed
        or st.session_state.vector_store is None
    )

)


# ============================================================
# STEP 15: PROCESS QUESTION
# ============================================================

if user_question:

    # ========================================================
    # DISPLAY USER MESSAGE
    # ========================================================

    with st.chat_message("user"):

        st.markdown(
            user_question
        )


    # ========================================================
    # SAVE USER MESSAGE
    # ========================================================

    st.session_state.chat_history.append(

        {
            "role": "user",

            "content": user_question
        }

    )


    # ========================================================
    # GENERATE ANSWER
    # ========================================================

    with st.chat_message("assistant"):

        with st.spinner(
            "🤔 Searching your PDF..."
        ):

            # ------------------------------------------------
            # Document chain
            # ------------------------------------------------

            document_chain = (

                create_stuff_documents_chain(

                    llm,

                    rag_prompt

                )

            )


            # ------------------------------------------------
            # Retriever
            # ------------------------------------------------

            retriever = (

                st.session_state.vector_store

                .as_retriever(

                    search_type="similarity",

                    search_kwargs={
                        "k": 4
                    }

                )

            )


            # ------------------------------------------------
            # RAG chain
            # ------------------------------------------------

            rag_chain = (

                create_retrieval_chain(

                    retriever,

                    document_chain

                )

            )


            # ------------------------------------------------
            # Ask question
            # ------------------------------------------------

            response = rag_chain.invoke(

                {
                    "input": user_question
                }

            )


            # ------------------------------------------------
            # Get answer
            # ------------------------------------------------

            answer = response["answer"]


        # ====================================================
        # DISPLAY ANSWER
        # ====================================================

        st.markdown(
            answer
        )


        # ====================================================
        # SOURCE CHUNKS
        # ====================================================

        with st.expander(
            "📚 View Source Chunks"
        ):

            for i, doc in enumerate(

                response["context"],

                1

            ):

                page_num = doc.metadata.get(
                    "page",
                    "?"
                )


                st.markdown(

                    f"**Chunk {i} — "
                    f"Page {page_num}**"

                )


                st.caption(

                    doc.page_content[:500]

                    + "..."

                )


                st.divider()


    # ========================================================
    # SAVE ASSISTANT MESSAGE
    # ========================================================

    st.session_state.chat_history.append(

        {
            "role": "assistant",

            "content": answer
        }

    )


# ============================================================
# STEP 16: EMPTY STATE
# ============================================================

if not st.session_state.pdf_processed:

    st.info(
        "👈 Upload a PDF from the sidebar to get started!"
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.markdown(
            "### 1️⃣ Upload"
        )

        st.write(
            "Upload your PDF document."
        )


    with col2:

        st.markdown(
            "### 2️⃣ Process"
        )

        st.write(
            "ChatPDF creates embeddings "
            "and stores them in ChromaDB."
        )


    with col3:

        st.markdown(
            "### 3️⃣ Ask"
        )

        st.write(
            "Ask questions and get answers "
            "from your PDF."
        )