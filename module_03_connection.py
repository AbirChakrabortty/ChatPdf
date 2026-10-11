from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter

import warnings
warnings.filterwarnings("ignore")

pdf_path = "TransformerAttenctionMechanism.pdf"

loader = PyPDFLoader(pdf_path)

pages = loader.load()

# print(f"📊 Total pages loaded: {len(pages)}")
# print(f"📊 Type of each page: {type(pages[0])}")
# print(f"\n📖 Preview of Page 1 (first 500 characters):")
# print("-" * 50)
# print(pages[0].page_content[:500])
# print("-" * 50)
total_characters = sum(len(page.page_content) for page in pages)
estimated_tokens = total_characters // 4  # rough estimate: 1 token ≈ 4 chars



print(f"\nTotal character present in pdf: {total_characters:,}")
print(f"So estimated tokens are: {estimated_tokens:,}")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,         # how to measure chunk size
    is_separator_regex=False     # treat separators as plain strings
)



chunks = text_splitter.split_documents(pages)

print(f"✂️  Original pages: {len(pages)}")
print(f"✂️  After splitting: {len(chunks)} chunks")




