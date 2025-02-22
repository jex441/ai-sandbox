from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings

file_path = "../../nke-10k-2023.pdf"

# Load the PDF file
loader = PyPDFLoader(file_path)
docs = loader.load()

# Split the documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Create an in-memory vector store
vector_store = InMemoryVectorStore(embeddings)

ids = vector_store.add_documents(documents=all_splits)

# Perform a similarity search
results = vector_store.similarity_search_with_score(
    "What was Nike's revenue in 2023?"
)

print(results[0])