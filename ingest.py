import os
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.document_loaders import PyPDFLoader

# Check if GPU is available and set the device accordingly
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model_name = "BAAI/bge-large-en"
model_kwargs = {'device': device}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

print("Embedding Model Loaded....")

loader = PyPDFLoader("pet.pdf")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_documents(documents)

vector_store = Chroma.from_documents(
              texts, 
              embeddings, 
              collection_metadata={"hnsw:space": "cosine"}, 
              persist_directory="stores/pet_cosine"
              )

print("Vector Store Created.......")