from langchain.prompts import PromptTemplate
import os
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA, LLMChain
from langchain.embeddings import HuggingFaceBgeEmbeddings
from io import BytesIO
from langchain.document_loaders import PyPDFLoader
import gradio as gr
from langchain.llms import VLLM

model_name = "lmsys/vicuna-7b-v1.3"

llm = VLLM(
    model=model_name,
    trust_remote_code=True,  # mandatory for hf models
    max_new_tokens=1024,
    top_k=50,
    top_p=0.9,
    temperature=0.1,
)

print("vLLM Initialized...")


prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

model_name = "BAAI/bge-large-en"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)


prompt = PromptTemplate(template = prompt_template, input_variables = ['context', 'question'])
load_vector_store = Chroma(persist_directory="stores/pet_cosine", embedding_function=embeddings)
retriever = load_vector_store.as_retriever(search_kwargs={"k":1})

#Check whether retriever working fine i.e for query it should find relevent chunk from document.
query = "what is the fastest speed for a greyhound dog?"
semantic_search = retriever.get_relevant_documents(query)
print(semantic_search)


print('#'*100)


chain_type_kwargs = {"prompt" : prompt}

qa = RetrievalQA.from_chain_type(
          llm = llm, 
          chain_type = "stuff", 
          retriever = retriever, 
          return_source_documents = True, 
          chain_type_kwargs = chain_type_kwargs, 
          verbose = True
          )

response = qa(query)
print(response)



