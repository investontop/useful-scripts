# model = 'sentence-transformers/all-MiniLM-L6-v2'

import os
import logging
import torch
from transformers import pipeline

# --- LangChain imports (modern LCEL version) ---
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# --- Logging setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Step 1: Load documents ---
def load_documents(folder_path):
    logger.info("Loading documents from: %s", folder_path)
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"The folder '{folder_path}' does not exist.")
    if not os.listdir(folder_path):
        raise FileNotFoundError(f"The folder '{folder_path}' is empty.")

    docs = []
    for file in os.listdir(folder_path):
        full_path = os.path.join(folder_path, file)
        logger.info(f"Loading {full_path}...")
        if file.endswith(".pdf"):
            loader = PyPDFLoader(full_path)
        elif file.endswith(".docx"):
            loader = Docx2txtLoader(full_path)
        else:
            continue
        docs.extend(loader.load())
    return docs

# --- Step 2: Split documents ---
def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    return splitter.split_documents(docs)

# --- Step 3: Create embeddings & FAISS vector store ---
def create_vector_store(chunks):
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

# --- Step 4: Load Model (auto GPU/CPU + fallback) ---
def load_local_model():
    if torch.cuda.is_available():
        logger.info("GPU detected — loading Mistral 7B...")
        model_name = "mistralai/Mistral-7B-Instruct-v0.2"
        dtype = torch.bfloat16
        device = "cuda"
    else:
        logger.info("No GPU detected — loading Phi-3-mini (CPU friendly)...")
        model_name = "microsoft/Phi-3-mini-4k-instruct"
        dtype = torch.float32
        device = "cpu"

    pipe = pipeline(
        "text-generation",
        model=model_name,
        device_map="auto" if device == "cuda" else None,
        torch_dtype=dtype,
        max_new_tokens=512,
        temperature=0.3,
    )
    return HuggingFacePipeline(pipeline=pipe)

# --- Step 5: Ask question (retrieval + generation) ---
def ask_question(vectorstore, llm, query):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    # retrieved_docs = retriever.get_relevant_documents(query)
    # print(f"Retrieved {len(retrieved_docs)} documents")

    prompt = ChatPromptTemplate.from_template(
        "Answer the following question based on the provided context:\n\n{context}\n\nQuestion: {question}"
    )

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
    )

    response = chain.invoke(query)

    # Handle both dict and string outputs
    if isinstance(response, dict):
        text = response.get("text") or response.get("generated_text") or str(response)
    else:
        text = str(response)

    return text.strip()

# --- Main ---
if __name__ == "__main__":
    folder = "data"
    logger.info("Loading and indexing documents...")
    docs = load_documents(folder)
    chunks = chunk_documents(docs)
    vectorstore = create_vector_store(chunks)
    llm = load_local_model()

    logger.info("✅ Ready! Ask your question (type 'exit' to quit):")
    while True:
        query = input("\n> ")
        if query.lower() in ["exit", "quit"]:
            break
        print(query)
        answer = ask_question(vectorstore, llm, query)
        print("\n🧠 Answer:\n", answer)
