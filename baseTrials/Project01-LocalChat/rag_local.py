import os
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
# from langchain_text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

# --- Step 1: Load documents ---
def load_documents(folder_path):
    print("Loading documents from:", folder_path)
    docs = []
    for file in os.listdir(folder_path):
        full_path = os.path.join(folder_path, file)
        print(f"Loading {full_path}...")
        if file.endswith(".pdf"):
            loader = PyPDFLoader(full_path)
        elif file.endswith(".docx"):
            loader = Docx2txtLoader(full_path)
        else:
            continue
        docs.extend(loader.load())
    print(docs)
    return docs

# --- Step 2: Split documents into chunks ---
def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    return splitter.split_documents(docs)

# --- Step 3: Create embeddings & store in FAISS ---
def create_vector_store(chunks):
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

# --- Step 4: Load local Mistral model ---
def load_mistral_model():
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    pipe = pipeline(
        "text-generation",
        model=model_name,
        torch_dtype="auto",
        device_map="auto",
        max_new_tokens=512,
        temperature=0.3,
    )
    return HuggingFacePipeline(pipeline=pipe)

# --- Step 5: Ask questions ---
def ask_question(vectorstore, llm, query):
    # from langchain_chains import RetrievalQA
    from langchain_community.chains import RetrievalQA
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")
    return qa.run(query)

# --- Main ---
if __name__ == "__main__":
    folder = "data"
    print("Loading and indexing documents...")
    docs = load_documents(folder)
    chunks = chunk_documents(docs)
    vectorstore = create_vector_store(chunks)
    llm = load_mistral_model()

    print("Ready! Ask your question:")
    while True:
        query = input("\n> ")
        if query.lower() in ["exit", "quit"]:
            break
        answer = ask_question(vectorstore, llm, query)
        print("\nAnswer:", answer)
