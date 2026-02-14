# 🧠 Local RAG (Retrieval-Augmented Generation) System — Explained

This Python script allows you to create a **local chatbot** that can read your PDFs and DOCX files, store them in a searchable vector database, and answer questions using a local **LLM** (like Phi-3-mini or Mistral).  
It works **offline** after the models are downloaded once.

---

## 🧩 Overview

The script performs the following steps:

1. **Load** local documents (`.pdf`, `.docx`)
2. **Split** documents into smaller chunks
3. **Embed** chunks using sentence-transformers
4. **Store** embeddings in a FAISS vector database
5. **Retrieve** relevant text when a question is asked
6. **Generate** an answer using a local LLM

---

## 🔧 Imports

```python
import os
import logging
import torch
from transformers import pipeline

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
```

### Purpose of each:
- **os, logging** → File management and progress logs  
- **torch** → Detects GPU/CPU  
- **transformers** → Loads LLM  
- **LangChain modules** → Handle document loading, text splitting, embeddings, FAISS, and LLM chaining

---

## 🧱 Step 1: Load Documents

```python
def load_documents(folder_path):
    ...
```

- Checks if the `folder_path` exists and is not empty.  
- Loads all PDFs and DOCX files using:
  - `PyPDFLoader` for `.pdf`
  - `Docx2txtLoader` for `.docx`
- Returns a list of `Document` objects.

**Example:**  
If your folder has:
```
data/
 ├── file1.pdf
 └── file2.docx
```
then both are read and converted into text documents.

---

## ✂️ Step 2: Split Documents

```python
def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    return splitter.split_documents(docs)
```

- Breaks long documents into smaller chunks (≈1000 chars).  
- Each chunk overlaps by 150 chars for better context continuity.  
- Smaller chunks = more accurate retrieval.

---

## 🧠 Step 3: Create Embeddings + FAISS

```python
def create_vector_store(chunks):
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore
```

- Converts each chunk into numerical vectors (embeddings).  
- Uses **MiniLM** model — fast and accurate.  
- Stores everything in a **FAISS** database for quick similarity search.

---

## ⚙️ Step 4: Load Model (LLM)

```python
def load_local_model():
    if torch.cuda.is_available():
        ...
    else:
        ...
```

- Detects GPU availability automatically.  
- Loads:
  - **Mistral 7B** (powerful, GPU required)
  - **Phi-3-mini** (lightweight, CPU friendly)
- Uses `transformers.pipeline("text-generation")`.  
- Wraps it into LangChain’s `HuggingFacePipeline`.

So the script adapts automatically to your system.

---

## ❓ Step 5: Ask Question (RAG Flow)

```python
def ask_question(vectorstore, llm, query):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
```

- Retrieves **top 3 most relevant** chunks based on your question.

Then defines a **prompt template**:

```python
prompt = ChatPromptTemplate.from_template(
    "Answer the following question based on the provided context:\n\n{context}\n\nQuestion: {question}"
)
```

Creates the LangChain pipeline:

```python
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
)
```

**Pipeline logic:**
1. Retrieve context from FAISS  
2. Combine context + question into the prompt  
3. Pass it to the LLM  
4. Return the generated answer

---

## 🧑‍💻 Main Execution Block

```python
if __name__ == "__main__":
    folder = "data"
    ...
    while True:
        query = input("\n> ")
        if query.lower() in ["exit", "quit"]:
            break
        answer = ask_question(vectorstore, llm, query)
        print("\n🧠 Answer:\n", answer)
```

- Loads and indexes all documents from `data/`.  
- Waits for user input in a chat loop.  
- Type “exit” or “quit” to stop.  
- Each question triggers retrieval + generation.

---

## ⚡ Summary Table

| Step | Function | Purpose |
|------|-----------|----------|
| 1️⃣ | `load_documents()` | Reads your PDFs/DOCX files |
| 2️⃣ | `chunk_documents()` | Splits them into small parts |
| 3️⃣ | `create_vector_store()` | Embeds and stores chunks in FAISS |
| 4️⃣ | `load_local_model()` | Loads a local LLM (GPU/CPU aware) |
| 5️⃣ | `ask_question()` | Retrieves relevant text and answers |
| 🔁 | `main loop` | Lets you chat with your documents |

---

## 💾 Optional Enhancement

You can add **persistent FAISS storage** so it doesn’t rebuild every time:

```python
vectorstore.save_local("faiss_index")
# Later:
vectorstore = FAISS.load_local("faiss_index", embeddings)
```

This will save your embeddings on disk and speed up future runs.

---

## ✅ Example Usage

1. Put your documents inside the `data/` folder.  
2. Run:
   ```bash
   python rag_local.py
   ```
3. Type:
   ```
   > What is the refund policy in this document?
   ```
4. Get instant answers powered by **local AI**.

---

### 🧠 Key Benefits

- 100% offline after first model download  
- Works with both GPU and CPU  
- Handles both PDF and DOCX  
- Uses open-source models (no API key needed)

---
