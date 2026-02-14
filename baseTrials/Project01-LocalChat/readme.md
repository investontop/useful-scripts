## Requirement

A local chatbot that can:
  - Read your PDFs and Word docs (.docx) from a folder
  - Store their content in a FAISS vector database
  - Use a local Mistral model (via Hugging Face Transformers) to answer questions
  - Run completely offline (no OpenAI API)


## Approach

## Notes
```
Model size: Mistral-7B needs ~12–16 GB VRAM (GPU) or ~16 GB RAM (CPU, slower).
If that’s too heavy, try:
"TheBloke/Mistral-7B-Instruct-v0.2-GGUF" with a GGUF quantized version (for use with llama.cpp)
"microsoft/phi-2" (much lighter but still good)
Vector storage: FAISS is in-memory; you can persist it by calling vectorstore.save_local("faiss_index") and later load it.
```

## rag_local.py

### Folder structure
```
project/
│
├── data/
│   ├── doc1.pdf
│   ├── notes.docx
│
├── rag_local.py
```

What this script does (in short)
1. It creates a private chatbot that:
2. Reads your PDF and Word documents from a folder
3. Splits them into small text pieces
4. Stores those pieces in a searchable database (FAISS)
5. Uses a local AI model (Mistral) to answer your questions based only on your documents, not the internet

Basically, it’s like:
 “ChatGPT, but trained on my own files.”

#### Step 1: Importing the tools
- os → lets Python read files/folders on your computer
- PyPDFLoader / Docx2txtLoader → read text from PDF and Word files
- RecursiveCharacterTextSplitter → breaks large text into smaller chunks
- HuggingFaceEmbeddings → converts text into numeric form (for AI understanding)
- FAISS → stores those numeric forms so the computer can search quickly
- HuggingFacePipeline → loads an AI model (Mistral here)
- transformers.pipeline → connects to the model for text generation

#### Step 2: def load_documents(folder_path):

This goes into a folder (like data/) and:
1. Looks at every file
2. If it ends with .pdf, reads it using PyPDFLoader
3. If it ends with .docx, reads it using Docx2txtLoader
4. Collects all the text from those files

#### Step 3: Split documents into smaller chunks
```python
def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    return splitter.split_documents(docs)
```
```
AI models can’t handle very long text in one go.
So this step breaks the documents into pieces of around 1000 characters each,
and overlaps 150 characters between chunks so context isn’t lost.
```

```lua
Document: "This is a very long file..."
Chunk 1: lines 1–1000
Chunk 2: lines 850–1850  (some overlap)
```

#### Step 4: Create “embeddings” and store them
```python
def create_vector_store(chunks):
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore
```
```
Now we take every text chunk and turn it into numbers (vectors)
using a model that understands language meaning.

Then we store those in a FAISS database,
so we can later ask: “Which chunks are similar to my question?”

Think of it like building a mini “Google Search” for your files.
```

#### Step 5: Load the Mistral AI model
```python
def load_mistral_model():
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    ...
```
```
This loads a local AI model (Mistral), which can:
 - Read questions
 - Generate natural language answers
 - The parameters like temperature=0.3 make it less random and more factual.
```
 #### Step 6: Ask questions
 ```python
 def ask_question(vectorstore, llm, query):
    from langchain_community.chains import RetrievalQA
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")
    return qa.run(query)
 ```
 ```
When you ask a question:
1. The retriever searches FAISS to find the 3 most relevant text chunks (k=3)
2. The Mistral model reads those chunks + your question
3. It generates an answer based only on that context
 ```
 
 #### Step 7: The main part (runs everything)

```python
if __name__ == "__main__":
    folder = "data"
    ...
```
```
This part:
1. Loads and indexes your documents
2. Creates embeddings and the FAISS store
3. Loads the Mistral model
4. Then waits for your questions
```

You can type questions like:
```csharp
> What is the revenue mentioned in the report?
```



## rag_local_01.py

This project builds a **fully local chatbot** that can read your **PDF** and **Word (DOCX)** files, understand their content, and answer your questions — **without any internet access** or cloud services.

### Folder Structure:
```
your_project/
│
├── baseTrials/
│   └── Project01-LocalChat/
│       ├── rag_local_01.py   ← main chatbot script
│       └── data/             ← put your PDFs or DOCXs here
│           ├── example.pdf
│           └── notes.docx
│
└── .venv/                    ← your Python virtual environment

```

### 🚀 Overview

The system uses **LangChain**, **FAISS**, and **Hugging Face models** to:
1. Load your local documents.
2. Split them into chunks for efficient searching.
3. Convert text into vector embeddings.
4. Use a local LLM (like *Mistral 7B*) to answer questions based only on your documents.

---

### 🧩 How It Works

| Step | Description |
|------|--------------|
| **1. Load Documents** | All `.pdf` and `.docx` files from the `data/` folder are read using LangChain loaders. |
| **2. Split Documents** | Each document is split into overlapping chunks (~1000 characters each) for better context search. |
| **3. Create Embeddings** | Sentences are converted into vectors using `sentence-transformers/all-MiniLM-L6-v2`. |
| **4. Build FAISS Index** | The vectorized text is stored locally in a FAISS index (no cloud database). |
| **5. Ask Questions** | When you ask a question, FAISS retrieves the most relevant text, and Mistral LLM generates an answer based on that. |

---

### 🛠️ Setup Instructions

### 1️⃣ Install dependencies
Create and activate a virtual environment:
```bash
python -m venv .venv
.venv\Scripts\activate


## rag_local_02.py

