# streamlit_app.py + Rebuilding index button

import streamlit as st
from ingest import ingest_folder
from index_builder import build_index
from search import search_index

# Configuration
DOCS_FOLDER = "./localFiles"
INDEX_FILE = "faiss_index.bin"
META_FILE = "meta.pkl"

# Streamlit Page Setup
st.set_page_config(page_title="Private Chatbot", layout="wide")
st.title("🔒 Private Document Chatbot")

# Sidebar
st.sidebar.header("🧠 Knowledge Base Options")
st.sidebar.write("Manage and update your document knowledge base here.")

# --- Button to rebuild index ---
if st.sidebar.button("🔁 Rebuild Index"):
    with st.spinner("📄 Indexing your documents... please wait ⏳"):
        chunks = ingest_folder(DOCS_FOLDER)
        build_index(chunks)
    st.sidebar.success("✅ Index successfully rebuilt!")

# --- Chat interface ---
st.write("### 💬 Ask a question based on your local documents:")
query = st.text_input("Your Question")

if st.button("Search"):
    if not query.strip():
        st.warning("⚠️ Please enter a question first.")
    else:
        with st.spinner("🔍 Searching your local index..."):
            results = search_index(query, top_k=3)

        if results:
            st.success("✅ Found relevant content:")
            for i, res in enumerate(results, start=1):
                st.markdown(f"**{i}.** {res}")
        else:
            st.warning("No relevant results found.")

# Footer
st.markdown("---")
st.caption("📁 This chatbot works fully offline — searches only your local documents.")
