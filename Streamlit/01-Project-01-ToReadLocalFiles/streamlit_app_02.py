import streamlit as st
from ingest import ingest_folder
from index_builder import build_index
from search import search_index

# Optional: local LLM generation
from local_gen import generate_answer

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
        build_index(chunks, index_path=INDEX_FILE, meta_path=META_FILE)
    st.sidebar.success("✅ Index successfully rebuilt!")

# --- Chat interface ---
st.write("### 💬 Ask a question based on your local documents:")
query = st.text_input("Your Question")

# Checkbox appears before searching
generate_flag = st.checkbox("📝 Generate concise answer")

if st.button("Search"):
    if not query.strip():
        st.warning("⚠️ Please enter a question first.")
    else:
        with st.spinner("🔍 Searching your local index..."):
            results = search_index(query, top_k=3, index_path=INDEX_FILE, meta_path=META_FILE)

        if results:
            st.success("✅ Retrieved relevant passages:")
            for i, res in enumerate(results, start=1):
                st.markdown(f"**{i}.** {res}")

            if generate_flag:
                with st.spinner("🤖 Generating concise answer..."):
                    context = "\n\n".join(results)
                    try:
                        answer = generate_answer(context, query)
                        st.subheader("💡 Concise Answer")
                        st.write(answer)
                    except Exception as e:
                        st.error(f"LLM generation failed: {e}")

        else:
            st.warning("No relevant results found.")


# Footer
st.markdown("---")
st.caption("📁 This chatbot works fully offline — searches only your local documents.")
