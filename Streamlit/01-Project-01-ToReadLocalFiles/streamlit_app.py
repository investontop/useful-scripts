import streamlit as st
import tempfile
import os
from ingest import extract_text_from_pdf, extract_text_from_docx, chunk_text
from index_builder import build_index
from search import Retriever

INDEX_PATH = 'faiss_index.bin'
META_PATH = 'meta.pkl'

st.set_page_config(page_title='Private Document Chatbot')
st.title('Private Document Chatbot (Local)')

st.markdown('Upload PDF or DOCX files. All indexing/search stays local.')

uploaded_files = st.file_uploader('Upload PDF or DOCX', accept_multiple_files=True)
if uploaded_files:
    all_chunks = []
    for uf in uploaded_files:
        suffix = uf.name.split('.')[-1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix='.' + suffix) as tmp:
            tmp.write(uf.getbuffer())
            tmp_path = tmp.name
        if suffix in ['pdf']:
            text = extract_text_from_pdf(tmp_path)
        elif suffix in ['docx']:
            text = extract_text_from_docx(tmp_path)
        else:
            st.warning(f"Unsupported file type: {suffix}")
            continue
        chunks = chunk_text(text)
        for c in chunks:
            c['source'] = uf.name
        all_chunks.extend(chunks)

    if all_chunks:
        st.info('Building index — this may take a moment')
        build_index(all_chunks, index_path=INDEX_PATH, meta_path=META_PATH)
        st.success('Index built and saved locally.')

# Load existing index if present
if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
    try:
        retriever = Retriever(INDEX_PATH, META_PATH)
    except Exception as e:
        st.error(f"Failed to load index: {e}")
        retriever = None

    if retriever:
        query = st.text_input('Ask a question about your documents:')
        top_k = st.slider('Top K', min_value=1, max_value=10, value=5)

        if st.button('Search') and query:
            results = retriever.retrieve(query, top_k=top_k)
            st.write('Top passages:')
            for i, r in enumerate(results, 1):
                st.markdown(f"**Result {i} — score {r['score']:.4f} — source: {r['meta'].get('source','unknown')}**")
                st.write(r['text'])

            if st.checkbox('Generate concise answer (local LLM)'):
                try:
                    from local_gen import generate_answer
                    context = "\n\n".join([r['text'] for r in results])
                    with st.spinner('Generating...'):
                        answer = generate_answer(context, query)
                        st.subheader('Generated Answer')
                        st.write(answer)
                except Exception as e:
                    st.error(f"Generation failed: {e}")

else:
    st.info('No index found yet. Upload files above to create an index.')