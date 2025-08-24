import streamlit as st
import base64
import os
from pathlib import Path
import tempfile
from faiss_rag import split_documents, add_to_faiss, query_rag
from langchain.document_loaders import PyPDFLoader

if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None

def display_pdf(uploaded_file):
    """Display PDF in Streamlit app in a scrollable iframe"""
    bytes_data = uploaded_file.getvalue()
    base64_pdf = base64.b64encode(bytes_data).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="900" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

def load_streamlit_page():
    """Load the streamlit page layout and file uploader"""
    st.set_page_config(layout="wide", page_title="RAG LLM Tool")

    col1, col2 = st.columns([0.5, 0.5], gap="large")

    with col1:
        st.title("RAG LLM Tool")
        st.header("Upload file")
        uploaded_file = st.file_uploader("Please upload your document:", type="pdf")

    return col1, col2, uploaded_file

def process_uploaded_file(uploaded_file):
    """Process uploaded PDF using existing backend functions"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    
    try:
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        for i, doc in enumerate(documents):
            doc.metadata['source'] = uploaded_file.name
            doc.metadata['page'] = i + 1
        
        chunks = split_documents(documents)
        db = add_to_faiss(chunks)
        return db
    finally:
        os.unlink(tmp_path)

col1, col2, uploaded_file = load_streamlit_page()

if uploaded_file is not None:
    with col2:
        display_pdf(uploaded_file)
        
    st.session_state.vector_store = process_uploaded_file(uploaded_file)

with col1:
    with st.form("question_form"):
        query = st.text_input("Enter your question:")
        submitted = st.form_submit_button("Ask Question")
        
        if submitted:
            if not uploaded_file:
                st.warning("Please upload a PDF first")
            elif not st.session_state.vector_store:
                st.warning("Please process the document first")
            elif query:
                with st.spinner("Searching documents..."):
                    response = query_rag(query)
                    st.markdown("### Response")
                    st.write(response)