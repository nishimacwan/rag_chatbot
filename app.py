# -*- coding: utf-8 -*-
import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

st.title("Document Chatbot")

# File uploader widget
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
        tmpfile.write(uploaded_file.getvalue())
        file_path = tmpfile.name

    # Check if a chat session exists
    if 'qa_chain' not in st.session_state:
        st.info("Processing document...")

        # Load the PDF document
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        # Split the document into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = text_splitter.split_documents(documents)

        # Create embeddings and a vector store
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # Create a temporary vector store
        vectordb = Chroma.from_documents(
            documents=texts,
            embedding=embeddings
        )

        # Load the Llama model
        llm = LlamaCpp(
            model_path="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
            n_gpu_layers=0,
            n_batch=64,
            n_ctx=4096,
            verbose=False,
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
        )

        # Set up the Retrieval QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectordb.as_retriever()
        )

        st.session_state['qa_chain'] = qa_chain
        st.success("Document processed! You can now ask questions.")

    os.unlink(file_path) # Clean up the temporary file

    # Chat interface
    st.markdown("---")
    query = st.text_input("Ask a question about your document:")

    if query:
        with st.spinner("Generating response..."):
            response = st.session_state['qa_chain'].invoke({"query": query})
            st.write(response['result'])