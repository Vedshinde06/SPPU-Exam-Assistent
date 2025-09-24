import streamlit as st
from app import load_documents, split_documents, create_vectorstore, build_rag_chain
import os

st.title("SPPU Exam Assistent")

temp_dir = "temp_pdfs"
os.makedirs(temp_dir, exist_ok=True)

uploaded_file = st.file_uploader("Upload Your Textbook PDF", type="pdf")

if uploaded_file is not None:
    file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    docs = load_documents(file_path)
    splits = split_documents(docs)
    vectorstore = create_vectorstore(splits)
    rag_chain = build_rag_chain(vectorstore)
    
    query = st.text_input("Ask the question")
    if query:
        response = rag_chain.invoke({"question": query})
        st.subheader("Answer:")
        st.write(response)