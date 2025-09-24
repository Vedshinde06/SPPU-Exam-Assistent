from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import streamlit as st
import os

load_dotenv()

def load_documents(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    return docs

def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 250
    )

    return splitter.split_documents(docs)

def create_vectorstore(splits):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore

def build_rag_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k":5})
    
    llm = HuggingFaceEndpoint(repo_id="openai/gpt-oss-20b", task="text-generation")
    
    model = ChatHuggingFace(llm=llm)
    
    parser = StrOutputParser()
    
    prompt = PromptTemplate(
        template = """ You are an exam assistant for SPPU students."
                    "Use the following context to answer the question."
                    "If you don't know, say 'I don't know'."

                    "Context:"
                    "{context}"

                    "Question:"
                    "{question}"

                    "Answer:""",
        input_variables = ['context', 'question']
    )
    
    rag_chain = (
        
        {
            "context": lambda x: retriever.invoke(x["question"]),
            "question": lambda x: x["question"]
        }
        | prompt | model | parser
    )
    
    return rag_chain


    