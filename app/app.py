from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace, HuggingFacePipeline
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_cerebras import ChatCerebras
import os
from dotenv import load_dotenv

load_dotenv()

def load_documents(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load()

def split_documents(docs, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap
    )
    return splitter.split_documents(docs)

def create_vectorstore(splits):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore

def format_retrieved_docs(docs, char_limit=800):
    """
    Build a single 'context' string for the prompt and also return structured sources for UI.
    Tries to extract page number from metadata keys commonly used. Falls back to 'unknown'.
    """
    ctx_parts = []
    sources = []
    for i, d in enumerate(docs):
        md = d.metadata or {}
        source_name = md.get("source") or md.get("file_name") or "unknown"
        page = md.get("page") or md.get("page_number") or md.get("pagenum") or md.get("page_no") or "unknown"
        content = (d.page_content or "").strip()
        snippet = content[:char_limit].strip()
        ctx_parts.append(f"Source: {source_name} | Page: {page}\n{snippet}\n---\n")
        sources.append({"source": source_name, "page": page, "snippet": snippet, "full_text": content})
    context_text = "\n".join(ctx_parts)
    return context_text, sources

def format_chat_history(history):
    """
    Convert session chat history (list of {'role','content'}) into a plain text string
    that we can pass to the prompt template.
    """
    lines = []
    for m in history:
        role = m.get("role", "user")
        content = m.get("content", "")
        
        if role == "user":
            lines.append(f"user: {content}")
        else:
            lines.append(f"Assistent: {content}")
            
    return "\n".join(lines)


def build_rag_chain():
    #retriever = vectorstore.as_retriever(search_kwargs={"k":5})
    
    llm = HuggingFaceEndpoint(repo_id="openai/gpt-oss-20b", task="text-generation")
    #model = ChatHuggingFace(llm=llm)
    model = ChatCerebras(
    model="llama-4-scout-17b-16e-instruct",
    api_key=os.getenv("CEREBRAS_API_KEY")
    )
    parser = StrOutputParser()
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """You are an AI exam assistant for SPPU students. You will be provided with a context extracted from the uploaded PDF and a question. 

                1. If the user asks an exam-related question:
                    - Answer only using the information from the context.
                    - Provide the answer in a point-wise format (bullets or numbered points).
                    - Each point should be a full sentence.
                    - If the answer is not in the context, respond clearly with: "I don't know based on the provided material."
                    - Keep answers concise, clear, and relevant.
                2. If the user sends a casual or conversational message (e.g., greetings, "Hi", "Ok", "How are you?"):
                    - Respond naturally and conversationally.
                    - Do not force point-wise answers or reference the context unnecessarily.
             
                Context:
                {context}

                Question:
                {question}

                Answer:
                 """),
                 
            ("human", "Conversion so far: \n{chat_history}\n\nContext:\n{context}\n\nQuestion:\n{question}\n\nAnswer:")
        ]
    )
    
    rag_chain = (
        
        {
            "chat_history": lambda x: x["chat_history"],
            "context": lambda x: x["context"],
            "question": lambda x: x["question"]
        }
        | prompt | model | parser
    )
    
    return rag_chain

def build_mcq_chain():
    
    llm = HuggingFaceEndpoint(repo_id="openai/gpt-oss-20b", task="text-generation")
    model = ChatCerebras(
    model="llama-4-scout-17b-16e-instruct",
    api_key=os.getenv("CEREBRAS_API_KEY")
    )
    #model = ChatHuggingFace(llm=llm)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an AI exam assistent/teacher. Generate multiple choice questions on following text."),
        ("human", "Text:\n{context}\n\nGenerate {num_questions} MCQs with 4 options each, and mark the correct answer.")
    ])
    
    parser = StrOutputParser()

    mcq_chain = prompt | model | parser
    
    return mcq_chain