from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
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
    model = ChatHuggingFace(llm=llm)
    parser = StrOutputParser()
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an exam assistant for SPPU students. Use the context provided to answer accurately. If the answer is not in the context, say \"I don't know.\""),
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


    