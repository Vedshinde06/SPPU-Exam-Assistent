# backend.py
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tempfile, os

from app import (
    load_documents, split_documents, create_vectorstore, format_retrieved_docs,
    format_chat_history, build_rag_chain, build_mcq_chain
)

app = FastAPI(title="SPPU Exam Assistant Backend")

# allow frontend CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage
vectorstore = None
chat_history = []

# ---------------- Request models ----------------
class QueryRequest(BaseModel):
    question: str

class MCQRequest(BaseModel):
    context: str
    num_questions: int

# ---------------- PDF Upload ----------------
@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    global vectorstore
    # Save temp PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        # Load, split, and vectorize
        docs = load_documents(tmp_path)
        splits = split_documents(docs)
        vectorstore = create_vectorstore(splits)

        return {"message": f"{file.filename} processed and stored.", "chunks": len(splits)}
    finally:
        os.remove(tmp_path)

# ---------------- Chat RAG ----------------
@app.post("/ask/")
async def ask_question(req: QueryRequest):
    global chat_history, vectorstore
    if not vectorstore:
        return {"answer": "Please upload a PDF first.", "sources": []}

    # Retrieve top-k relevant docs
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    docs = retriever.get_relevant_documents(req.question)
    context, sources = format_retrieved_docs(docs)

    # Run RAG chain
    rag_chain = build_rag_chain()
    try:
        answer = rag_chain.invoke({
            "chat_history": format_chat_history(chat_history),
            "context": context,
            "question": req.question
        })
    except Exception as e:
        answer = f"Error generating answer: {e}"

    # Update in-memory chat history
    chat_history.append({"role": "user", "content": req.question})
    chat_history.append({"role": "assistant", "content": answer})

    return {"answer": answer, "sources": sources}

# ---------------- MCQ Generation ----------------
@app.post("/generate_mcqs/")
async def generate_mcqs(req: MCQRequest):
    mcq_chain = build_mcq_chain()
    try:
        mcqs = mcq_chain.invoke({
            "context": req.context,
            "num_questions": req.num_questions
        })
    except Exception as e:
        mcqs = f"Error generating MCQs: {e}"

    return {"mcqs": mcqs}
