import streamlit as st
import os
from app import (
    load_documents, split_documents, create_vectorstore, build_rag_chain,
    format_retrieved_docs, format_chat_history, build_mcq_chain
)

st.set_page_config(page_title="SPPU Exam Assistant", layout="wide", page_icon="📘")

# ---------------- Custom CSS for Dark Theme ----------------
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        margin-bottom: 20px;
    }
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2.5em;
    }
    .main-header p {
        color: #f0f0f0;
        font-size: 1.2em;
        margin-top: 10px;
    }
    .fun-fact {
        background-color: rgba(255, 193, 7, 0.15);
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        margin: 20px 0;
        color: #ffd54f;
    }
    .stats-card {
        background-color: rgba(102, 126, 234, 0.1);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid rgba(102, 126, 234, 0.3);
        text-align: center;
    }
    .stats-card h3 {
        color: #667eea;
        margin: 0;
        font-size: 2em;
    }
    .stats-card p {
        color: #b0bec5;
        margin: 5px 0 0 0;
    }
    .source-box {
        background-color: rgba(255, 255, 255, 0.05);
        padding: 15px;
        border-radius: 8px;
        border-left: 3px solid #667eea;
        margin: 10px 0;
    }
    .source-box strong {
        color: #81c784;
    }
    .mcq-container {
        background-color: rgba(255, 255, 255, 0.05);
        padding: 20px;
        border-radius: 10px;
        border: 1px solid rgba(102, 126, 234, 0.2);
        margin: 15px 0;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- Main Header ----------------
st.markdown("""
<div class="main-header">
    <h1>📘 SPPU Exam Assistant</h1>
    <p>
        Your AI-powered study companion to chat with textbooks, generate practice MCQs, and ace your exams!
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------- Fun Fact ----------------
st.markdown("""
<div class="fun-fact">
    <strong>💡 Fun Fact:</strong> If an engineer is studying, that means the exam is tomorrow! 😅
</div>
""", unsafe_allow_html=True)

# ---------------- Session State ----------------
if "history" not in st.session_state:
    st.session_state.history = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chunks_loaded" not in st.session_state:
    st.session_state.chunks_loaded = 0

# ---------------- Sidebar ----------------
with st.sidebar:
    st.markdown("### 🎯 Mode Selection")
    mode = st.radio("", ["💬 Chat", "📝 Generate MCQs"], label_visibility="collapsed")
    
    st.markdown("---")
    st.markdown("### 📁 Upload PDF")
    uploaded_file = st.file_uploader("Upload your textbook PDF", type="pdf", accept_multiple_files=False)
    
    # Show stats if PDF is loaded
    if st.session_state.vectorstore is not None:
        st.markdown("---")
        st.markdown("### 📊 Document Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Chunks", st.session_state.chunks_loaded, delta=None)
        with col2:
            st.metric("Messages", len(st.session_state.history), delta=None)
        
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.history = []
            st.rerun()

temp_dir = "temp_pdfs"
os.makedirs(temp_dir, exist_ok=True)

# ---------------- Handle PDF Upload ----------------
if uploaded_file is not None and st.session_state.vectorstore is None:
    file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.sidebar.success(f"✅ Uploaded: {uploaded_file.name}")

    with st.spinner("🔄 Processing PDF..."):
        # Load documents & create vectorstore
        docs = load_documents(file_path)
        for idx, d in enumerate(docs):
            if d.metadata is None:
                d.metadata = {}
            if "page" not in d.metadata:
                d.metadata["page"] = idx + 1
            if "source" not in d.metadata:
                d.metadata["source"] = uploaded_file.name

        splits = split_documents(docs)
        st.session_state.vectorstore = create_vectorstore(splits)
        st.session_state.chunks_loaded = len(splits)
        os.remove(file_path)

    st.sidebar.info(f"📄 {len(splits)} chunks indexed successfully!")


# ---------------- Main Area ----------------
if mode == "💬 Chat":
    st.markdown("### 💬 Chat with Your PDF")
    
    # Show chat messages
    for msg in st.session_state.history:
        if msg["role"] == "user":
            with st.chat_message("user", avatar="👨‍🎓"):
                st.write(msg["content"])
        else:
            with st.chat_message("assistant", avatar="🤖"):
                st.write(msg["content"])
    
    # Chat input
    user_input = st.chat_input("Ask a question about the uploaded PDF...")
    
    if user_input:
        # Add user message
        st.session_state.history.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="👨‍🎓"):
            st.write(user_input)
        
        if st.session_state.vectorstore is None:
            result = "❌ Please upload a PDF first to start chatting."
            st.session_state.history.append({"role": "assistant", "content": result})
            with st.chat_message("assistant", avatar="🤖"):
                st.write(result)
        else:
            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("🔍 Searching through your textbook..."):
                    # RAG retrieval
                    top_docs = st.session_state.vectorstore.similarity_search(user_input, k=5)
                    context_text, sources = format_retrieved_docs(top_docs, char_limit=800)
                    chat_history_str = format_chat_history(st.session_state.history)
                    
                    rag_chain = build_rag_chain()
                    result = rag_chain.invoke({
                        "chat_history": chat_history_str,
                        "context": context_text,
                        "question": user_input
                    })
                
                st.write(result)
                st.session_state.history.append({"role": "assistant", "content": result})
                
                # Show sources in expander
                with st.expander("📌 View Retrieved Sources"):
                    for i, s in enumerate(sources):
                        st.markdown(f"""
                        <div class="source-box">
                            <strong>Source {i+1}:</strong> {s['source']} — <strong>Page:</strong> {s['page']}<br>
                            <p style="color: #b0bec5; margin-top: 8px;">{s["snippet"]}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
elif mode == "📝 Generate MCQs":
    st.markdown("### 📝 Generate Practice MCQs")
    
    if st.session_state.vectorstore is None:
        st.warning("⚠️ Please upload a PDF first to generate MCQs.")
        st.info("👉 Use the sidebar to upload your textbook PDF")
    else:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            topic_text = st.text_area(
                "Enter topic or keywords (optional)",
                placeholder="E.g., Data Structures, Algorithms, Database Management...",
                help="Leave empty to generate questions from the entire document"
            )
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            num_mcqs = st.number_input(
                "Number of MCQs", 
                min_value=1, 
                max_value=20, 
                value=5, 
                step=1
            )
        
        if st.button("🎲 Generate MCQs", type="primary", use_container_width=True):
            with st.spinner("✨ Generating MCQs from your textbook..."):
                # Use RAG-based MCQ chain
                rag_mcq_chain = build_mcq_chain()
                top_docs = st.session_state.vectorstore.similarity_search(topic_text or "", k=5)
                context_text, sources = format_retrieved_docs(top_docs, char_limit=1000)
                
                mcqs = rag_mcq_chain.invoke({
                    "context": context_text,
                    "num_questions": num_mcqs
                })
            
            st.success("✅ MCQs Generated Successfully!")
            
            # Display MCQs in a nice container
            st.markdown("""
            <div class="mcq-container">
            """, unsafe_allow_html=True)
            st.markdown(mcqs)
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Show sources
            with st.expander("📚 Sources Used for MCQ Generation"):
                for i, s in enumerate(sources):
                    st.markdown(f"""
                    <div class="source-box">
                        <strong>Source {i+1}:</strong> {s['source']} — <strong>Page:</strong> {s['page']}<br>
                        <p style="color: #b0bec5; margin-top: 8px;">{s["snippet"]}</p>
                    </div>
                    """, unsafe_allow_html=True)

# ---------------- Footer ----------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p> Vedant Shinde | Powered by OpenAI's opensource model gpt-oss-20b 🤖</p>
</div>
""", unsafe_allow_html=True)