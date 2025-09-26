import streamlit as st
from app import load_documents, split_documents, create_vectorstore, build_rag_chain, format_retrieved_docs, format_chat_history, build_mcq_chain
import os

st.set_page_config(page_title="SPPU Exam Assistant — Chat RAG", layout="wide")
st.title("📘 SPPU Exam Assistant")

# session state init
if "history" not in st.session_state:
    st.session_state.history = []   # list of {'role': 'user'|'assistant', 'content': ...}

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chunks_loaded" not in st.session_state:
    st.session_state.chunks_loaded = 0

temp_dir = "temp_pdfs"
os.makedirs(temp_dir, exist_ok=True)

uploaded_file = st.file_uploader("Upload one or more PDF Textbooks", type="pdf", accept_multiple_files=False)

if uploaded_file is not None:
    # save file
    file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.info("Loading PDF and building vector store (this may take a moment)...")
    docs = load_documents(file_path)

    # Ensure page metadata exists (PyPDFLoader often includes metadata 'source' and 'page'; we enforce page if missing)
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
    st.success(f"Vector store ready — {len(splits)} chunks indexed from {uploaded_file.name}.")

# show current conversation
chat_col, side_col = st.columns([3, 1])

with chat_col:
    st.subheader("Conversation")
    # render chat messages
    for msg in st.session_state.history:
        if msg["role"] == "user":
            st.chat_message("user").write(msg["content"])
        else:
            st.chat_message("assistant").write(msg["content"])

    # chat input
    user_input = st.chat_input("Ask a question about the uploaded PDF(s)...")
    if user_input:
        # append user message to history and display immediately
        st.session_state.history.append({"role": "user", "content": user_input})
        st.chat_message("user").write(user_input)

        # require vectorstore
        if st.session_state.vectorstore is None:
            answer = "Please upload a PDF first."
            st.session_state.history.append({"role": "assistant", "content": answer})
            st.chat_message("assistant").write(answer)
        else:
            # 1) Retrieve top-k chunks (we call vectorstore.similarity_search directly)
            top_docs = st.session_state.vectorstore.similarity_search(user_input, k=4)

            # 2) Build context string + structured sources
            context_text, sources = format_retrieved_docs(top_docs, char_limit=800)

            # 3) Format chat_history string (for the prompt)
            chat_history_str = format_chat_history(st.session_state.history)

            # 4) Build & invoke LCEL rag chain
            rag_chain = build_rag_chain()
            chain_input = {
                "chat_history": chat_history_str,
                "context": context_text,
                "question": user_input,
            }

            # invoke chain -> returns parsed string (StrOutputParser)
            try:
                result = rag_chain.invoke(chain_input)
                # result should be the assistant answer string
            except Exception as e:
                result = f"Error while generating answer: {e}"

            # 5) append assistant message to history and display
            st.session_state.history.append({"role": "assistant", "content": result})
            st.chat_message("assistant").write(result)

            # 6) show sources in expander (page numbers & snippets)
            with st.expander("📌 Retrieved sources (click to expand)"):
                for i, s in enumerate(sources):
                    st.markdown(f"**Source {i+1}:** `{s['source']}` — **Page:** {s['page']}")
                    st.write(s["snippet"])
                    st.markdown("---")

with side_col:
    st.subheader("Info")
    st.write(f"Chunks indexed: {st.session_state.chunks_loaded}")
    st.write("Tips:")
    st.write("- Ask follow-up questions — history is preserved during this session.")
    st.write("- If the assistant says \"I don't know\", try rephrasing the question or upload more material.")

st.sidebar.title("Exam Assistent")

if uploaded_file:
    st.sidebar.subheader("Uploaded File")
    st.sidebar.write(f"📘{uploaded_file.name}")
    
if "source" in st.session_state:
    st.sidebar.subheader("Sources")
    for src in st.session_state["sources"]:
        st.sidebar.write(f"📄 Page {src.metadata.get('page', '?')} – {src.metadata.get('source', '')}")
    
    
st.subheader("Generate Practice MCQ's")
    
mcq_topic = st.text_input("Enter topic for MCQs (Topic should be from the uploaded PDF):")
num_mcqs = st.number_input("Number of MCQs", min_value=1, max_value=20, value=5, step=1)

if st.button("Generate MCQs"):
    if mcq_topic.strip():
        with st.spinner("Generating questions..."):
            mcq_chain = build_mcq_chain()

            questions = mcq_chain.invoke({
                "context": mcq_topic,   
                "num_questions": num_mcqs
            })

        st.success("✅ MCQs Generated")
        st.write(questions)
    else:
        st.warning("Please enter a topic before generating MCQs.")