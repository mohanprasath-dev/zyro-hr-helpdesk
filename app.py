
# ============================================================
#   Zyro Dynamics HR Help Desk — Streamlit Chatbot
#   Deploy: https://share.streamlit.io
# ============================================================

import os
import glob
import streamlit as st
from pathlib import Path

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.schema.output_parser import StrOutputParser
from langchain_groq import ChatGroq

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Zyro Dynamics HR Help Desk",
    page_icon="🏢",
    layout="centered",
)

# ── Constants ────────────────────────────────────────────────────────────────
PDF_DIR     = "./hr_docs/"           # put PDFs here in your Streamlit repo
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL   = "llama3-70b-8192"

HR_KEYWORDS = [
    "leave", "vacation", "sick", "maternity", "paternity", "salary",
    "ctc", "pay", "bonus", "wfh", "work from home", "remote", "hybrid",
    "probation", "onboarding", "resignation", "notice", "performance",
    "appraisal", "pip", "conduct", "posh", "harassment", "laptop",
    "data", "security", "travel", "reimbursement", "expense", "employee",
    "hr", "policy", "zyro", "compensation", "benefits", "insurance",
    "gratuity", "holiday", "casual", "earned", "attendance",
]

OUT_OF_SCOPE_MSG = (
    "I'm sorry, but I can only answer HR-related questions based on Zyro Dynamics' "
    "official policy documents. Your question appears to be outside the scope of "
    "HR policies. Please reach out to the HR team directly for other queries, or "
    "ask me about topics like leave policy, compensation, WFH rules, performance "
    "reviews, or employee conduct."
)

RAG_PROMPT = """\
You are an intelligent HR Help Desk assistant for Zyro Dynamics Pvt. Ltd.
Answer ONLY from the provided HR policy context below. Be precise and professional.
If the answer is not in the context, say you couldn't find it in the HR documents.
Cite the source document when possible.

--- HR POLICY CONTEXT ---
{context}
--- END CONTEXT ---

Employee Question: {question}
Answer:"""

# ── Build RAG Pipeline (cached so it only runs once) ─────────────────────────
@st.cache_resource(show_spinner="🔧 Loading HR documents and building search index...")
def build_rag_pipeline():
    # 1. Load PDFs
    all_docs = []
    for pdf_path in sorted(glob.glob(PDF_DIR + "*.pdf")):
        loader = PyPDFLoader(pdf_path)
        pages  = loader.load()
        for doc in pages:
            doc.metadata["source"] = Path(pdf_path).stem
        all_docs.extend(pages)

    # 2. Chunk
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(all_docs)

    # 3. Embed + FAISS
    embeddings  = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever   = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.6}
    )

    # 4. LLM + Chain
    groq_key = st.secrets["GROQ_API_KEY"]
    llm      = ChatGroq(model=LLM_MODEL, temperature=0, max_tokens=512, api_key=groq_key)
    prompt   = ChatPromptTemplate.from_template(RAG_PROMPT)

    def format_docs(docs):
        return "\n\n".join(
            f"[{doc.metadata.get('source','?')} | p{doc.metadata.get('page','?')}]\n"
            f"{doc.page_content.strip()}" for doc in docs
        )

    chain = (
        {"context": retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )
    return retriever, chain


def is_in_scope(question: str) -> bool:
    return any(kw in question.lower() for kw in HR_KEYWORDS)


# ── UI ───────────────────────────────────────────────────────────────────────
st.title("🏢 Zyro Dynamics HR Help Desk")
st.markdown(
    "Welcome! I can answer your HR policy questions about **leave, salary, "
    "WFH, performance reviews, conduct, travel reimbursements**, and more."
)
st.divider()

# Load pipeline
retriever, rag_chain = build_rag_pipeline()

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I'm your Zyro Dynamics HR assistant. How can I help you today? 👋"}
    ]

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📚 Source Documents"):
                for s in msg["sources"]:
                    st.caption(f"• {s['document']} | Page {s['page']}")

# Input
if user_input := st.chat_input("Ask an HR policy question..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate answer
    with st.chat_message("assistant"):
        if not is_in_scope(user_input):
            response = OUT_OF_SCOPE_MSG
            sources  = []
            st.markdown(f"🚫 {response}")
        else:
            with st.spinner("Searching HR policies..."):
                source_docs = retriever.invoke(user_input)
                response    = rag_chain.invoke(user_input)
                sources     = [
                    {"document": d.metadata.get("source","?"), "page": d.metadata.get("page","?")}
                    for d in source_docs
                ]
            st.markdown(response)
            if sources:
                with st.expander("📚 Source Documents"):
                    seen = set()
                    for s in sources:
                        key = (s["document"], s["page"])
                        if key not in seen:
                            st.caption(f"• {s['document']} | Page {s['page']}")
                            seen.add(key)

    st.session_state.messages.append({"role": "assistant", "content": response, "sources": sources})
