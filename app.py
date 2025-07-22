import streamlit as st
import sys
import pysqlite3
sys.modules["sqlite3"] = pysqlite3

import sqlite3
import os
from dotenv import load_dotenv
from datetime import datetime

from streamlit_oauth import OAuth2Component

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.output_parsers import StrOutputParser

# Avoid HuggingFace conflict: use alias
from langchain.embeddings import HuggingFaceEmbeddings as LCHuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer

# === Environment setup ===
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "Q&A WITH DOCUMENT")
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# === UI ===
st.set_page_config(page_title="QnA Assistant", layout="wide")
st.title("🧠 QnA Assistant")

# Sidebar: OpenAI model settings
selected_llm = st.sidebar.selectbox("🤖 OpenAI Model", ["gpt-4o", "gpt-4-turbo", "gpt-4"])
temperature = st.sidebar.slider("Temperature (OpenAI)", 0.0, 1.0, 0.7)
max_tokens = st.sidebar.slider("Max Tokens (OpenAI)", 50, 500, 200)
if st.sidebar.button("🧹 Clear OpenAI Chat History"):
    st.session_state["openai_history"] = []

# Upload PDF
uploaded_files = st.file_uploader("📄 Upload a PDF file", type=["pdf"], accept_multiple_files=True)
user_input = st.text_input("💬 I'm Idle, Shot me with some questions!")

# Session state for Groq history
session_id = "default"
if "store" not in st.session_state:
    st.session_state.store = {}

def get_session_history(session_id) -> BaseChatMessageHistory:
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

# === Path 1: PDF + Groq + RAG ===
if uploaded_files and GROQ_API_KEY:
    st.write("💡 Using Groq + RAG...")

    documents = []
    for uploaded_file in uploaded_files:
        temp_path = "./temp.pdf"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        loader = PyPDFLoader(temp_path)
        docs = loader.load()
        documents.extend(docs)

    # Split docs
    splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    splits = splitter.split_documents(documents)

    # Load SBERT manually to avoid meta tensor error
    sbert_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = CHuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model=sbert_model
    )

    # Vector DB
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    # Groq LLM
    llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama3-8b-8192")

    # Prompt setup
    standalone_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given a chat history and a follow-up question, rewrite it as a standalone question."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "Use the retrieved context to answer the question. If unknown, say so.\n\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    # Chains
    history_aware_retriever = create_history_aware_retriever(llm, retriever, standalone_prompt)
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

    conversation_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    # QnA flow
    if user_input:
        session_history = get_session_history(session_id)
        response = conversation_chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}},
        )
        st.write("🤖 Assistant:", response["answer"])
        with st.expander("🕓 Chat History"):
            for msg in session_history.messages:
                role = "🧑‍💬 You" if msg.type == "human" else "🤖 Assistant"
                st.markdown(f"**{role}:** {msg.content}")

# === Path 2: No PDF → OpenAI GPT QnA ===
elif not uploaded_files and user_input:
    st.write("💡 Using OpenAI GPT for general QnA...")

    if "openai_history" not in st.session_state:
        st.session_state["openai_history"] = []

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Answer in a kind, layman-friendly way."),
        MessagesPlaceholder("chat_history"),
        ("user", "{input}")
    ])

    llm = ChatOpenAI(model=selected_llm, temperature=temperature, max_tokens=max_tokens)
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser

    history_messages = st.session_state["openai_history"]
    inputs = {
        "input": user_input,
        "chat_history": history_messages
    }

    answer = chain.invoke(inputs)

    history_messages.append({"role": "user", "content": user_input})
    history_messages.append({"role": "assistant", "content": answer})

    st.write("🤖 Assistant:", answer)
    with st.expander("🕓 Chat History"):
        for msg in history_messages:
            role = "🧑‍💬 You" if msg["role"] == "user" else "🤖 Assistant"
            st.markdown(f"**{role}:** {msg['content']}")

# === Catch ===
elif user_input:
    st.warning("⚠️ Please upload a PDF (for Groq) or check if keys are properly set in .env")
