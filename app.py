import streamlit as st
from streamlit_oauth import OAuth2Component
import requests

# Get secrets
client_id = st.secrets["GOOGLE_CLIENT_ID"]
client_secret = st.secrets["GOOGLE_CLIENT_SECRET"]

# Setup OAuth component
oauth2 = OAuth2Component(
    client_id=client_id,
    client_secret=client_secret,
    authorize_url="https://accounts.google.com/o/oauth2/v2/auth",
    token_url="https://oauth2.googleapis.com/token",
    redirect_uri="https://asknget.streamlit.app/"
)

# OAuth params for Google
params = {
    "client_id": client_id,
    "response_type": "code",
    "redirect_uri": oauth2.redirect_uri,
    "scope": "https://www.googleapis.com/auth/userinfo.email https://www.googleapis.com/auth/userinfo.profile",
    "access_type": "offline",
    "prompt": "consent"
}

# Show login button
token = oauth2.authorize_button("Login with Google", params=params, key="google")

# Once logged in, get user info
if token:
    user_info = requests.get(
        "https://www.googleapis.com/oauth2/v2/userinfo",
        headers={"Authorization": f"Bearer {token['access_token']}"}
    ).json()

    st.success(f"Welcome, {user_info['name']} 👋")
    st.image(user_info["picture"])
    st.write("Email:", user_info["email"])

    # ✅ Now your full app code continues here...
else:
    st.info("Please login to use this app.")
    st.stop()
import sys
import pysqlite3
sys.modules["sqlite3"] = pysqlite3
import streamlit as st
from dotenv import load_dotenv
import os

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
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
from langchain_huggingface import HuggingFaceEmbeddings
import langchain

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "Q&A WITH DOCUMENT")


# Load .env for API keys
load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")



st.set_page_config(page_title="Q&A Assistant with PDF + Chat", layout="wide")
st.title("🧠QnA Assistant")

# Sidebar model settings
selected_llm = st.sidebar.selectbox("🤖 OpenAI Model", ["gpt-4o", "gpt-4-turbo", "gpt-4"])
temperature = st.sidebar.slider("Temperature (OpenAI)", 0.0, 1.0, 0.7)
max_tokens = st.sidebar.slider("Max Tokens (OpenAI)", 50, 500, 200)

if st.sidebar.button("🧹 Clear OpenAI Chat History"):
    st.session_state["openai_history"] = []

# Upload PDF
uploaded_files = st.file_uploader("📄 Upload a PDF file ", type=["pdf"], accept_multiple_files=True)
user_input = st.text_input("💬I'm Idle,Shot me with some questions!")

# Store session history for Groq
session_id = "default"
if 'store' not in st.session_state:
    st.session_state.store = {}

def get_session_history(session_id) -> BaseChatMessageHistory:
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

# 1. If PDF uploaded → Use Groq + RAG
if uploaded_files and GROQ_API_KEY:
    st.write("Groq....")

    # Load and process PDFs
    documents = []
    for uploaded_file in uploaded_files:
        temp_path = "./temp.pdf"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        loader = PyPDFLoader(temp_path)
        docs = loader.load()
        documents.extend(docs)

    # Split and embed
    splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    splits = splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    # Groq LLM
    llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama3-8b-8192")

    # Prompt templates
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

    # Create chain
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

    if user_input:
        session_history = get_session_history(session_id)
        response = conversation_chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}},
        )
        st.write("🤖 Assistant:", response['answer'])
        with st.expander("🕓 Chat History"):
            for msg in session_history.messages:
                role = "🧑‍💬 You" if msg.type == "human" else "🤖 Assistant"
                st.markdown(f"**{role}:** {msg.content}")

# 2. If no PDF → Use OpenAI GPT for QnA with memory
elif not uploaded_files and user_input:
    st.write("💡 Using OpenAI GPT for general QnA...")

    if "openai_history" not in st.session_state:
        st.session_state["openai_history"] = []

    # Prompt with memory
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

# Else: Missing input
elif user_input:
    st.warning("⚠️ Please upload a PDF (for Groq) or check if keys are properly set in .env")
