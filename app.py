import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Set Streamlit page config
st.set_page_config(page_title="Ask-n-Get 📄", layout="wide")
st.title("Ask-n-Get 🤖📄")

# User session handling
if "user_id" not in st.session_state:
    st.session_state.user_id = os.urandom(4).hex()

if "message_history" not in st.session_state:
    st.session_state.message_history = []

user_id = st.session_state.user_id
message_history = st.session_state.message_history

uploaded_files = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)

# Chat model config
llm_groq = ChatGroq(temperature=0, groq_api_key=GROQ_API_KEY, model_name="gemma-7b-it")
llm_openai = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)

retriever = None

if uploaded_files:
    raw_text = ""
    for i, pdf in enumerate(uploaded_files):
        temp_path = f"./temp_{user_id}_{i}.pdf"
        with open(temp_path, "wb") as f:
            f.write(pdf.read())
        loader = PyPDFLoader(temp_path)
        pages = loader.load_and_split()
        for page in pages:
            raw_text += page.page_content
        os.remove(temp_path)

    # Split and embed
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.create_documents([raw_text])

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=f"chroma_db_{user_id}"
    )
    vectorstore.persist()
    retriever = vectorstore.as_retriever()

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm_groq,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False
    )

# Chat input
user_input = st.chat_input("Ask me anything...")

if user_input:
    st.chat_message("user").write(user_input)
    message_history.append(HumanMessage(content=user_input))

    if retriever:
        response = qa_chain.run(user_input)
    else:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant."),
            *[(msg.role, msg.content) for msg in message_history],
            ("human", "{question}")
        ])
        chain = prompt | llm_openai
        response = chain.invoke({"question": user_input}).content

    message_history.append(AIMessage(content=response))
    st.chat_message("ai").write(response)
