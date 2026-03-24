# 🧠 QnA Assistant with PDF + Chat

This project is a **Streamlit-based Chat Assistant** that can:
- Answer general questions using OpenAI GPT models
- Extract and answer questions from uploaded PDF documents using **Groq LLM** + **RAG**
- Maintain session-aware chat history for context-rich interactions

## 🚀 Features

- 📄 Upload PDFs and query them using LLMs (RAG with LangChain)
- 🤖 Choose from multiple OpenAI models: `gpt-4`, `gpt-4-turbo`, `gpt-4o`
- ⚙️ Powered by:
  - **LangChain**
  - **HuggingFace Embeddings**
  - **ChromaDB Vector Store**
  - **Groq + OpenAI**
- 🧠 Maintains chat memory per session

## 📂 Folder Contents

```
project/
├── app.py              # Main Streamlit app
├── requirements.txt    # Python dependencies
├── .env                # API keys (not committed)
```

## 🧪 Requirements

Install all dependencies using:

```bash
pip install -r requirements.txt
```

Make sure you have Python 3.10+.

## 🔐 Environment Setup

Create a `.env` file in the root with the following:

```env
OPENAI_API_KEY=your-openai-api-key
GROQ_API_KEY=your-groq-api-key
HF_TOKEN=your-huggingface-token
```

> `HF_TOKEN` is used for HuggingFace Embeddings  
> `GROQ_API_KEY` is required to use Groq's `llama3` model on PDFs  
> `OPENAI_API_KEY` is required for general chat using OpenAI

## ▶️ Running the App

```bash
streamlit run app.py
```

## 💡 Behavior

- **PDF Uploaded** → Uses Groq + RAG to answer PDF-related questions
- **No PDF** → Defaults to OpenAI GPT chat
- **Chat Memory**:
  - Session-aware memory via LangChain for Groq
  - App-level history for OpenAI chat

## 🛠️ Tech Stack

- `Streamlit` for the UI
- `LangChain` to manage chaining and memory
- `ChromaDB` as vectorstore
- `HuggingFace` Sentence Transformers for embedding
- `Groq` + `OpenAI` for LLM inference


## 👨‍💻 Author

Developed by **Ruthik Chitti** This is done in as chekc for the branch

