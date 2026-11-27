# RAG Chatbot 🤖

An AI-powered Retrieval-Augmented Generation (RAG) chatbot built using Python and Streamlit.

## About the Project

This application allows users to upload documents and ask questions based on their content. It uses embeddings and vector search to retrieve relevant information and generates answers using an OpenRouter language model.

## Features

- Upload multiple documents.
- Supports PDF, TXT, MD, DOCX, and CSV files.
- Document processing and chunking.
- Semantic search using embeddings.
- Context-based question answering.
- Streaming AI responses.
- Source information for retrieved content.

## Technologies Used

- Python
- Streamlit
- LangChain
- OpenRouter API
- In-memory vector store

## How to Run

1. Clone this repository.
2. Install the required dependencies:

   `pip install -r requirements.txt`

3. Create a `.env` file and configure your API key if required by your application.
4. Run the application:

   `python -m streamlit run app.py`

## Security

Never upload API keys or `.env` files to GitHub.

## Author

Shravani# RAG-Chatbot
An AI-powered RAG chatbot built with Python and Streamlit that allows users to upload documents and ask questions based on their content using OpenRouter LLMs.
