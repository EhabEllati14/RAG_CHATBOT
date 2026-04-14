# RAG Chatbot - Mental Health Q&A

A Retrieval-Augmented Generation chatbot that answers mental health questions by retrieving relevant context from a dataset of real mental health texts and generating responses using a language model.

## About

This project builds a RAG pipeline that combines document retrieval with text generation to answer mental health-related queries. A CSV dataset containing mental health descriptions and categories is loaded, chunked, embedded into a vector store, and used as context for a generative model at query time.

## How It Works

The dataset is downloaded from Google Drive and each record's text is combined with its category label. These documents are split into 500-character chunks using LangChain's RecursiveCharacterTextSplitter. Chunks are embedded with the all-MiniLM-L6-v2 sentence transformer and stored in a ChromaDB vector store. At query time, the top 5 most relevant chunks are retrieved and passed as context to EleutherAI's GPT-Neo-125M through a prompt template. The full chain is wired together using LangChain's LCEL (LangChain Expression Language).

## Pipeline

1. Load and preprocess mental health text data from CSV
2. Chunk documents with RecursiveCharacterTextSplitter
3. Embed chunks using HuggingFace all-MiniLM-L6-v2
4. Store and retrieve from ChromaDB vector store
5. Generate answers with GPT-Neo-125M via a LangChain retrieval chain

## Tools

- Python
- LangChain, LangChain Community
- HuggingFace Transformers (GPT-Neo-125M, all-MiniLM-L6-v2)
- ChromaDB
- PyTorch
- Pandas

## Author

Ehab Ellati
