# Project Report: AI-Powered Insurance Policy Information Chatbot

## Problem Statement
Insurance companies often receive numerous queries about policy types, coverage options, premiums, and claim processes. Addressing these questions efficiently is crucial for customer satisfaction and informed decision-making. This project aims to develop an AI-powered chatbot that provides accurate, real-time responses to such queries using a document-based knowledge base.

## Objective
To build an intelligent, interactive chatbot that:
- Understands natural language queries.
- Retrieves relevant information from uploaded insurance policy documents.
- Escalates complex or unsupported queries to human agents.
- Provides a user-friendly web interface.

## Technology Stack

| Component                     | Tool/Library                                     | Purpose |
|------------------------------|--------------------------------------------------|---------|
| Frontend UI                  | Streamlit                                       | Builds a simple, web-based chatbot interface. |
| PDF Text Extraction          | PyPDF2                                           | Parses text content from uploaded PDF files. |
| Text Chunking                | LangChain's RecursiveCharacterTextSplitter      | Splits extracted text into manageable chunks for embedding. |
| Embedding Model              | GoogleGenerativeAIEmbeddings (embedding-001)    | Generates vector embeddings for text chunks using Google Gemini. |
| Vector Database              | FAISS (via LangChain)                           | Stores and retrieves similar document chunks based on query similarity. |
| Language Model               | ChatGoogleGenerativeAI (gemini-1.5-pro-latest)  | Understands and responds to natural language queries using context. |
| RAG Pipeline                 | LangChain create_retrieval_chain()              | Combines vector-based document retrieval with LLM response generation. |
| Environment Variable Loader | python-dotenv                                    | Loads sensitive keys and configurations (e.g., Google API key). |

## Workflow

1. **PDF Upload and Text Extraction**: Users upload insurance-related PDF documents. PyPDF2 reads and extracts text from each page.
2. **Text Preprocessing**: The extracted text is split into smaller chunks using LangChain's text splitter.
3. **Vector Embedding and Storage**: Chunks are converted into vector embeddings using Google's embedding model and stored in a FAISS vector index.
4. **Question Answering**:
   - User enters a question in the chatbot.
   - Relevant chunks are retrieved from the vector store.
   - Gemini LLM generates a response using the provided context.
5. **Fallback Mechanism**: If the question includes keywords like "human", "support", or "agent", the chatbot displays human contact information for escalation.

## Features

- Natural language processing using Google Gemini LLM.
- Real-time document-aware answering using RAG architecture.
- Upload and process multiple PDF documents.
- Built-in fallback to human agents for unsupported queries.
- Minimal, responsive user interface built with Streamlit.

## Conclusion

This chatbot provides an efficient, scalable solution to handle common customer queries in the insurance domain. It reduces the burden on human support teams while offering quick and reliable information. The combination of LLMs, document retrieval, and a clean UI ensures a smooth and informative user experience.

The project architecture is modular and can be extended to support more domains, file formats, or multilingual interactions in the future.
