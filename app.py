
# import streamlit as st
# from PyPDF2 import PdfReader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# import os
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# import google.generativeai as genai
# from langchain_community.vectorstores import FAISS
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains import create_retrieval_chain  # Updated import
# from langchain.chains.combine_documents import create_stuff_documents_chain  # Updated import
# from langchain_core.prompts import ChatPromptTemplate  # Updated import
# from dotenv import load_dotenv

# load_dotenv()
# api_key = os.getenv("GOOGLE_API_KEY")
# genai.configure(api_key=api_key)

# def get_pdf_text(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         try:
#             pdf_reader = PdfReader(pdf)
#             for page in pdf_reader.pages:
#                 page_text = page.extract_text()
#                 if not page_text.strip():
#                     st.warning(f"⚠️ Detected image-based page in {pdf.name}. Text-based PDFs only.")
#                     continue
#                 text += page_text
#         except Exception as e:
#             st.error(f"Error reading {pdf.name}: {str(e)}")
#     return text

# def get_text_chunks(text):
#     if not text.strip():
#         st.error("No text extracted from PDFs. Please upload text-based PDFs.")
#         return []
    
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=10000, 
#         chunk_overlap=1000
#     )
#     chunks = text_splitter.split_text(text)
#     return [chunk for chunk in chunks if len(chunk.strip()) > 50]

# def get_vector_store(text_chunks):
#     if not text_chunks:
#         st.error("No valid text chunks to process!")
#         return
    
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
#     vector_store.save_local("faiss_index")

# def get_conversational_chain():
#     prompt = ChatPromptTemplate.from_template("""
#     Answer the question as detailed as possible from the provided context.
#     If the answer isn't in the context, say "answer is not available in the context".
    
#     Context: {context}
#     Question: {input}
#     Answer:""")

#     # Updated model name to the current Gemini version
#     model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.3)
    
#     # Updated chain creation
#     document_chain = create_stuff_documents_chain(model, prompt)
#     return document_chain

# def user_input(user_question):
#     try:
#         embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
#         new_db = FAISS.load_local(
#             "faiss_index", 
#             embeddings, 
#             allow_dangerous_deserialization=True
#         )
        
#         # Updated retrieval approach
#         retriever = new_db.as_retriever()
#         document_chain = get_conversational_chain()
#         retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
#         response = retrieval_chain.invoke({"input": user_question})
#         st.write("Reply:", response["answer"])
        
#     except Exception as e:
#         st.error(f"Error processing your question: {str(e)}")

# def main():
#     st.set_page_config("Chat PDF")
#     st.header("Chat with PDF using Gemini💁")

#     user_question = st.text_input("Ask a Question from the PDF Files")

#     if user_question:
#         user_input(user_question)

#     with st.sidebar:
#         st.title("Menu:")
#         pdf_docs = st.file_uploader(
#             "Upload your PDF Files and Click on the Submit & Process Button", 
#             accept_multiple_files=True,
#             type="pdf"
#         )
#         if st.button("Submit & Process"):
#             with st.spinner("Processing..."):
#                 raw_text = get_pdf_text(pdf_docs)
#                 if not raw_text.strip():
#                     st.error("No text could be extracted. Please upload text-based PDFs.")
#                     return
                
#                 text_chunks = get_text_chunks(raw_text)
#                 if text_chunks:
#                     get_vector_store(text_chunks)
#                     st.success("Processing complete! You can now ask questions.")
#                 else:
#                     st.error("No valid text chunks were created from the PDFs.")

# if __name__ == "__main__":
#     main()

import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

def show_human_support():
    st.markdown("""
    **👋 Need human assistance? Here's how to reach us:**
    
    📞 **Phone Support**: +91 7085959173 
    ✉️ **Email**: aritra9901@gmail.com  
    🕒 **Hours**: Monday-Friday, 9AM-5PM EST
    
    *Our team will get back to you within 24 hours.*
    """)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if not page_text.strip():
                    st.warning(f"⚠️ Detected image-based page in {pdf.name}. Text-based PDFs only.")
                    continue
                text += page_text
        except Exception as e:
            st.error(f"Error reading {pdf.name}: {str(e)}")
    return text

def get_text_chunks(text):
    if not text.strip():
        st.error("No text extracted from PDFs. Please upload text-based PDFs.")
        return []
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000, 
        chunk_overlap=1000
    )
    chunks = text_splitter.split_text(text)
    return [chunk for chunk in chunks if len(chunk.strip()) > 50]

def get_vector_store(text_chunks):
    if not text_chunks:
        st.error("No valid text chunks to process!")
        return
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt = ChatPromptTemplate.from_template("""
    Answer the question as detailed as possible from the provided context.
    If the answer isn't in the context, say "answer is not available in the context".
    
    Context: {context}
    Question: {input}
    Answer:""")

    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.3)
    
    document_chain = create_stuff_documents_chain(model, prompt)
    return document_chain

def user_input(user_question):
    support_triggers = ["human", "agent", "support", "representative", "talk to someone"]
    if any(trigger in user_question.lower() for trigger in support_triggers):
        show_human_support()
        return
    
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        new_db = FAISS.load_local(
            "faiss_index", 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        
        retriever = new_db.as_retriever()
        document_chain = get_conversational_chain()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        response = retrieval_chain.invoke({"input": user_question})
        st.write("Reply:", response["answer"])
        
    except Exception as e:
        st.error(f"Error processing your question: {str(e)}")

def main():
    st.set_page_config("Ai-Powered Insurance Policy ChatBot")
    st.header("Chat with Agent regarding insurance ")

    user_question = st.text_input("Ask a Question")
    st.caption("💡 Tip: Type 'support', 'agent', or 'human' to connect with our support team")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files and Click on the Submit & Process Button", 
            accept_multiple_files=True,
            type="pdf"
        )
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                if not raw_text.strip():
                    st.error("No text could be extracted. Please upload text-based PDFs.")
                    return
                
                text_chunks = get_text_chunks(raw_text)
                if text_chunks:
                    get_vector_store(text_chunks)
                    st.success("Processing complete! You can now ask questions.")
                else:
                    st.error("No valid text chunks were created from the PDFs.")

if __name__ == "__main__":
    main()