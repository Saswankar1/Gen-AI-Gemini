import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""  # Ensure no NoneType errors
        except Exception as e:
            st.warning(f"Error reading {pdf.name}: {e}")
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=500)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks, append=False):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    try:
        # Check if the FAISS index exists and append if required
        if append and os.path.exists("faiss_index"):
            st.write("Loading existing FAISS index...")
            # Load the FAISS index with deserialization enabled
            vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            vector_store.add_texts(text_chunks)  # Add new chunks
        else:
            st.write("Creating a new FAISS index...")
            # If no index exists, create a new FAISS index from the text chunks
            vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        
        # Save the FAISS index locally
        vector_store.save_local("faiss_index")
        st.success("Vector store created/updated successfully!")
        
    except Exception as e:
        st.error(f"Error creating/updating FAISS index: {e}")


def get_conversational_chain():
    prompt_template = """
    Answer the following question based on the provided context. If the context does not contain the answer, respond with:
    "The answer is not available in the context."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    try:
        new_db = FAISS.load_local("faiss_index", embeddings)
        docs = new_db.similarity_search(user_question)
    except Exception as e:
        st.error(f"Error loading FAISS index: {e}")
        return
    
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply:", response["output_text"])

def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Gemini💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("📄 PDF Processor")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                if not raw_text:
                    st.error("No text found in the uploaded PDFs.")
                    return
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Processing complete! You can now ask questions.")

if __name__ == "__main__":
    main()
