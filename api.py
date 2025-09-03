from dotenv import load_dotenv
import os
import streamlit as st
from google import genai
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.chains.retrieval_qa.base import RetrievalQA
import tkinter as tk
from tkinter import filedialog

load_dotenv()

api_key = st.secrets("key")
os.environ["GOOGLE_API_KEY"] = api_key

#load file
databases = [
    "Differentiation_Guides.pdf",
    "MaCann Yadav CT elem.pdf",
    "MultilingualLearnersGuide.pdf",
    "UDL_Table_accessible_CS.pdf",
    "CS_Pedagogy.pdf",
    "CS_Content.pdf"
]
vectorstore_path = "edu_chroma_index"
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# load saved vector or store new vectors
if os.path.exists(vectorstore_path):
    print("🔄 Loading cached vectorstore...")
    vectorstore = Chroma(persist_directory=vectorstore_path, embedding_function=embeddings)
else:
    print("⚙️ Building new vectorstore...")
    all_bases = []
    for file in databases:
        loader = PyMuPDFLoader(file)
        docs = loader.load()
        all_bases.extend(docs)

    # Chunk text
    splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=64)
    chunks = splitter.split_documents(all_bases)

    # Create vectorstore and save
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=vectorstore_path
    )
    vectorstore.persist()
    print("Vectorstore saved")

#set up LLM
LLM = GoogleGenerativeAI(model="gemini-2.5-pro")

#build RAG
QA_chain = RetrievalQA.from_chain_type(
    llm=LLM,
    retriever=vectorstore.as_retriever(),
    chain_type="stuff"
)




def main():
    st.set_page_config(page_title="AI Lesson Assistant", layout="centered")
    st.title("📚 AI Lesson Assistant")

    st.markdown("Upload your educational PDF and enter a prompt to update the lesson plan.")

    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    prompt = st.text_area("Enter your prompt", height=100)

    if uploaded_file and prompt:
        if st.button("Generate Updated Lesson Plan"):
            with st.spinner("Processing..."):
                # Save file temporarily to load it
                with open("temp_uploaded.pdf", "wb") as f:
                    f.write(uploaded_file.read())

                loader = PyMuPDFLoader("temp_uploaded.pdf")
                docs = loader.load()
                input_txt = "".join(doc.page_content for doc in docs)
                truncated = input_txt[:8000]

                response = LLM.invoke(f"Update this lesson plan based on this prompt: {prompt}: {truncated}")
                st.text_area("Updated Lesson Plan", response, height=300)


if __name__ == "__main__":
    main()

