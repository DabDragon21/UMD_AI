from dotenv import load_dotenv
import os
from google import genai
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.chains.retrieval_qa.base import RetrievalQA
import tkinter as tk
from tkinter import filedialog

load_dotenv()

api_key = os.getenv("key")
os.environ["GOOGLE_API_KEY"] = api_key

client = genai.Client(api_key=api_key)

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

window = tk.Tk()
window.title("AI Lesson Assistant")
window.geometry("600x400")

# Label above the button
label = tk.Label(window, text="📎 Upload your educational PDF for analysis:", font=("Arial", 12))
label.pack(pady=10)

# Text area for output
output_text = tk.Text(window, height=15, width=70, wrap="word")
output_text.pack(pady=10)

def upload_file():
    filepath = filedialog.askopenfilename()
    if filepath:
        input_loader = PyMuPDFLoader(filepath)
        input_docs = input_loader.load()
        input_txt = "".join(doc.page_content for doc in input_docs)
        truncated = input_txt[:8000]
        prompt = input("Enter your prompt here: ")
        response = LLM.invoke(f"Update this lesson plan based on this prompt {prompt}: {truncated}")

        output_text.delete("1.0", tk.END)
        output_text.insert(tk.END, response)

button = tk.Button(window, text="Upload PDF", command = upload_file)
button.pack()

window.mainloop()
