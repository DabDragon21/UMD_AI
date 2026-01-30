import os
import langchain
import langchain_core
import streamlit as st

st.write("langchain version:", langchain.__version__)
st.write("langchain_core version:", langchain_core.__version__)
st.write("langchain.chains contents:", dir(langchain.chains))

import google.generativeai as genai
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough



api_key = st.secrets["key"]
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
vectorstore_path = "edu_faiss_index"
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# load saved vector or store new vectors
if os.path.exists(vectorstore_path):
    print("🔄 Loading cached vectorstore...")
    vectorstore = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)
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
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(vectorstore_path)
    print("Vectorstore saved")
    st.write("Vectorstore saved")

#set up LLM
LLM = GoogleGenerativeAI(model="gemini-2.5-pro")

#build RAG
retriever = vectorstore.as_retriever()
prompt_template = ChatPromptTemplate.from_template("""
You are an expert education assistant.

Use the following research excerpts to improve the lesson plan.
If the research is irrelevant, say so explicitly.

Research context:
{context}

Lesson plan and request:
{input}
""")

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {
        "context": retriever | format_docs,
        "input": RunnablePassthrough(),
    }
    | prompt_template
    | LLM
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

                query = f"""
                Update the following lesson plan using best practices from educational research.

                User request:
                {prompt}

                Lesson plan:
                {truncated}
                """

                answer = rag_chain.invoke(query)

                st.text_area("Updated Lesson Plan", answer, height=300)

if __name__ == "__main__":
    main()

