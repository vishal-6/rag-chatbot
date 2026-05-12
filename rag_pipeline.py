from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

def get_qa_chain():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(
        "faiss_index", embeddings, allow_dangerous_deserialization=True
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    llm = ChatGroq(
        model_name="llama-3.1-8b-instant",  # Fixed: llama3-8b-8192 is decommissioned
        temperature=0.2,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )

    def qa_chain(query):
        docs = retriever.invoke(query)
        context = "\n".join([doc.page_content for doc in docs])
        prompt = f"Use the following context to answer the question.\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        response = llm.invoke(prompt)
        return {"result": response.content}

    return qa_chain