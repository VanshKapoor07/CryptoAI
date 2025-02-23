from flask import Flask, request, jsonify
#from flask_cors import CORS
import pickle
import time
#from dotenv import load_dotenv
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceHub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import UnstructuredURLLoader

app = Flask(__name__)
#CORS(app)  # Enable CORS for frontend requests

#load_dotenv()
urls = ['https://en.wikipedia.org/wiki/Elon_Musk','https://en.wikipedia.org/wiki/Mark_Zuckerberg']
file_path = "faiss_store_hf.pkl"

custom_prompt_template = """Use the following pieces of context to answer the question. Give a straight, direct answer.

Context: {context}

Question: {question}

Direct Answer:"""

PROMPT = PromptTemplate(
    template=custom_prompt_template,
    input_variables=["context", "question"]
)

# Initialize LLM
llm = HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    model_kwargs={
        "temperature": 0.5,
        "max_length": 500,
        "task": "text-generation"
    }
)

# Load or Create FAISS Vector Store (Embeddings Computed Once)
try:
    with open(file_path, "rb") as f:
        vectorstore = pickle.load(f)
    print("Loaded existing FAISS index.")
except FileNotFoundError:
    print("Creating new FAISS index...")
    valid_urls = [url for url in urls if url.strip()]
    loader = UnstructuredURLLoader(urls=valid_urls)
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=500,
        chunk_overlap=200
    )

    docs = text_splitter.split_documents(data)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)

    with open(file_path, "wb") as f:
        pickle.dump(vectorstore, f)
    print("FAISS index created and saved.")

@app.route('/askques', methods=['POST'])
def signin():
    data = request.json
    question = data.get("question", "")

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        return_source_documents=False,
        chain_type_kwargs={"prompt": PROMPT}
    )

    result = qa_chain({"query": question})
    answer = result["result"].strip()

    if "Context:" in answer:
            answer = answer.split("Direct Answer:")[-1]

    return jsonify({"answer": answer})

if __name__ == '__main__':
    app.run(debug=True)
