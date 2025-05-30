import os
import fitz
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
from flask import Flask, render_template, request, jsonify
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
POPPLER_PATH = "/usr/bin"

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
VECTORSTORE_FOLDER = "vectorstores"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VECTORSTORE_FOLDER, exist_ok=True)

llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0.3, google_api_key=GEMINI_API_KEY)
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
retrievers = {}

def load_pdf_chunks(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = "\n".join([page.get_text() for page in doc])
        if text.strip():
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            return splitter.split_text(text)
    except Exception as e:
        print(f"[ML Extract FAIL] {e}")
    try:
        print("[INFO] Falling back to OCR...")
        pages = convert_from_path(pdf_path, dpi=300, poppler_path=POPPLER_PATH)
        ocr_text = ""
        for page in pages:
            ocr_text += pytesseract.image_to_string(page, lang="eng+hin")
        if ocr_text.strip():
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            return splitter.split_text(ocr_text)
    except Exception as e:
        print(f"[OCR FAIL] {e}")
        return []

def initialize_vectorstores():
    for filename in os.listdir(UPLOAD_FOLDER):
        if not filename.endswith(".pdf"):
            continue
        pdf_path = os.path.join(UPLOAD_FOLDER, filename)
        store_path = os.path.join(VECTORSTORE_FOLDER, filename)
        if os.path.exists(store_path):
            db = FAISS.load_local(store_path, embedding, allow_dangerous_deserialization=True)
        else:
            chunks = load_pdf_chunks(pdf_path)
            if not chunks:
                continue
            db = FAISS.from_texts(chunks, embedding)
            db.save_local(store_path)
        retrievers[filename] = db.as_retriever()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_question = request.json.get("message")
    if not user_question:
        return jsonify({"error": "Please provide a question"}), 400
    if not retrievers:
        return jsonify({"error": "No PDFs available to search"}), 400
    latest_pdf = sorted(retrievers.keys())[-1]
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retrievers[latest_pdf])
    answer = qa.run(user_question)
    return jsonify({"response": answer})

@app.route("/pdfs")
def list_pdfs():
    files = [f for f in os.listdir(UPLOAD_FOLDER) if f.lower().endswith(".pdf")]
    return jsonify(files)

if __name__ == "__main__":
    from werkzeug.utils import secure_filename
    initialize_vectorstores()
    app.run(host="0.0.0.0", port=7860)
    
