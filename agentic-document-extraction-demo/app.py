import os
import json
import numpy as np
import re  # For text cleaning
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from typing import Dict, Any
from PIL import Image
import fitz  # PyMuPDF for PDF handling
from vision_agent.agent import VisionAgent
import vision_agent.tools as T
from sentence_transformers import SentenceTransformer, util  # Reranking

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Directory to store uploaded documents
UPLOAD_DIR = os.path.abspath("uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# JSON file to store extracted data
DATA_FILE = "extracted_data.json"

# Load sentence transformer model for reranking
reranker_model = SentenceTransformer("all-MiniLM-L6-v2")

class DocumentExtractor:
    def __init__(self, verbose: bool = False):
        """Initialize the document extractor with Landing AI Vision Agent."""
        self.agent = VisionAgent(verbosity=1 if verbose else 0)
        self.verbose = verbose

    def load_document(self, file_path: str, page_num: int = 0):
        """Load a PDF or image document and convert to an image array."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        file_ext = os.path.splitext(file_path)[1].lower()

        if file_ext == ".pdf":
            doc = fitz.open(file_path)
            if doc.page_count == 0:
                raise ValueError("PDF document has no pages")

            if page_num < 0 or page_num >= doc.page_count:
                raise ValueError(f"Invalid page number {page_num}. Document has {doc.page_count} pages.")

            page = doc.load_page(page_num)
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            return np.array(img)

        else:
            img = Image.open(file_path)
            return np.array(img)

    def extract_from_document(self, file_path: str, prompt: str, page_num: int = 0) -> Dict[str, Any]:
        """Extract information from a document using the Vision Agent."""
        image_data = self.load_document(file_path, page_num)
        response = T.document_qa(prompt, image=image_data)
        return response

extractor = DocumentExtractor()

def save_data_to_file(data):
    """Save extracted data to a file."""
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            try:
                existing_data = json.load(f)
            except json.JSONDecodeError:
                existing_data = {}  # Handle corrupted JSON file
    else:
        existing_data = {}

    existing_data.update(data)

    with open(DATA_FILE, "w") as f:
        json.dump(existing_data, f, indent=4)

def load_data_from_file():
    """Load extracted data from a file."""
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}  # Handle corrupted JSON file
    return {}

@app.route("/upload", methods=["POST"])
def upload_document():
    """Upload a document and extract text."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    file.save(file_path)
    
    prompt = request.form.get("prompt", "Extract relevant information")
    try:
        extracted_data = extractor.extract_from_document(file_path, prompt)
        save_data_to_file({file.filename: extracted_data})
        return jsonify({"message": "File uploaded and processed", "data": extracted_data}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/query", methods=["POST"])
def query_data():
    data = request.json
    query = data.get("query", "").strip()

    if not query:
        return jsonify({"error": "Query is required"}), 400

    stored_data = load_data_from_file()
    if not stored_data:
        return jsonify({"error": "No extracted data found. Upload documents first."}), 404

    query_results = []

    # Process query against each document
    for doc_name, extracted_content in stored_data.items():
        try:
            text = extracted_content if isinstance(extracted_content, str) else json.dumps(extracted_content)
            query_embedding = reranker_model.encode(query, convert_to_tensor=True)
            text_embedding = reranker_model.encode(text, convert_to_tensor=True)
            similarity_score = util.pytorch_cos_sim(query_embedding, text_embedding).item()
            
            # Extract skills and experience keywords
            keywords = set()
            for line in text.split("\n"):
                matches = re.findall(r'\b(Java|Python|Django|Flask|React|SQL|Machine Learning|5\+ years|3 years|Senior)\b', line, re.IGNORECASE)
                keywords.update(matches)
            
            query_results.append({
                "name": doc_name,
                "score": similarity_score,
                "result": ", ".join(keywords)
            })
        except Exception as e:
            continue

    # Sort results by similarity score and get top 3
    query_results = sorted(query_results, key=lambda x: x["score"], reverse=True)[:3]
    
    return jsonify({"top_candidates": query_results}), 200

if __name__ == "__main__":
    app.run(debug=True)
