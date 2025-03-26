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

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Directory to store uploaded documents
UPLOAD_DIR = os.path.abspath("uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# JSON file to store extracted data
DATA_FILE = "extracted_data.json"


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
def upload_documents():
    """Upload multiple documents and extract text."""
    if "files" not in request.files:
        return jsonify({"error": "No files uploaded"}), 400

    files = request.files.getlist("files")
    prompt = request.form.get("prompt", "Extract all text from this document.")

    extracted_data = {}

    for file in files:
        try:
            file_path = os.path.join(UPLOAD_DIR, file.filename)
            file.save(file_path)

            extracted_text = extractor.extract_from_document(file_path, prompt)
            extracted_data[file.filename] = extracted_text
        except FileNotFoundError as e:
            return jsonify({"error": f"File error: {str(e)}"}), 400
        except ValueError as e:
            return jsonify({"error": f"Document error: {str(e)}"}), 400
        except Exception as e:
            return jsonify({"error": f"Failed to process {file.filename}: {str(e)}"}), 500

    # Save extracted data to a file
    save_data_to_file(extracted_data)

    return jsonify({"message": "Documents uploaded successfully"}), 200


import re

@app.route("/query", methods=["POST"])
def query_data():
    """Retrieve extracted text from extracted_data.json and process with VisionAgent."""
    data = request.json
    query = data.get("query", "").strip()

    if not query:
        return jsonify({"error": "Query is required"}), 400

    stored_data = load_data_from_file()

    if not stored_data:
        return jsonify({"error": "No extracted data found. Upload documents first."}), 404

    query_results = {}

    print("\n🔹 Stored Data Before Querying:")
    print(json.dumps(stored_data, indent=2))  # Debugging line

    # Process query against each document using AI
    for doc_name in stored_data.keys():
        try:
            # Load the document as an image
            image_data = extractor.load_document(os.path.join(UPLOAD_DIR, doc_name))

            # Generate an AI-based response for the query
            response = T.document_qa(query, image=image_data)

            # Ensure response is a dictionary
            if isinstance(response, dict):
                answer = response.get("answer", "No relevant answer found.")
            else:
                answer = response  # If it's a string, return as is.

            # Exclude results with "I cannot find the answer in the provided document."
            if answer != "I cannot find the answer in the provided document.":
                query_results[doc_name] = answer

        except Exception as e:
            query_results[doc_name] = {"error": f"Failed to process query: {str(e)}"}

    if not query_results:
        return jsonify({"query_results": "No relevant response found"}), 200

    return jsonify({"query_results": query_results}), 200




if __name__ == "__main__":
    app.run(debug=True)


