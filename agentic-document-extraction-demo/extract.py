import os
import json
import argparse
from dotenv import load_dotenv
from typing import Dict, List, Any, Optional
import time
import numpy as np
from PIL import Image
import fitz  # PyMuPDF for PDF handling

# Import Landing AI Vision Agent library
from vision_agent.agent import VisionAgent
from vision_agent.models import AgentMessage
import vision_agent.tools as T

# Load environment variables
load_dotenv()


class DocumentExtractor:
    def __init__(self, verbose: bool = False):
        """Initialize the document extractor with Landing AI Vision Agent."""
        self.agent = VisionAgent(verbosity=1 if verbose else 0)
        self.verbose = verbose

    def load_document(self, file_path: str, page_num: int = 0):
        """
        Load a document file (PDF or image) and convert to the appropriate format.

        For PDFs, converts the specified page to an image.
        For images, loads directly.

        Args:
            file_path: Path to the document file
            page_num: Page number to load for PDF documents (0-indexed)

        Returns:
            numpy.ndarray: The image data
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        file_ext = os.path.splitext(file_path)[1].lower()

        if file_ext == ".pdf":
            # Convert PDF to image using PyMuPDF
            doc = fitz.open(file_path)
            if doc.page_count == 0:
                raise ValueError("PDF document has no pages")

            # Check if page number is valid
            if page_num < 0 or page_num >= doc.page_count:
                raise ValueError(
                    f"Invalid page number {page_num}. Document has {doc.page_count} pages (0-{doc.page_count-1})"
                )

            # Get the specified page
            page = doc.load_page(page_num)
            pix = page.get_pixmap(
                matrix=fitz.Matrix(2, 2)
            )  # 2x zoom for better resolution

            # Convert to PIL Image
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            # Convert to numpy array
            return np.array(img)
        else:
            # Assume it's an image file
            img = Image.open(file_path)
            return np.array(img)

    def extract_from_document(
        self, file_path: str, prompt: str, page_num: int = 0
    ) -> Dict[str, Any]:
        """
        Extract information from a document using the Vision Agent.

        Args:
            file_path: Path to the document/image file
            prompt: The instruction for what to extract from the document
            page_num: Page number to process for PDF documents (0-indexed)

        Returns:
            Dictionary containing the extracted information
        """
        if self.verbose:
            print(f"Processing file: {file_path}")
            print(f"Using prompt: {prompt}")
            if page_num > 0:
                print(f"Processing page: {page_num}")

        # Load the document as an image
        try:
            image_data = self.load_document(file_path, page_num)

            # Use document_qa with the loaded image
            response = T.document_qa(prompt, image=image_data)

            return response
        except Exception as e:
            print(f"Error processing document: {str(e)}")
            import traceback

            traceback.print_exc()
            raise

    def format_response(self, response: Dict[str, Any]) -> str:
        """Format the response for CLI display."""
        if isinstance(response, dict):
            return json.dumps(response, indent=2)
        return str(response)


def main():
    parser = argparse.ArgumentParser(
        description="Extract information from documents using Landing AI Vision Agent"
    )
    parser.add_argument("file_path", help="Path to the document or image file")
    parser.add_argument(
        "--prompt",
        help="Instruction for what to extract from the document",
        default="Extract all text from this document and organize it.",
    )
    parser.add_argument(
        "--output",
        help="Path to save the extraction results (JSON format)",
        default=None,
    )
    parser.add_argument("--verbose", help="Enable verbose output", action="store_true")
    parser.add_argument(
        "--page",
        type=int,
        help="Page number to process for multi-page documents (0-indexed)",
        default=0,
    )

    args = parser.parse_args()

    extractor = DocumentExtractor(verbose=args.verbose)

    try:
        print(f"Processing document: {args.file_path}")
        print(f"Using prompt: {args.prompt}")
        if args.page > 0:
            print(f"Processing page: {args.page}")

        response = extractor.extract_from_document(
            args.file_path, args.prompt, args.page
        )

        if args.output:
            with open(args.output, "w") as f:
                json.dump(response, f, indent=2)
            print(f"Results saved to: {args.output}")

        print("\n===== EXTRACTION RESULTS =====\n")

        if isinstance(response, dict) and "content" in response:
            print(response["content"])
        else:
            print(extractor.format_response(response))

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
