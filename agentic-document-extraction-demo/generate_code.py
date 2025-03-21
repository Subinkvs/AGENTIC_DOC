import os
import argparse
import json
from dotenv import load_dotenv
from typing import Dict, Any, Optional
import numpy as np
from PIL import Image
import fitz  # PyMuPDF for PDF handling

# Import Landing AI Vision Agent library
from vision_agent.agent import VisionAgentCoderV2
from vision_agent.models import AgentMessage, CodeContext

# Load environment variables
load_dotenv()


class CodeGenerator:
    def __init__(self, verbose: bool = False):
        """Initialize the code generator with Landing AI Vision Agent."""
        self.agent = VisionAgentCoderV2(verbose=verbose)
        self.verbose = verbose

    def load_document(self, file_path: str, page_num: int = 0):
        """
        Load a document file (PDF or image) and return the file path.
        For PDFs, extracts the specified page as an image and returns its path.

        Args:
            file_path: Path to the document file
            page_num: Page number to load for PDF documents (0-indexed)

        Returns:
            str: Path to the image file
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

            # Create a temporary file for the extracted page
            temp_img_path = f"temp_page_{page_num}.png"
            pix.save(temp_img_path)

            return temp_img_path
        else:
            # For image files, return the original path
            return file_path

    def generate_code(
        self, file_path: str, prompt: str, page_num: int = 0
    ) -> CodeContext:
        """
        Generate code based on the document and prompt.

        Args:
            file_path: Path to the document/image file
            prompt: The instruction for what code to generate
            page_num: Page number to process for PDF documents (0-indexed)

        Returns:
            CodeContext object containing the generated code and test
        """
        if self.verbose:
            print(f"Processing file: {file_path}")
            print(f"Using prompt: {prompt}")
            if page_num > 0:
                print(f"Processing page: {page_num}")

        try:
            # Get the image path (may be a temporary file for PDFs)
            img_path = self.load_document(file_path, page_num)

            # Generate code using the vision agent
            code_context = self.agent.generate_code(
                [AgentMessage(role="user", content=prompt, media=[img_path])]
            )

            # Clean up temporary file if created
            if img_path != file_path and os.path.exists(img_path):
                os.remove(img_path)

            return code_context

        except Exception as e:
            print(f"Error generating code: {str(e)}")
            import traceback

            traceback.print_exc()
            raise

    def save_code(
        self, code_context: CodeContext, output_path: str = "generated_code.py"
    ) -> str:
        """
        Save the generated code to a file.

        Args:
            code_context: CodeContext object containing the generated code
            output_path: Path to save the generated code

        Returns:
            str: Path to the saved code file
        """
        with open(output_path, "w") as f:
            f.write(code_context.code + "\n\n" + code_context.test)

        return output_path

    def execute_code(self, code_context: CodeContext) -> Any:
        """
        Execute the generated code and return the results.

        Args:
            code_context: CodeContext object containing the generated code

        Returns:
            Any: Result of executing the code
        """
        # Create a temporary namespace for execution
        namespace = {}

        # Execute the code
        exec(code_context.code, namespace)

        # Return the main function result if it exists
        if "main" in namespace and callable(namespace["main"]):
            return namespace["main"]()

        return None


def main():
    parser = argparse.ArgumentParser(
        description="Generate code from documents using Landing AI Vision Agent"
    )
    parser.add_argument("file_path", help="Path to the document or image file")
    parser.add_argument(
        "--prompt",
        help="Instruction for what code to generate",
        default="Analyze this document and generate code to extract its key information.",
    )
    parser.add_argument(
        "--output", help="Path to save the generated code", default="generated_code.py"
    )
    parser.add_argument(
        "--execute",
        help="Execute the generated code",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--page",
        type=int,
        help="Page number to process for multi-page documents (0-indexed)",
        default=0,
    )
    parser.add_argument(
        "--verbose", help="Enable verbose output", action="store_true", default=True
    )

    args = parser.parse_args()

    generator = CodeGenerator(verbose=args.verbose)

    try:
        print(f"Processing document: {args.file_path}")
        print(f"Using prompt: {args.prompt}")
        if args.page > 0:
            print(f"Processing page: {args.page}")

        # Generate code
        code_context = generator.generate_code(args.file_path, args.prompt, args.page)

        # Save the generated code
        output_path = generator.save_code(code_context, args.output)
        print(f"Generated code saved to: {output_path}")

        # Print a summary of the generated code
        code_lines = code_context.code.split("\n")
        print("\nGenerated Code Summary:")
        print(f"- {len(code_lines)} lines of code")
        print(
            f"- Main functionality: {code_lines[0] if code_lines else 'No code generated'}"
        )

        # Execute the code if requested
        if args.execute:
            print("\nExecuting generated code...")
            result = generator.execute_code(code_context)
            print("\nExecution Result:")
            print(result)

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
