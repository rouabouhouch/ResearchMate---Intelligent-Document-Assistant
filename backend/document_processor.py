import os
import tempfile
from typing import Dict, Any

class DocumentProcessor:
    def __init__(self):
        pass
    
    def process(self, file_path: str) -> Dict[str, Any]:
        """Process a document file and extract content and metadata."""
        filename = os.path.basename(file_path)
        content = ""
        file_type = "unknown"
        
        try:
            if filename.lower().endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                file_type = "text"
            
            elif filename.lower().endswith('.pdf'):
                try:
                    import PyPDF2
                    with open(file_path, 'rb') as f:
                        pdf_reader = PyPDF2.PdfReader(f)
                        for page in pdf_reader.pages:
                            text = page.extract_text()
                            if text:
                                content += text + "\n"
                    file_type = "pdf"
                except ImportError:
                    content = f"[PDF file: {filename} - install PyPDF2 for text extraction]"
                    file_type = "pdf"
                except Exception as pdf_error:
                    content = f"[PDF file: {filename} - error: {str(pdf_error)}]"
                    file_type = "pdf"
            
            else:
                # Try reading as text; fallback to truncated content if binary
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read(5000)
                file_type = "other"
        
        except Exception as e:
            content = f"[Error reading file: {filename} - {str(e)}]"
            file_type = "error"
        
        return {
            "type": file_type,
            "content": content,
            "metadata": {
                "filename": filename,
                "size": os.path.getsize(file_path) if os.path.exists(file_path) else 0
            }
        }
