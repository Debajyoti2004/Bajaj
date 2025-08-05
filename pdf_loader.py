import os
import re
import fitz
from langchain_core.documents import Document
from langchain_community.document_loaders import UnstructuredEmailLoader, Docx2txtLoader
from typing import List

class PDFLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def _is_table_line(self, line_text: str) -> bool:
        return bool(re.search(r"(\s{2,}|\t)", line_text)) and len(line_text.strip()) > 10

    def load(self) -> list[Document]:
        documents = []
        with fitz.open(self.file_path) as doc:
            for page_number, page in enumerate(doc):
                page_dict, text_lines, table_lines = page.get_text("dict"), [], []
                for block in page_dict.get("blocks", []):
                    if block["type"] == 0:
                        for line in block.get("lines", []):
                            line_text = " ".join([span["text"] for span in line.get("spans", [])]).strip()
                            if self._is_table_line(line_text): table_lines.append(line_text)
                            else: text_lines.append(line_text)
                
                full_text, table_text = "\n".join(text_lines).strip(), "\n".join(table_lines).strip()
                combined_text = ""
                if full_text: combined_text += f"### Text Content ###\n{full_text}\n"
                if table_text: combined_text += f"\n### Table Content ###\n{table_text}\n"
                
                if combined_text:
                    documents.append(Document(page_content=combined_text.strip(), metadata={"page": page_number + 1}))
        return documents

def loader_factory(file_path: str, file_extension: str):
    if file_extension == ".pdf": return PDFLoader(file_path)
    elif file_extension == ".docx": return Docx2txtLoader(file_path)
    elif file_extension in [".eml", ".msg"]: return UnstructuredEmailLoader(file_path, mode="single", process_attachments=False)
    else: raise ValueError(f"Unsupported file type for loading: {file_extension}")