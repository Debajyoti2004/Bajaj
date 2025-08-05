import os
import uuid
import requests
from typing import Tuple

class DocumentManager:
    DIR = 'doc_cache'

    def __init__(self, document_url: str):
        os.makedirs(self.DIR, exist_ok=True)
        self.document_url = document_url
        self.file_path, self.file_extension = self._download_document()

    def _download_document(self) -> Tuple[str, str]:
        response = requests.get(str(self.document_url), timeout=60)
        response.raise_for_status()
        
        content_type = response.headers.get('content-type', '')
        if 'pdf' in content_type: ext = '.pdf'
        elif 'vnd.openxmlformats-officedocument.wordprocessingml.document' in content_type: ext = '.docx'
        elif 'message/rfc822' in content_type: ext = '.eml'
        else:
            parsed_ext = os.path.splitext(str(self.document_url).split('?')[0])[1].lower()
            ext = parsed_ext if parsed_ext in ['.pdf', '.docx', '.eml', '.msg'] else '.tmp'
            
        filename = f"{uuid.uuid4()}{ext}"
        file_path = os.path.join(self.DIR, filename)

        with open(file_path, "wb") as f: f.write(response.content)
        return file_path, ext

    def get_filepath(self) -> str:
        return self.file_path

    def get_file_extension(self) -> str:
        return self.file_extension

    def cleanup(self):
        if self.file_path and os.path.exists(self.file_path):
            os.remove(self.file_path)