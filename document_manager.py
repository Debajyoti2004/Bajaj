import os
import uuid
import requests
from typing import Tuple
import shelve

class DocumentManager:
    DIR = 'doc_cache'

    def __init__(self, document_url: str):
        self.document_url = document_url
        self.file_path, self.filename = self._download_document()
        self.DIR = 'doc_cache'
        os.makedirs(self.DIR, exist_ok=True)
        self.document_url = document_url
        self.file_path, self.file_extension = self._download_document()


    def _download_document(self) -> Tuple[str,str]:
        response = requests.get(str(self.document_url), timeout=60)
        response.raise_for_status()
        filename = uuid.uuid4()
        file_path = os.path.join(self.DIR, f"{filename}.pdf")
        with open(file_path, "wb") as f:
            f.write(response.content)
        return file_path,'f{filename}' 

    def get_filepath(self) -> str:
        return self.file_path
    
    def get_filename(self) ->str:
        return self.filename

    def cleanup(self):
        if getattr(self, 'file_path', None) and os.path.exists(self.file_path):
            os.remove(self.file_path)
