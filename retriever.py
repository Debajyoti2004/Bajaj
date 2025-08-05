from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from document_manager import DocumentManager
from pdf_loader import loader_factory
from config import EMBEDDING_MODEL

class VectorStoreProvider:
    def __init__(self, manager: DocumentManager):
        self.manager = manager
        self.retriever = self._create_retriever()

    def _create_retriever(self) -> VectorStoreRetriever:
        loader = loader_factory(self.manager.get_filepath(), self.manager.get_file_extension())
        raw_documents = loader.load()
        if not raw_documents: raise ValueError("Could not load any content from the document.")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        split_docs = text_splitter.split_documents(raw_documents)
        embedding_model = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
        db = Chroma.from_documents(split_docs, embedding=embedding_model)
        return db.as_retriever(search_kwargs={"k": 5})

    def get_retriever(self) -> VectorStoreRetriever:
        return self.retriever
