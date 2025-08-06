from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores.base import VectorStoreRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from document_manager import DocumentManager
from config import EMBEDDING_MODEL

class VectorStoreProvider:
    def __init__(self, manager: DocumentManager):
        self.manager = manager
        self.retriever = self._create_retriever()

    def _create_retriever(self) -> VectorStoreRetriever:
        file_path = self.manager.get_filepath()
        loader = PyMuPDFLoader(file_path, mode="single")
        raw_documents = loader.load()
        if not raw_documents:
            raise ValueError(f"Couldn’t load any content from {file_path}")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1700, chunk_overlap=300)
        split_docs = text_splitter.split_documents(raw_documents)
        for doc in split_docs:
            doc.metadata["source"] = self.manager.document_url

        embedding_model = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
        db = Chroma(
            collection_name="pdf_docs",
            embedding_function=embedding_model,
            persist_directory="./vector_db",
        )

        # check if already stored ANY chunk for this URL
        existing = db.get(
            ids=None,
            include=["metadatas"],
            where={"source": self.manager.document_url}
        )["metadatas"]

        if not existing:
            print("Creating new embeddings.")
            db.add_documents(split_docs)
        else:
            print("Embeddings already exist")

        retriever = db.as_retriever(
            search_kwargs={"k": 5, "filter": {"source": self.manager.document_url}}
        )
        return retriever