from typing import List
from fastapi import BackgroundTasks
from models import FinalAnswer
from document_manager import DocumentManager
from retriever import VectorStoreProvider
from workflow import RAGWorkflow

class QueryService:
    def __init__(self):
        self.rag_workflow = RAGWorkflow()

    def process_queries(
        self,
        document_url: str,
        questions: List[str],
        background_tasks: BackgroundTasks
    ) -> List[str]:
        
        manager = DocumentManager(document_url)
        retriever = VectorStoreProvider(manager).get_retriever()
        
        results = self.rag_workflow.invoke_batch(questions, retriever)
        background_tasks.add_task(manager.cleanup)
        return results
