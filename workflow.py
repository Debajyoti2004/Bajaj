import json
import re
from typing import TypedDict, List
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.output_parsers import StrOutputParser
from models import FinalAnswer
from config import ANSWER_LLM_MODEL, QUERY_LLM_MODEL, GOOGLE_API_KEY

class GraphState(TypedDict):
    original_questions: List[str]
    decomposed_queries: List[str]
    retriever: VectorStoreRetriever
    documents: List[Document]
    generation: List[str]

class RAGWorkflow:
    def __init__(self):
        self.base_generation_llm = ChatGoogleGenerativeAI(model=ANSWER_LLM_MODEL, api_key=GOOGLE_API_KEY, temperature=0)
        self.decomposition_llm = ChatGoogleGenerativeAI(model=QUERY_LLM_MODEL, api_key=GOOGLE_API_KEY, temperature=0)
        self.graph = self._build_graph()

    def _query_decomposition_node(self, state: GraphState):
        prompt = ChatPromptTemplate.from_template(
            """🔍 You are an intelligent query decomposition engine.
Given a list of user questions, generate 3 diverse search queries for each one.

⚠️ FORMAT:
Return a valid JSON list of lists:
[
  ["query1_a", "query1_b", "query1_c"],
  ["query2_a", "query2_b", "query2_c"],
  ...
]

QUESTIONS:
{questions}
"""
        )
        chain = prompt | self.decomposition_llm | StrOutputParser()
        joined_questions = "\n".join([f"{i+1}. {q}" for i, q in enumerate(state["original_questions"])])
        response_str = chain.invoke({"questions": joined_questions})

        try:
            clean_str = re.sub(r'^```json\s*|\s*```$', '', response_str, flags=re.MULTILINE).strip()
            parsed_lists = json.loads(clean_str)
        except json.JSONDecodeError:
            parsed_lists = [[q] for q in state["original_questions"]]

        all_queries = [q for sublist in parsed_lists for q in sublist] + state["original_questions"]
        return {"decomposed_queries": all_queries}

    def _retrieval_node(self, state: GraphState):
        retriever = state["retriever"]
        results = retriever.batch(state["decomposed_queries"])
        all_docs = [doc for result in results for doc in result]
        unique_docs = list({doc.page_content: doc for doc in all_docs}.values())
        return {"documents": unique_docs}

    def _generation_node(self, state: GraphState):
        context = "\n\n---\n\n".join([doc.page_content for doc in state["documents"]])

        structured_llm = self.base_generation_llm.with_structured_output(FinalAnswer)

        prompt = ChatPromptTemplate.from_template(
            """📚 CONTEXT:
{context}

🧠 TASK:
Answer each of the following questions based ONLY on the context above.
If a question cannot be answered, say "Not enough information".

QUESTIONS:
{questions}

⚠️ FORMAT:
Return a list of objects with the field `answer` only.
Example:
[{{"answer": "..."}}]
"""
        )

        joined_questions = "\n".join([f"{i+1}. {q}" for i, q in enumerate(state["original_questions"])])
        chain = prompt | structured_llm
        response = chain.invoke({
            "context": context,
            "questions": joined_questions,
        })

        answers = [item.answer for item in response] if isinstance(response, list) else [response.answer]
        return {"generation": answers}

    def _build_graph(self):
        workflow = StateGraph(GraphState)
        workflow.add_node("decompose_query", self._query_decomposition_node)
        workflow.add_node("retrieve", self._retrieval_node)
        workflow.add_node("generate", self._generation_node)
        workflow.set_entry_point("decompose_query")
        workflow.add_edge("decompose_query", "retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)
        return workflow.compile()

    def invoke_batch(self, questions: List[str], retriever: VectorStoreRetriever) -> List[str]:
        input_data = {"original_questions": questions, "retriever": retriever}
        result = self.graph.invoke(input_data)
        return result["generation"]
