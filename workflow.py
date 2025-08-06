from typing import TypedDict, List
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import VectorStoreRetriever
from models import *
from config import ANSWER_LLM_MODEL,QUERY_LLM_MODEL,GOOGLE_API_KEY

class GraphState(TypedDict):
    original_questions: List[Question]
    decomposed_questions: GeneratedQueries
    retriever: VectorStoreRetriever
    documents: List[List[Document]]
    answers: List[FinalAnswer]

class RAGWorkflow:
    def __init__(self):
        self.generation_llm = ChatGoogleGenerativeAI(model=ANSWER_LLM_MODEL, api_key=GOOGLE_API_KEY, temperature=0)
        self.decomposition_llm = ChatGoogleGenerativeAI(model=QUERY_LLM_MODEL, api_key=GOOGLE_API_KEY, temperature=0)
        self.graph = self._build_graph()

    def _query_decomposition_node(self, state: GraphState):
        prompt = ChatPromptTemplate.from_template(
            """You are an expert search query generator specializing in complex insurance policy documents.

            For each user question provided, create exactly 3 distinct, self-contained search queries. These queries are designed to be run against a vector database to find the most relevant text chunks.

            **Instructions for Query Generation:**
            1.  **Specificity is Key:** Use precise terminology found in insurance policies.
            2.  **Target All Facets:** Generate queries that cover different aspects of the question, especially:
                - The core topic (e.g., "maternity expenses").
                - Associated **conditions and eligibility criteria** (e.g., "maternity coverage waiting period").
                - Specific **limits, sub-limits, or exclusions** (e.g., "monetary limit for childbirth expenses").
            3.  **Self-Contained:** Each query must make sense on its own without relying on the original question's context.

            Return a Pydantic `GeneratedQueries` object with field `lst`, a list of length N. Each `lst[i]` is a `GeneratedQueriesForEachQuestion` containing exactly 3 queries for question i.

            USER QUESTIONS:
            {questions}"""
        )

        questions_str = "\n".join(f"{i+1}. {q.question}" for i, q in enumerate(state["original_questions"]))

        structured_llm = self.decomposition_llm.with_structured_output(GeneratedQueries)
        chain = prompt | structured_llm
        generated_lists: GeneratedQueries = chain.invoke({"questions": questions_str})  # type: ignore

        if not generated_lists or len(generated_lists.lst) != len(state["original_questions"]):
            lst = []
            for q in state["original_questions"]:
                lst.append(GeneratedQueriesForEachQuestion(queries=[q.question]))

            default = GeneratedQueries(lst=lst)
            return {"decomposed_questions": default}

        for i,el in enumerate(generated_lists.lst):
            el.queries.append(state["original_questions"][i].question)
        return {"decomposed_questions": generated_lists}

    def _retrieval_node(self, state: GraphState):
        queries = []
        for query_object in state["decomposed_questions"].lst:
            queries.extend(query_object.queries)
        """
        batch run rather than sequential invoke
        queries is a list of strings
        N initialy queries. Total 4 queries per original questions.
        4N queries in queries[].
        3 Chunks are returned.
        docs_list->4N size. ech element containing a list of 3 Documents.
        need to flatten every 4 nested lists together.
        documents: N length nested list od documents.
        """
        docs_lists = state["retriever"].batch(queries)
        per_q = len(state["decomposed_questions"].lst[0].queries)
        # flatten and dedupe by page content (or metadata)
        documents:List[List[Document]] = []
        for i in range(0,len(docs_lists), per_q):
            single_question_docs = [doc for docs in docs_lists[i:i+per_q] for doc in docs]
            unique = {doc.page_content: doc for doc in single_question_docs}
            documents.append(list(unique.values()))

        # self.pretty_print_documents_simple(documents)

        return {"documents": documents}

    def _generation_node(self, state: GraphState):
        contexts = [
            "\n\n---\n\n".join([doc.page_content for doc in docs])
            for docs in state["documents"]
        ]
        questions = [q.question for q in state["original_questions"]]
        N = len(questions)

        prompt = ChatPromptTemplate.from_template(
            """You are a meticulous and expert insurance policy analyst. Your task is to answer the user's QUESTION based *strictly and exclusively* on the provided CONTEXT from a policy document.

            **Instructions:**
            1.  **Comprehensive Analysis:** Carefully read the entire CONTEXT to find all relevant information. The answer is often spread across multiple sentences.
            2.  **Extract All Details:** Your answer MUST include all specific conditions, waiting periods, monetary limits, sub-limits, eligibility criteria, and quantitative details (like bed counts, percentages, or timeframes).
            3.  **Direct and Factual:** Begin with a direct answer to the question. Follow up with the detailed supporting information you extracted.
            4.  **No External Knowledge:** Do not use any information outside of the provided CONTEXT. If the context does not contain the answer, state that clearly.
            5.  **Be Factual, Not Conversational:** Do not add pleasantries. Stick to the facts from the policy document.

            CONTEXT:
            {context}

            QUESTION:
            {question}

            Based on your analysis, provide the complete answer in the required Pydantic `FinalAnswer` object format."""
        )
        structured_llm = self.generation_llm.with_structured_output(FinalAnswer)
        chain = prompt | structured_llm
        batch_inputs = [
            {"context": contexts[i], "question": questions[i]} for i in range(N)
        ]

        final_answers: List[FinalAnswer] = chain.batch(batch_inputs) # type: ignore
        return {"answers": final_answers}

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

    def invoke(self, questions: List[Question], retriever: VectorStoreRetriever)->List[FinalAnswer]:
        initial_state = {"original_questions": questions, "retriever": retriever}
        final_state = self.graph.invoke(initial_state) # type: ignore
        answer_objects:List[FinalAnswer] = final_state.get("answers") # type: ignore
        return answer_objects