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
        """You’re an expert insurance researcher specializing in Bajaj Finserv policies.

            Given a list of N user questions about health insurance (for example: grace period, waiting periods, maternity cover, etc.), generate for each question exactly 3 high-precision, self-contained search queries that:

            1. Use policy terminology (e.g. “National Parivar Mediclaim Plus grace period”).  
            2. Target distinct sub-topics or phrasing to maximize recall.  
            3. Are optimized for retrieving specific clauses or limits in a PDF.  

            Return a Pydantic `GeneratedQueries` object with field `lst`, a list of length N. Each `lst[i]` is a `GeneratedQueriesForEachQuestion` containing exactly 3 queries for question i.

            QUESTIONS:
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
        # batch run rather than sequential invoke
        # queries is a list of strings
        """
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
        """You’re an insurance expert answering queries *only* from provided excerpted policy text.

        Use the CONTEXT to locate exact policy clauses or numbers. Then give a **concise** answer (≤150 words) with:
        - A direct statement (“The grace period is 30 days.”)
        - A one-sentence rationale referencing the context (“As per Section 5.2, …”).

        Do **not** hallucinate or use external info—stick strictly to the text.

        CONTEXT:
        {context}

        QUESTION:
        {question}

        RESPONSE FORMAT:
        Return a Pydantic `FinalAnswer` object with a single field:
        - `answer`: Your fact-based, ≤150-word answer."""
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