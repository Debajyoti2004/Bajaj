from rich import print as rprint
from rich.panel import Panel
from models import Question, FinalAnswer
from query_service import QueryService
from typing import List

def run_test():
    TEST_DOCUMENT_URL = "https://arxiv.org/pdf/1706.03762.pdf"
    questions = [
        "What is a Transformer and what are its components?",
        "What was the BLEU score for the big Transformer model on the English-to-German translation task shown in Table 2?",
        "How is the transformer model better from its predecessors?",
        "Which organization published the paper?"
    ]
    TEST_QUESTIONS = [Question(question=question) for question in questions]
    rprint(Panel(
        f"Document URL: [blue]{TEST_DOCUMENT_URL}[/blue]\n"
        f"Questions: [yellow]{len(TEST_QUESTIONS)}[/yellow]",
        title="[bold green]Starting Multi-Query Service Test[/bold green]",
        border_style="green"
    ))

    query_service = QueryService()

    results:List[FinalAnswer] = query_service.process_queries(
        document_url=TEST_DOCUMENT_URL,
        questions=TEST_QUESTIONS
    )
    rprint(Panel("[bold green]Processing Complete. Displaying Results...[/bold green]"))

    for i, (response) in enumerate(results):

        question_panel = Panel(
            f"[bold]Answer:[/bold] {response.answer}\n\n",
            title=f"[bold magenta]Result for Question #{i+1}[/bold magenta]: {TEST_QUESTIONS[i]}",
            border_style="magenta",
            expand=True
        )

        rprint(question_panel)

if __name__ == "__main__":
    run_test()