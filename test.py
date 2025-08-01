import os
import sys
from rich import print as rprint
from rich.panel import Panel
from rich.text import Text
from fastapi import BackgroundTasks

from query_service import QueryService

class MockBackgroundTasks(BackgroundTasks):
    def __init__(self):
        self.tasks_to_run = []

    def add_task(self, func, *args, **kwargs):
        self.tasks_to_run.append(lambda: func(*args, **kwargs))

    def run_all(self):
        for task in self.tasks_to_run:
            task()
        rprint(Panel("[green]Mock background task (document cleanup) executed successfully.[/green]", title="[bold cyan]Cleanup[/bold cyan]"))


def run_test():
    TEST_DOCUMENT_URL = "https://arxiv.org/pdf/1706.03762.pdf"
    TEST_QUESTIONS = [
        "What is a Transformer and what are its components?",
        "What was the BLEU score for the big Transformer model on the English-to-German translation task shown in Table 2?",
        "What is the capital of France?"
    ]

    rprint(Panel(
        f"Document URL: [blue]{TEST_DOCUMENT_URL}[/blue]\n"
        f"Questions: [yellow]{len(TEST_QUESTIONS)}[/yellow]",
        title="[bold green]Starting Multi-Query Service Test[/bold green]",
        border_style="green"
    ))

    query_service = QueryService()
    mock_background_tasks = MockBackgroundTasks()

    try:
        results_with_queries = query_service.process_queries(
            document_url=TEST_DOCUMENT_URL,
            questions=TEST_QUESTIONS,
            background_tasks=mock_background_tasks
        )
    except Exception as e:
        rprint(Panel(f"[bold red]An error occurred during processing:[/bold red]\n{e}", title="[red]Test Failed[/red]"))
        return

    rprint(Panel("[bold green]Processing Complete. Displaying Results...[/bold green]"))

    for i, (res, gen_queries) in enumerate(results_with_queries):
        queries_text = Text()
        for q in gen_queries:
            queries_text.append(f"  - {q}\n")

        question_panel = Panel(
            f"[bold]Answer:[/bold] {res.answer}\n\n"
            f"[bold]Rationale:[/bold] {res.rationale}\n\n"
            f"[bold]Source Page:[/bold] {res.source_page}",
            title=f"[bold magenta]Result for Question #{i+1}[/bold magenta]: {TEST_QUESTIONS[i]}",
            border_style="magenta",
            expand=True
        )

        generated_queries_panel = Panel(
            queries_text,
            title="[bold yellow]Decomposed Queries Generated by LLM[/bold yellow]",
            border_style="yellow",
            expand=True
        )

        rprint(question_panel)
        rprint(generated_queries_panel)

    mock_background_tasks.run_all()


if __name__ == "__main__":
    run_test()