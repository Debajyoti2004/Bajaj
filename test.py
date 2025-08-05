import os
import sys
from rich import print as rprint
from rich.panel import Panel
from fastapi import BackgroundTasks

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from query_service import QueryService

class MockBackgroundTasks(BackgroundTasks):
    def __init__(self):
        self.tasks_to_run = []
        super().__init__()

    def add_task(self, func, *args, **kwargs) -> None:
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
        title="[bold green]Starting Batch Query Service Test[/bold green]",
        border_style="green"
    ))

    query_service = QueryService()
    mock_background_tasks = MockBackgroundTasks()

    try:
        results = query_service.process_queries(
            document_url=TEST_DOCUMENT_URL,
            questions=TEST_QUESTIONS,
            background_tasks=mock_background_tasks
        )
    except Exception as e:
        rprint(Panel(f"[bold red]An error occurred during processing:[/bold red]\n{e}", title="[red]Test Failed[/red]"))
        return

    rprint(Panel("[bold green]Processing Complete. Displaying Results...[/bold green]"))

    for i, res in enumerate(results):
        question_panel = Panel(
            f"[bold]Answer:[/bold] {res.answer}",
            title=f"[bold magenta]Result for Question #{i+1}[/bold magenta]: {TEST_QUESTIONS[i]}",
            border_style="magenta",
            expand=True
        )
        rprint(question_panel)

    mock_background_tasks.run_all()


if __name__ == "__main__":
    run_test()