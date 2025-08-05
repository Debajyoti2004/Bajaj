import os
import time
import traceback
from fastapi import FastAPI, HTTPException, Body, status, Depends, Request, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import List
from rich import print as rprint
from rich.panel import Panel
import requests

from models import QueryRequest, QueryResponse, FinalAnswer
from query_service import QueryService
from config import GOOGLE_API_KEY, API_AUTH_TOKEN

app = FastAPI(
    title="Batch-Optimized RAG System",
    description="Processes documents and answers a list of questions together with high efficiency.",
    version="11.0.0",
)
query_service = QueryService()
security = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.scheme != "Bearer" or credentials.credentials != API_AUTH_TOKEN:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API token")
    return True

@app.on_event("startup")
def on_startup():
    if not GOOGLE_API_KEY or not API_AUTH_TOKEN:
        raise RuntimeError("Missing critical environment variables: GOOGLE_API_KEY and API_AUTH_TOKEN")
    rprint(Panel("Application startup complete.", title="[green]System Status[/green]"))

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = f"{process_time:.4f} sec"
    rprint(f"[cyan]Request[/cyan] '{request.method} {request.url.path}' [bold green]completed in {process_time:.4f}s[/bold green]")
    return response

@app.post(
    "/hackrx/run",
    response_model=QueryResponse,
    tags=["Query Processing"],
    summary="Process a Document and Answer a Batch of Questions"
)
async def run_submission(
    request_body: QueryRequest,
    background_tasks: BackgroundTasks,
    authenticated: bool = Depends(verify_token)
):
    rprint(Panel(f"Processing request for document: [blue]{request_body.documents}[/blue]", title="[cyan]New Request[/cyan]"))
    try:
        results: List[str] = query_service.process_queries(
            document_url=str(request_body.documents),
            questions=request_body.questions,
            background_tasks=background_tasks
        )
        return QueryResponse(answers=results)
    except requests.exceptions.RequestException as e:
        rprint(Panel(f"[bold red]Document Download Failed:[/bold red]\n{e}", title="[red]Error[/red]"))
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Failed to download document: {e}")
    except ValueError as e:
        rprint(Panel(f"[bold red]Processing Error:[/bold red]\n{e}", title="[red]Error[/red]"))
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except Exception as e:
        tb_str = traceback.format_exc()
        rprint(Panel(f"[bold red]An unexpected server error occurred:[/bold red]\n{tb_str}", title="[red]Server Error[/red]"))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An internal server error occurred.")

@app.get("/health", tags=["Monitoring"], summary="API Health Check")
def health_check():
    return {"status": "ok"}