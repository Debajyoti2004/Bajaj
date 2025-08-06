import time
import traceback
from fastapi import FastAPI, HTTPException, status, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import List
from rich import print as rprint
from rich.panel import Panel
from config import GOOGLE_API_KEY, API_AUTH_TOKEN
import requests

from models import QueryRequest, QueryResponse, FinalAnswer, Question
from query_service import QueryService

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Batch-Optimized RAG System",
    description="Processes documents and answers a list of questions together with high efficiency.",
    version="1.0.0",
)

query_service = QueryService()
security = HTTPBearer()

# --- Security Dependency ---
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verifies the bearer token provided in the Authorization header."""
    if credentials.scheme != "Bearer" or credentials.credentials != API_AUTH_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return True

# --- Application Events ---
@app.on_event("startup")
def on_startup():
    """Checks for necessary configurations on application startup."""
    if not GOOGLE_API_KEY:
        raise RuntimeError("Missing critical environment variable: GOOGLE_API_KEY")
    rprint(Panel("Application startup complete. API token loaded.", title="[green]System Status[/green]"))

# --- Middleware ---
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Adds a custom X-Process-Time header to all responses."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = f"{process_time:.4f} sec"
    rprint(f"[cyan]Request[/cyan] '{request.method} {request.url.path}' [bold green]completed in {process_time:.4f}s[/bold green]")
    return response

# --- API Endpoints ---
@app.post(
    "/api/v1/hackrx/run",
    response_model=QueryResponse,
    tags=["Query Processing"],
    summary="Process a Document and Answer a Batch of Questions",
    status_code=status.HTTP_200_OK
)
async def run_submission(
    request_body: QueryRequest,
    # The 'authenticated' dependency runs 'verify_token' for every request to this endpoint
    authenticated: bool = Depends(verify_token)
):
    """
    This endpoint receives a document URL and a list of questions,
    processes them using the RAG pipeline, and returns a list of answers.
    """
    rprint(Panel(f"Processing request for document: [blue]{str(request_body.documents)}[/blue]", title="[cyan]New Request[/cyan]"))
    try:
        questions_as_models = [Question(question=q) for q in request_body.questions]

        # The core logic is executed by the QueryService
        results: List[FinalAnswer] = query_service.process_queries(
            document_url=str(request_body.documents),
            questions=questions_as_models,
        )

        final_answers = [result.answer for result in results]

        return QueryResponse(answers=final_answers)

    except requests.exceptions.RequestException as e:
        rprint(Panel(f"[bold red]Document Download Failed:[/bold red]\n{e}", title="[red]Error[/red]"))
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Failed to download document: {e}")
    except ValueError as e:
        rprint(Panel(f"[bold red]Processing Error:[/bold red]\n{e}", title="[red]Error[/red]"))
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except Exception as e:
        # Catch-all for any other unexpected errors
        tb_str = traceback.format_exc()
        rprint(Panel(f"[bold red]An unexpected server error occurred:[/bold red]\n{tb_str}", title="[red]Server Error[/red]"))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An internal server error occurred.")

@app.get("/health", tags=["Monitoring"], summary="API Health Check")
def health_check():
    """A simple endpoint to check if the API is running."""
    return {"status": "ok"}