from fastapi import APIRouter, Depends, Request, UploadFile, File, Form
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse
import os
import uuid

from exceptions.rag import ItemNotFoundError
from schemas.rag import QuestionRequest, QuestionResponse
from services.ragservice import RagService

router = APIRouter()

# Load templates
templates_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "templates"))

def load_template(filename: str) -> str:
    with open(os.path.join(templates_dir, filename), 'r') as f:
        return f.read()

home_template = load_template("home.html")
upload_template = load_template("upload.html")


async def item_not_found_handler(request: Request, exc: ItemNotFoundError):
    return JSONResponse(status_code=404, content={"detail": exc.message})


@router.get("/", response_class=HTMLResponse)
async def home_page(request: Request):
    return HTMLResponse(content=home_template)


@router.get("/rag", response_class=HTMLResponse)
async def home_page_alias(request: Request):
    return HTMLResponse(content=home_template)


@router.get("/upload", response_class=HTMLResponse)
async def upload_page(request: Request):
    return HTMLResponse(content=upload_template)

# python-multipart provides the multipart parser required by FastAPI for form uploads
import multipart  # type: ignore

@router.post("/upload")
async def upload_file(
    request: Request,
    file: UploadFile | None = File(None),
    files: list[UploadFile] | None = File(None),
    user_name: str = Form(...),
    user_id: str = Form(...),
):
    import os

    user_name = user_name.strip()
    user_id = user_id.strip()
    if not user_name:
        return JSONResponse(status_code=400, content={"detail": "user_name is required"})
    if not user_id:
        return JSONResponse(status_code=400, content={"detail": "user_id is required"})

    incoming_files: list[UploadFile] = list(files or [])
    if file is not None:
        incoming_files.append(file)
    if not incoming_files:
        return JSONResponse(status_code=400, content={"detail": "At least one PDF file is required"})
    if len(incoming_files) > 2:
        return JSONResponse(status_code=400, content={"detail": "You can upload up to 2 files at a time"})

    # Save uploaded PDF into the bundle's database/documents folder
    dest_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "database", "documents")
    )
    os.makedirs(dest_dir, exist_ok=True)

    session_id = request.headers.get("X-Session-ID")
    if session_id is None:
        session_id = str(uuid.uuid4())

    # Load only the newly uploaded PDF into the vector store and mark it with upload_id/user metadata.
    from database.vectorstore import load_documents_from_pdf_paths, vector_store, set_latest_upload_id

    upload_id = str(uuid.uuid4())
    uploaded_filenames: list[str] = []
    for uploaded_file in incoming_files:
        dest_path = os.path.join(dest_dir, uploaded_file.filename)
        with open(dest_path, "wb") as f:
            content = await uploaded_file.read()
            f.write(content)

        docs = load_documents_from_pdf_paths(
            [dest_path],
            upload_id=upload_id,
            user_id=user_id,
            user_name=user_name,
        )
        vector_store.add_documents(docs)
        uploaded_filenames.append(uploaded_file.filename)
    set_latest_upload_id(upload_id)

    service = RagService()
    service.record_upload(
        session_id,
        upload_id,
        uploaded_filenames,
        user_id=user_id,
        user_name=user_name,
    )
    service.set_session_user_context(session_id, user_id=user_id, user_name=user_name)

    uploaded_files = service.get_session_files(session_id)

    accept = request.headers.get("accept", "")
    if "text/html" in accept:
        return RedirectResponse(url="/", status_code=303)

    return {
        "filename": uploaded_filenames[0] if len(uploaded_filenames) == 1 else uploaded_filenames,
        "upload_id": upload_id,
        "session_id": session_id,
        "uploaded_files": uploaded_files,
        "user_name": user_name,
        "user_id": user_id,
        "available_users": service.get_session_user_options(session_id),
        "reloaded": True,
    }


@router.post("/rag/upload")
async def upload_file_alias(
    request: Request,
    file: UploadFile | None = File(None),
    files: list[UploadFile] | None = File(None),
    user_name: str = Form(...),
    user_id: str = Form(...),
):
    return await upload_file(request, file, files, user_name=user_name, user_id=user_id)


# def chat_alias(
#     request: QuestionRequest,
#     service: RagService = Depends(RagService),
# ):
#     return service.ask(request.question, metadata=request.metadata)


# @router.get("/items/{item_id}")
# async def read_item(item_id: int, service: RagService = Depends(RagService)):
#     return service.get_item(item_id)
    

@router.post("/rag/chat", response_model=QuestionResponse)
@router.post("/chat", response_model=QuestionResponse)
def chat(
    request: QuestionRequest,
    service: RagService = Depends(RagService),
):
    return service.ask(
        request.question,
        metadata=request.metadata,
        upload_id=request.upload_id,
        session_id=request.session_id,
        user_id=request.user_id,
        user_name=request.user_name,
    )
