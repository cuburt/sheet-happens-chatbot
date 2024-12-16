import os
import uvicorn
from fastapi import FastAPI, APIRouter, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import logging
from pipeline import ModelPipeline
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from pathlib import Path
import os


app = FastAPI(debug=True)
origins = ["*"]
router = APIRouter()
model_pipeline_obj = ModelPipeline()


@router.post("/{model}/generate", tags=["sheet-happens"])
def generate(model: str, request: dict, session_id: Optional[str] = Query(None)):
    try:
        assert len(request.values()) != 0, "No entity found"
        is_success = True
        print(session_id)
        response_dict = model_pipeline_obj.generate_prediction(model, request['query'], request['enable_rag'],
                                                               session_id)
        error_message = ""
    except Exception as e:
        logging.error(str(e))
        response_dict = {}
        is_success = False
        error_message = str(e)

    data_res = {"success": is_success,
                "data": response_dict,
                "errors": error_message}

    return data_res


# Directory to save uploaded files
PROJECT_ROOT_DIR = str(Path(__file__).parent.parent)
UPLOAD_FOLDER = "data/uploads"
os.makedirs(os.path.join(PROJECT_ROOT_DIR, UPLOAD_FOLDER), exist_ok=True)

@router.post("/upload", tags=["sheet-happens"])
async def upload_files(files: list[UploadFile] = File(...)):
    uploaded_files = []

    for file in files:
        # Save each file to the upload folder
        file_path = os.path.join(PROJECT_ROOT_DIR, UPLOAD_FOLDER, file.filename)
        uploaded_files.append(file_path)
    model_pipeline_obj.upload_data(uploaded_files)
    return JSONResponse(
        {
            "message": f"{len(files)} files uploaded successfully!",
        }
    )


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=[],
)

app.include_router(
    router,
    prefix="/sheet-happens",
    tags=["sheet-happens"],
)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv('PORT', "8080")))
