from fastapi import APIRouter, UploadFile, File
from typing import List
from fastapi.responses import JSONResponse
from logger import logger
from modules.local_vectorstore import load_vectorstore

router = APIRouter()

MAX_FILE_SIZE_MB = 10
ALLOWED_TYPES = {"application/pdf"}

@router.post("/upload_pdfs")
async def upload_pdfs(files: List[UploadFile] = File(...)):
    try:
        for file in files:
            if file.content_type not in ALLOWED_TYPES:
                return JSONResponse(
                    status_code=400,
                    content={"error": f"{file.filename} is not a PDF"}
                )

            contents = await file.read()
            if len(contents) / (1024 * 1024) > MAX_FILE_SIZE_MB:
                return JSONResponse(
                    status_code=400,
                    content={"error": f"{file.filename} exceeds {MAX_FILE_SIZE_MB}MB"}
                )

            file.file.seek(0)

        load_vectorstore(files)
        return {"message": "Files processed successfully"}

    except Exception as e:
        logger.exception("Upload failed")
        return JSONResponse(status_code=500, content={"error": str(e)})
