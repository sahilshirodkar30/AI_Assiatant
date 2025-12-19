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
        logger.info("Received uploaded files")

        for file in files:
            if file.content_type not in ALLOWED_TYPES:
                return JSONResponse(
                    status_code=400,
                    content={"error": f"{file.filename} is not a PDF"}
                )

            contents = await file.read()
            size_mb = len(contents) / (1024 * 1024)

            if size_mb > MAX_FILE_SIZE_MB:
                return JSONResponse(
                    status_code=400,
                    content={"error": f"{file.filename} exceeds {MAX_FILE_SIZE_MB}MB"}
                )

            # Reset pointer so downstream code can read again
            file.file.seek(0)

        load_vectorstore(files)

        logger.info("Documents added to vectorstore")
        return {"message": "Files processed and vectorstore updated"}

    except Exception as e:
        logger.exception("Error during PDF upload")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
