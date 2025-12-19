import os
import shutil
from fastapi import UploadFile
from typing import List

UPLOAD_DIR = "./uploaded_docs"

def save_uploaded_files(files: List[UploadFile]) -> List[str]:
    """
    Save uploaded PDF files to disk and return file paths.
    """
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    saved_paths: List[str] = []

    for file in files:
        file_path = os.path.join(UPLOAD_DIR, file.filename)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        saved_paths.append(file_path)

    return saved_paths
