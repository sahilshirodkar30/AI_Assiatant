from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from middleware.exception_handler import catch_exception_middleware
from routes.ask_questions import router as ask_questions_router
from routes.upload_files import router as upload_files_router

app = FastAPI(title="Medical Assistant API",description="Medical Assistant Chatbot")

# cross-origin resource sharing setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.middleware("http")(catch_exception_middleware)

app.include_router(ask_questions_router)
app.include_router(upload_files_router)