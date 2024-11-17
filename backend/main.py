from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import json

app = FastAPI()

origins = [
    "http://localhost:5173",
    "https://datathon.container.aed.cat",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def health_check():
    return {"status": "ok"}

@app.post("/upload/")
async def upload_file(file: UploadFile):
    path = "./response.json"
    with open(path, 'r') as file:
        response = json.load(file)
    return response
