from fastapi import FastAPI, UploadFile

app = FastAPI()

@app.get("/")
async def health_check():
    return {"status": "ok"}


@app.post("/upload/")
async def upload_file(file: UploadFile):
    return {"filename": file.filename}


