import os
from fastapi import FastAPI
from fastapi.responses import JSONResponse

# Get the URL prefix from environment
url_prefix = os.getenv("URL_PREFIX", "")

# Initialize the FastAPI app
app = FastAPI()

# Mount routes under the prefix using an APIRouter
from fastapi import APIRouter
router = APIRouter()

@router.get("/")
async def root():
    return JSONResponse(status_code=200, content={"message": "OK"})

@router.post("/test")
async def test():
    return {"result": "passed"}

# Include router with URL prefix
app.include_router(router, prefix=url_prefix)
