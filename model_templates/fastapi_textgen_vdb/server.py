import csv
import inspect
import io
from typing import List, Optional, Annotated
import json
import time
import uuid
import faiss
from fastapi import FastAPI, File, UploadFile, Request, Depends
import uvicorn
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.utils.discovery import all_estimators
from models import ToolCallRequest, TextContent, ToolCallResult, ToolCallResponse
from contextlib import asynccontextmanager
from dataclasses import dataclass


INDEX_PATH = "sklearn_docs.index"

MODEL_DIR = "embedding_model"
model_name = "prajjwal1/bert-tiny"

class RetrieverWorker:
    def __init__(self, model_dir: str = "embedding_model", model_name: str = "prajjwal1/bert-tiny"):
        self.model_dir = model_dir
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=model_dir)
        self.model = AutoModel.from_pretrained(model_name, cache_dir=model_dir).cpu()
        self.index = faiss.read_index(INDEX_PATH)
        self.docstrings = []
        self.class_names = []
        estimators = all_estimators()
        for name, estimator in estimators:
            # Check if it's actually an estimator (has fit method)
            if hasattr(estimator, 'fit') and inspect.isclass(estimator):
                doc = estimator.__doc__
                if doc is not None and len(doc.strip()) > 0:
                    self.docstrings.append(doc)
                    self.class_names.append(name)

    def get_relevant_docs(self, query: str):
        encoded_input = self.tokenizer(
            [query], padding=True, truncation=True, max_length=256, return_tensors="pt"
        )
        device = torch.device("cpu")
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        token_embeddings = model_output[0]
        input_mask_expanded = (
            encoded_input["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        query_embedding = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )
        query_embedding = query_embedding.detach().cpu().numpy()

        k = 3  # Return top 3 results
        results = []
        distances, indices = self.index.search(query_embedding, k)
        print(f"\nTop {k} results for query: '{query}'")
        for i, (idx, distance) in enumerate(zip(indices[0], distances[0])):
            result = f"""{self.class_names[idx]} (Distance: {distance:.4f}) \n
            {self.docstrings[idx][:150]}"""
            results.append(result)
        return results

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/predict")
async def predict(request: Request, retriever: RetrieverWorker = Depends(RetrieverWorker)):
    content_type = request.headers.get("content-type", "")

    if "multipart/form-data" in content_type:
        # Handle multipart form-data with CSV file
        form = await request.form()
        file = form.get("file")
        if file and isinstance(file, UploadFile):
            content = await file.read()
            text = content.decode("utf-8")

    elif "text/plain" in content_type or "text/csv" in content_type:
        # Handle text/plain or text/csv input
        content = await request.body()
        text = content.decode("utf-8")
        if isinstance(text, str):
            text = [text]
    else:
        raise ValueError("Unsupported content type")

    if text:
        results_for_all_queries = []
        for query in text:
            results = retriever.get_relevant_docs(query)
            results_for_all_queries.append(results)
        return {"relevant": results_for_all_queries}
    else:
        raise ValueError("No text provided in the request")


@app.post("/predictUnstructured")
async def predict_unstructured(request: Request, retriever: RetrieverWorker = Depends(RetrieverWorker)):
    content_type = request.headers.get("content-type", "")

    if "multipart/form-data" in content_type:
        # Handle multipart form-data with CSV file
        form = await request.form()
        file = form.get("file")
        if file and isinstance(file, UploadFile):
            content = await file.read()
            text = content.decode("utf-8")

    elif "text/plain" in content_type or "text/csv" in content_type:
        # Handle text/plain or text/csv input
        content = await request.body()
        text = content.decode("utf-8")

    elif "application/json" in content_type:
        # Handle JSON input
        content = await request.body()
        try:
            data = json.loads(content)
            if isinstance(data, list):
                text = [
                    str(item) for item in data if item
                ]  # Convert all items to strings and filter empty items
            else:
                return ValueError("Invalid JSON format")
        except json.JSONDecodeError:
            return ValueError("Invalid JSON format")
    else:
        return ValueError("Unsupported content type")
    if len(text) != 1:
        raise ValueError("Only one text input is allowed")
    query = text[0]
    return retriever.get_relevant_docs(query)


@app.post("/chat/completions")
async def chat_completions(request: Request, retriever: RetrieverWorker = Depends(RetrieverWorker)):
    try:
        # Parse the request body
        body = await request.json()

        # Extract messages from the request
        messages = body.get("messages", [])

        # Find the latest user message
        latest_user_message = ""
        for message in reversed(messages):
            if message.get("role") == "user":
                latest_user_message = message.get("content", "")
                break

        # Generate a response based on the user's message
        # In a real implementation, this would call a language model
        response_content = "Here are the relevant documents I found: \n" + "\n".join(
            retriever.get_relevant_docs(latest_user_message)
        )

        # Generate unique ID with "chatcmpl-" prefix
        completion_id = f"chatcmpl-{str(uuid.uuid4())[:8]}"

        # Get current timestamp
        timestamp = int(time.time())

        # Create response with the specified structure
        response = {
            "id": completion_id,
            "object": "chat.completion",
            "created": timestamp,
            "model": "datarobot-retriever",
            "system_fingerprint": f"fp_{str(uuid.uuid4())[:10]}",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": response_content},
                    "logprobs": None,
                    "finish_reason": "stop",
                }
            ],
        }

        return response
    except Exception as e:
        return {"error": str(e)}


@app.post("/tools/call", response_model=ToolCallResponse)
async def tools_call(request: ToolCallRequest, retriever: RetrieverWorker = Depends(RetrieverWorker)):
    """
    Handle searches in an MCP tool call style. This
    route will be reachable at `directAccess/tools/call`.
    """
    try:
        request_id = request.id
        query = request.params.arguments.query

        text_response = "Here are the relevant documents I found: \n" + "\n".join(
            retriever.get_relevant_docs(query)
        )

        response = ToolCallResponse(
            id=request_id,
            result=ToolCallResult(content=[TextContent(text=text_response)], isError=False),
        )

        return response
    except Exception as e:
        # Return error response
        return ToolCallResponse(
            id=request.id if hasattr(request, "id") else 0,
            result=ToolCallResult(content=[TextContent(text=f"Error: {str(e)}")], isError=True),
        )


def process_csv_content(text):
    # Parse CSV content
    reader = csv.reader(io.StringIO(text))
    rows = list(reader)

    # Skip header row and extract strings
    if len(rows) > 0:
        data = rows[1:]  # Skip header row
        strings = [
            item for sublist in data for item in sublist if item
        ]  # Flatten and remove empty strings
        return strings

    return []




if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="trace")
