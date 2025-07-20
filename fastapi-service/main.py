from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import re
import os
from inference import main

app = FastAPI(title="Question Evaluation API")

#VLLM_URL = os.getenv("VLLM_URL", "http://localhost:8000")
#MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3-30B-A3B")

class QuestionRequest(BaseModel):
    question: str

class QuestionResponse(BaseModel):
    answer: str
    raw_output: str

@app.post("/eval", response_model=QuestionResponse)
async def evaluate_question(request: QuestionRequest):
    try:
        # Create the user content with the dynamic question
        user_content = request.question
        answer, raw_output = main(user_content)
        
        return QuestionResponse(
            answer=answer,
            raw_output=raw_output
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.get("/")
async def health_check():
    return {"status": "API is running"}
