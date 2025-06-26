from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
import re
import os

app = FastAPI(title="Question Evaluation API")

VLLM_URL = os.getenv("VLLM_URL", "http://localhost:8000")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3-30B-A3B")

# Set OpenAI's API key and API base to use vLLM's API server
openai_api_key = "EMPTY"
openai_api_base = f"{VLLM_URL}/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

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
        system_prompt = """You are an exceptional financial analyst competing in Mr. Beast's ultimate challenge.
Carefully read the following question, which may be in Chinese or English, and respond with only one word. Your answer must be one of the following six options:
- A
- B
- C
- D
- E
- Rise
- Fall

Instructions:
- If it's a multiple-choice question, answer with A, B, C, D or E.
- If it's a trend-related question (e.g., market or price movements), answer with Rise or Fall.
- Do not explain your answer.
- Do not use any other words or symbols.
- Apply your sharpest reasoning—you're competing for a prize of over 100 million RMB"""
        chat_response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]
        )
        
        # Extract the response content
        response_content = chat_response.choices[0].message.content.strip()
        
        # Extract just the letter (A, B, C, D, E, Fall or Rise) from the response
        answer_match = re.search(r'[ABCDE]|Fall|Rise', response_content)
        answer = answer_match.group() if answer_match else 'A'
        
        return QuestionResponse(
            answer=answer,
            raw_output=response_content
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.get("/")
async def health_check():
    return {"status": "API is running"}