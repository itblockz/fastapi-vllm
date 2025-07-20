# ================== Imports ==================
import re
import torch
import numpy as np
from helper import *
import os
from openai import OpenAI

# ================== Constants ==================
VLLM_URL = os.getenv("VLLM_URL", "http://localhost:8000")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3-30B-A3B")

# ================== Model Setup ==================
# Set OpenAI's API key and API base to use vLLM's API server
openai_api_key = "EMPTY"
openai_api_base = f"{VLLM_URL}/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

# ================== Multiple Choice Logic ==================
def multiple_choice(text, max_retry=3):
    # --- Judge Prompt ---
    def get_judge_prompt():
        return """You are a strict but fair evaluator in a financial analysis competition.

    Your task is to determine if the given answer is correct.

    Instructions:
    1. Analyze whether the answer is correct based on the context or financial logic.
    2. Provide your reasoning inside <WHY> ... </WHY> — **do not repeat the question**.
    3. At the end, output <FINAL>TRUE</FINAL> if the answer is correct, or <FINAL>FALSE</FINAL> if it is incorrect.

    Guidelines:
    - If it's a multiple-choice question (A, B, C, D), judge the logic of the selected option based on known knowledge or hints.
    - If it's a trend question (e.g., Rise/Fall), use financial reasoning or context cues to decide if the prediction is likely accurate.
    - If the answer seems invalid, irrelevant, or lacks justification, mark it as incorrect.
    - Your explanation should be clear enough to guide model correction if needed.

    Format:

    <WHY>
    {Explain here why the answer is correct or incorrect without repeating the question}
    </WHY>
    <FINAL>{TRUE or FALSE}</FINAL>
    """

    # --- Estimate Token Count ---
    def estimate_tokens(text):
        latin = all(ord(c) < 128 for c in text)
        return int(len(text.split()) * (1.3 if latin else 2.5))

    # --- Prompt for the answering model ---
    system_prompt = """You are an exceptional financial analyst competing in Mr. Beast's ultimate challenge.
    Carefully read the following question, which may be in Chinese or English, and respond with only one word. Your answer must be one of the following six options:
    - A
    - B
    - C
    - D
    - Rise
    - Fall

    Instructions:
    - If it's a multiple-choice question, answer with A, B, C, or D.
    - If it's a trend-related question (e.g., market or price movements), answer with Rise or Fall.
    - Do not explain your answer.
    - Do not use any other words or symbols.
    - Apply your sharpest reasoning—you're competing for a prize of over 100 million RMB"""

    question = text.strip()
    answers = []
    raw_output = ""

    # --- Retry Loop ---
    for attempt in range(max_retry):
        output = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ]
        )
        text_output = output[0].outputs[0].text.strip()
        raw_output += text_output

        # Extract answer
        match = re.search(r"\b(A|B|C|D|Rise|Fall|เพิ่มขึ้น|ลง)\b", text_output)
        answer = match.group(1) if match else "?"

        # Normalize Thai
        if answer == "เพิ่มขึ้น": answer = "Rise"
        elif answer == "ลง": answer = "Fall"
        if answer == "?" and attempt > 0:
            answer = answers[-1] if answers else "?"

        # Judge answer
        judge_prompt = get_judge_prompt()
        judge_msg = [
            {"role": "system", "content": judge_prompt},
            {"role": "user", "content": f"<QUESTION>{question}</QUESTION><ANSWER>{answer}</ANSWER>"}
        ]
        token_est = estimate_tokens(answer)
        if token_est > 150:
            break  # skip judge if too long

        judge_out = client.chat.completions.create(
            model=MODEL_NAME,
            messages=judge_msg
        )
        
        raw_output += judge_out[0].outputs[0].text
        
        is_correct = "<FINAL>TRUE</FINAL>" in judge_out
        why = re.search(r"<WHY>(.*?)</WHY>", judge_out, re.DOTALL)
        feedback = why.group(1).strip() if why else ""

        if is_correct:
            break
        question += f"\n\n#Feedback: {feedback}"
        answers.append(answer)
        
    # token = len(tokenizer.encode(raw_output, add_special_tokens=False))
    # print(token)
    #print(raw_output)
    return answer, raw_output

# ================== Rise/Fall Logic ==================
def model_generate(question):
    # --- System prompt for trend reasoning ---
    system_prompt = '''
        ###Role
        You are a financial time series expert. 

        ###Instructions
        The following is a summary of engineered features and exploratory data analysis from a stock price dataset. 
        Use this information to reason whether the price is likely to rise or fall in the next 5 days.

        ###Details
        - Rise is when the price increases after 5 days.
        - Fall is when the price decreases after 5 days.

        ###Output
        1. **Always** give short reasons why the price is likely to rise or fall.
        2. Use concise language.
        3. Answer with a single choice: "Rise" or "Fall". **ONLY use these two options**
        4. Your final answer must be wrapped **exactly** in: <answer>YourAnswer</answer> **THIS IS VERY IMPORTANT**
        '''
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question + " Let's think step by step and answer the question."},
    ]
    
    outputs = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages
    )
    raw_output = outputs[0].outputs[0].text
    match = re.search(r'<answer>(.*?)</answer>', outputs[0].outputs[0].text)
    if match:
        return match.group(1), raw_output
    else:
        # print(np.nan)
        return np.nan, raw_output

def risefall(question):
    # --- Preprocess question using custom stats ---
    prompt = cal_stat(question) # type: ignore
    output, raw_output = model_generate(prompt)
    return output, raw_output

# ================== Question Type Classifier ==================
def count_digits_in_string(input_string):
    digit_count = sum(char.isdigit() for char in input_string)
    return digit_count

def is_rise_fall_question(input_string):
    amount_digit = count_digits_in_string(input_string)
    if amount_digit > 200:
        return 'Rise/Fall'
    return 'Multiple'

# ================== Main Router ==================
def main(question):
    if is_rise_fall_question(question) == 'Rise/Fall':
        answer, raw_output = risefall(question)
        return answer, raw_output
    else:
        answer, raw_output = multiple_choice(question)
        return answer, raw_output
