from openai import OpenAI
# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://172.16.30.130:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

chat_response = client.chat.completions.create(
    model="Qwen/Qwen3-30B-A3B",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": """Answer the question with the appropriate options A, B, C and D. Please respond with the exact answer A, B, C or D only. Do not be verbose or provide extra information. 
Question: Who of these is the entrepreneur?
Answer Choices: A: Barack Obama, B: James Dyson, C: Damien Hirst, D: Mo Farah 
Answer:"""},
    ]
)
print("Chat response:", chat_response.choices[0].message.content.strip())