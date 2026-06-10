# fastapi-vllm

A containerized **model-serving API** that puts a FastAPI service in front of a
[vLLM](https://github.com/vllm-project/vllm) backend, built as the serving layer
for an **LLM financial-reasoning evaluator**.

Built for the **SCBX Financial Analysis Agent Hackathon** (winning team) in
Thailand's Super AI Engineer program. Given a financial multiple-choice question,
the service runs it through the model and returns the chosen answer.

## Architecture

```
client ──HTTP──> FastAPI  (port 4000)  ──OpenAI API──> vLLM server  (port 8000)
                 /eval, /                                Qwen3-30B-A3B (reasoning)
```

`docker-compose` brings up two services:

| Service | What it does |
| --- | --- |
| `vllm-server` | `vllm/vllm-openai` serving **Qwen3-30B-A3B** on a GPU, with the DeepSeek-R1 reasoning parser enabled |
| `fastapi-app` | FastAPI wrapper exposing `POST /eval` (evaluate a question) and `GET /` (health check) |

## Run it

```bash
# starts both the vLLM server and the FastAPI app
docker compose up --build
```

Then call the API:

```bash
curl -X POST http://localhost:4000/eval \
  -H "Content-Type: application/json" \
  -d '{"question": "..."}'
```

`MODEL_NAME` is configurable via env (defaults to `Qwen/Qwen3-30B-A3B`).

## Load testing

`fastapi-service/locustfile.py` drives the endpoint under concurrent load with
[Locust](https://locust.io/), used to check throughput and latency of the serving
stack before submission.

## Files

| Path | What it is |
| --- | --- |
| `docker-compose.yml` | vLLM server + FastAPI app, GPU-enabled |
| `fastapi-service/main.py` | FastAPI app — `/eval` and health endpoints |
| `fastapi-service/inference.py` | Calls the vLLM OpenAI-compatible API |
| `fastapi-service/locustfile.py` | Locust load test |
| `api_example.py` | Minimal client example |
