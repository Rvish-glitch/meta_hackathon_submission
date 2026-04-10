from __future__ import annotations
import json
import threading
from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import JSONResponse
import gradio as gr
from environment import OpenEnvEnvironment, Action, TaskName

_lock = threading.Lock()
_env: OpenEnvEnvironment | None = None

app = FastAPI(title="OpenEnv Benchmark")


@app.post("/reset")
async def handle_reset(request: Request):
    global _env
    try:
        body = json.loads(await request.body())
    except Exception:
        body = {}
    task_str = body.get("task", "email_triage")
    if task_str not in {t.value for t in TaskName}:
        task_str = "email_triage"
    with _lock:
        _env = OpenEnvEnvironment(task_str)
        obs = _env.reset()
    return JSONResponse({"observation": obs.content, "metadata": obs.metadata,
                         "task": task_str, "step": obs.step, "done": obs.done})


@app.post("/step")
async def handle_step(request: Request):
    global _env
    try:
        body = json.loads(await request.body())
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)
    with _lock:
        if _env is None:
            return JSONResponse({"error": "Call /reset first"}, status_code=400)
        try:
            result = _env.step(Action(task=_env.task_name, payload=body.get("action", {})))
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
    return JSONResponse({"observation": result.observation.content,
                         "reward": result.reward.value, "done": result.done,
                         "info": result.info, "step": result.observation.step})


@app.get("/health")
async def handle_health():
    return {"status": "ok"}


@app.get("/state")
async def handle_state():
    with _lock:
        return _env.state() if _env else {"error": "call /reset first"}


# Mount Gradio UI at root
with gr.Blocks(title="OpenEnv Benchmark") as demo:
    gr.Markdown("## OpenEnv Benchmark\nAPI: `POST /reset` · `POST /step` · `GET /health`")

app = gr.mount_gradio_app(app, demo, path="/")


def main():
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=7860)
