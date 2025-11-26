import argparse
from fastapi import FastAPI, Request
import httpx
import uvicorn
from semantic_cache import SemanticCache
from transformers import AutoTokenizer

app = FastAPI()

def clip_text(text: str) -> str:
    """Clip text to MAX_TOKENS_INPUT using the tokenizer."""
    tokenized = tokenizer(text, truncation=True, max_length=MAX_TOKENS_INPUT)
    return tokenizer.decode(tokenized["input_ids"], skip_special_tokens=True)


@app.post("/v1/completions")
async def proxy_completion(request: Request):
    payload = await request.json()
    prompt_text = payload.get("prompt")
    if prompt_text is None:
        return {"error": "Missing 'prompt' field in request"}

    # Clip prompt
    clipped_prompt = clip_text(prompt_text)
    payload["prompt"] = clipped_prompt

    # Check cache first
    cached_response = cache.get_from_cache(clipped_prompt)
    if cached_response is not None:
        return {"response": cached_response, "cached": True}

    # Forward to vLLM server
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(VLLM_URL, json={"prompt": clipped_prompt})
            response.raise_for_status()
        except httpx.RequestError as exc:
            return {"error": f"Request failed: {exc}"}
        except httpx.HTTPStatusError as exc:
            return {"error": f"vLLM returned HTTP {exc.response.status_code}"}

    result_json = response.json()

    # Cache the response
    cache.add_to_cache(clipped_prompt, result_json)

    return {"response": result_json, "cached": False}


@app.post("/v1/chat/completions")
async def proxy_chat_completion(request: Request):
    payload = await request.json()
    messages = payload.get("messages")
    if not messages:
        return {"error": "Missing 'messages' field in request"}

    # Clip each message content individually
    clipped_messages = []
    for m in messages:
        content = m.get("content", "")
        clipped_content = clip_text(content)
        clipped_messages.append({**m, "content": clipped_content})

    payload["messages"] = clipped_messages

    # Use concatenated clipped content for cache key
    chat_text = " ".join([m["content"] for m in clipped_messages])

    # Check cache first
    cached_response = cache.get_from_cache(chat_text)
    if cached_response is not None:
        return {"response": cached_response, "cached": True}

    # Forward to vLLM server
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(VLLM_CHAT_URL, json={"messages": clipped_messages})
            response.raise_for_status()
        except httpx.RequestError as exc:
            return {"error": f"Request failed: {exc}"}
        except httpx.HTTPStatusError as exc:
            return {"error": f"vLLM returned HTTP {exc.response.status_code}"}

    result_json = response.json()

    # Cache the response
    cache.add_to_cache(chat_text, result_json)

    return {"response": result_json, "cached": False}


if __name__ == "__main__":
    # =======================
    # Command-line arguments
    # =======================
    parser = argparse.ArgumentParser(description="Run vLLM proxy server with caching and input clipping")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--vllm-url", type=str, default="http://localhost:8001/v1/completions")
    parser.add_argument("--vllm-chat-url", type=str, default="http://localhost:8001/v1/chat/completions")
    parser.add_argument("--max-tokens-input", type=int, default=512)
    parser.add_argument("--cache-model", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--cache-sim-thresh", type=float, default=0.95)
    parser.add_argument("--cache-max-size", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    # vLLM server URLs
    VLLM_URL = args.vllm_url
    VLLM_CHAT_URL = args.vllm_chat_url

    # Initialize semantic cache
    cache = SemanticCache(model_name=args.cache_model,
                        similarity_threshold=args.cache_sim_thresh,
                        max_cache_size=args.cache_max_size,
                        device=args.device)

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-4-mini-instruct")
    MAX_TOKENS_INPUT = args.max_tokens_input

    # Start app
    uvicorn.run("server:app", host=args.host, port=args.port, reload=True)
