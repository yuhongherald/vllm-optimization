import time
import requests
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

# =======================
# Function to send a single request
# =======================
def send_request(prompt, idx, print_interval):
    payload = {
        "prompt": prompt,
        "stream": False
    }
    start = time.time()
    try:
        r = requests.post(SERVER_URL, json=payload, timeout=60)
        latency = time.time() - start

        output_text = ""
        if r.status_code == 200:
            # Extract first choice text if available
            resp_json = r.json()
            choices = resp_json.get("response", {}).get("choices", [])
            if choices and isinstance(choices, list):
                output_text = choices[0].get("text", "")
        else:
            output_text = "None"

        if idx % print_interval == 0:
            print(
                f"[{idx}] Input: {prompt[:100]} | Output: {str(output_text)[:100]}")

        return latency
    except Exception as e:
        print(f"[{idx}] Exception: {e}")
        return None

# =======================
# Main script
# =======================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stress test FastAPI server")
    parser.add_argument("--server-url", type=str, default="http://localhost:8000/v1/completions")
    parser.add_argument("--prompts-file", type=str, default="data/test_dataset.txt")
    parser.add_argument("--num-requests", type=int, default=500)
    parser.add_argument("--request-rate", type=float, default=50)
    parser.add_argument("--print-interval", type=int, default=100)
    parser.add_argument("--max-workers", type=int, default=16)

    args = parser.parse_args()

    SERVER_URL = args.server_url
    PROMPTS_FILE = args.prompts_file
    NUM_REQUESTS = args.num_requests
    REQUEST_RATE = args.request_rate
    PRINT_INTERVAL = args.print_interval
    MAX_WORKERS = args.max_workers

    # =======================
    # Load prompts from text file
    # =======================
    print(f"Loading prompts from {PROMPTS_FILE}...")
    with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
        all_prompts = [line.strip() for line in f if line.strip()]

    prompts = all_prompts[:NUM_REQUESTS]
    print(f"Loaded {len(prompts)} prompts.")

    # =======================
    # Run concurrent requests at a controlled rate
    # =======================
    latencies = []
    start_time = time.time()
    print("Starting stress test:")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {}
        for i, prompt in enumerate(prompts):
            elapsed = time.time() - start_time
            expected_time = i / REQUEST_RATE
            if expected_time > elapsed:
                time.sleep(expected_time - elapsed)

            futures[executor.submit(send_request, prompt, i, PRINT_INTERVAL)] = i

        for future in as_completed(futures):
            latency = future.result()
            if latency:
                latencies.append(latency)

    end_time = time.time()
    total_time = end_time - start_time
    effective_rps = len(latencies) / total_time if total_time > 0 else 0

    # =======================
    # Print summary
    # =======================
    if latencies:
        latencies_np = np.array(latencies)

        print("\n=== Stress Test Summary ===")
        print(f"Total requests sent: {len(latencies)}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Effective RPS: {effective_rps:.2f} req/s")
        print(f"Average latency: {latencies_np.mean():.3f}s")
        print(f"Max latency: {latencies_np.max():.3f}s")
        print(f"Min latency: {latencies_np.min():.3f}s")
        print(f"P90 latency: {np.percentile(latencies_np, 90):.3f}s")
        print(f"P95 latency: {np.percentile(latencies_np, 95):.3f}s")
        print(f"P99 latency: {np.percentile(latencies_np, 99):.3f}s")
