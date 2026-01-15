# Prompt for Claude Sonnet / Opus

**Role:**
You are a Senior Python Expert and Quality Assurance (QA) Specialist, focusing on performance testing and Large Language Models (LLMs).

**Context:**
I am developing a "stress-test" tool for LLM servers (specifically `llama.cpp` and `Ollama`) using Python (`asyncio`/`aiohttp`).
The goal is to measure Latency, TTFT (Time To First Token), and Throughput (Tokens/sec) under load.
The infrastructure consists of a remote GPU server running Docker containers.

**Infrastructure Details (from `docker ps`):**
We have two models running simultaneously on different ports:
1.  **Model A (e.g., "Flash" model)**: `llama-server` on container port 8080 mapped to host port `38703`.
2.  **Model B (e.g., "Thinker" model)**: `llama-server-2` on container port 8081 mapped to host port `38704`.

**Objectives:**
I need a global improvement of this repository to support advanced testing scenarios and reliable reporting.

1.  **Multi-Model Warfare Support**:
    *   Update the code to handle testing **one** model (single endpoint) OR **two** models simultaneously (mixed warfare).
    *   We need to simulate realistic scenarios:
        *   Scenario A: Heavy load on "Flash" model only.
        *   Scenario B: Heavy load on "Thinker" model only.
        *   Scenario C: Concurrent load on BOTH (e.g., 50% traffic to Flash, 50% to Thinker).
    *   The configuration (YAML) should support defining multiple targets easily.

2.  **Anti-Hallucination Reporting**:
    *   The current reporting logic is too naive. If a test fails or has no data, it sometimes prints "0.00", which looks like "super fast".
    *   **Requirement**: Distinguish strictly between "0" (actual value) and "N/A" (no data).
    *   If a metric is unreliable (e.g., specific endpoint failed 100%), explicitly flag it (e.g., "N/A" or "FAILED").
    *   Add **Standard Deviation (stdev)** to measure Jitter.
    *   Categorize errors: `Timeout` vs `500 Internal Error` vs `Connection Refused`.

3.  **Code & robust updates**:
    *   Analyze `main.py`, `src/metrics/stats.py`, and `config/`.
    *   Make the code defensive: handle empty lists in numpy percentiles, avoid division by zero.
    *   Update the `README.md` to reflect these new capabilities (Flash/Thinker scenarios) and how to configure them.

**Instructions:**
*   You have full freedom to improve the code structure, add new files, or refactor existing ones.
*   Use any standard Python libraries you see fit.
*   Provide the improved file contents for: `main.py`, `src/metrics/stats.py` (or a better structure), and an updated `config/workload.yaml` example for the dual-model setup.
*   Provide the updated `README.md`.

**Attached Files:**
*   `main.py`
*   `src/metrics/stats.py`
*   `config/remote_gpu.yaml` (current simple config)

**Go!**
