# LLM Stress Test Tool (llama.cpp & Ollama)

A structured, async Python tool designed to stress-test local LLM servers (specifically `llama.cpp` and `Ollama`) equipped with NVIDIA GPUs.

It generates reproducible "noise" prompts to bypass the KV cache, forcing the GPU to re-compute attention for every request, simulating a worst-case load scenario.

## Features

- **High Concurrency**: Uses `asyncio` and `aiohttp` to simulate multiple users without blocking.
- **Cache Busting**: Generates deterministic but random "charabia" prompts to prevent KV cache hits.
- **Multi-Model Warfare**: Test one or multiple models simultaneously to observe resource contention.
- **Precise Metrics**:
    - **TTFT (Time To First Token)**: Reaction speed.
    - **End-to-End Latency**: Total generation time.
    - **Client-Side Throughput**: Effective tokens/sec received.
    - **Jitter (Stdev)**: Latency consistency measurement.
- **Anti-Hallucination Reporting**: Distinguishes "N/A" (no data) from "0.00" (actual zero).
- **Error Categorization**: Classifies errors as Timeout, HTTP 5xx, Connection Refused, or Other.
- **Reproducible**: Uses a fixed seed (default `42`) to ensure identical test runs for comparison.

## Requirements

- Python 3.10+
- A running instance of `llama.cpp` server or `Ollama`.

## Installation

```bash
# Clone the repository
git clone https://github.com/Syphiliis/llm-stress-tests.git
cd llm-stress-tests

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Edit `config/workload.yaml` to match your environment.

### For llama.cpp (Native)
```yaml
server:
  base_url: "http://localhost:8080/completion"
  model_alias: "my-model" # Optional

workload:
  users: 10               # Number of concurrent users
  duration_seconds: 60    # Test duration
  ramp_up_seconds: 10     # Time to reach full concurrency
  seed: 42                # Random seed for prompts
```

### For Ollama
If you are using Ollama, update the `base_url`:

```yaml
server:
  base_url: "http://localhost:11434/api/generate"
```

> **Note**: For Ollama, you might need to ensure the model is loaded before starting the test, as the first request often incurs model loading latency.

## Multi-Model Warfare

The tool supports testing multiple models simultaneously to observe GPU resource contention. This is useful when running multiple LLM instances on the same hardware.

### Scenario A: Flash Model Only
Heavy load on the faster, lighter model:

```bash
python main.py --config config/flash_only.yaml
```

### Scenario B: Thinker Model Only
Heavy load on the larger, more resource-intensive model:

```bash
python main.py --config config/thinker_only.yaml
```

### Scenario C: Dual Warfare (Mixed Load)
Concurrent load on both models (50%/50% split):

```bash
python main.py --config config/dual_warfare.yaml
```

### Configuration Example (Multi-Model)
```yaml
servers:
  - name: "Flash"
    base_url: "http://24.124.32.70:38703/completion"
    model_alias: "flash-model"
    weight: 0.5  # 50% of requests

  - name: "Thinker"
    base_url: "http://24.124.32.70:38704/completion"
    model_alias: "thinker-model"
    weight: 0.5  # 50% of requests

workload:
  users: 10
  duration_seconds: 300
  ramp_up_seconds: 60
  scenario: "dual_warfare"
```

## Remote Testing
For testing a remote GPU server (e.g., from a VPS), use the provided wrapper script:

```bash
./run_remote.sh
```

This script:
1.  Target configuration is in `config/remote_gpu.yaml`.
2.  Checks connectivity to the remote server (default `24.124.32.70:38703`).
3.  Runs the stress test using the remote configuration.

**Requirement**: The remote server must allow incoming traffic on port `38703` (mapped from Docker container port `8080`).

## Usage

Run the load tester:

```bash
python main.py --config config/workload.yaml
```

### Output

The tool will display a real-time progress log and a final summary:

```text
=== Test Summary ===
Total Requests: 150
Successful: 150
Failed: 0
Duration: 60.05s
RPS: 2.50
Global Throughput: 125.40 tokens/sec

=== Latency (s) ===
P50: 2.100
P90: 3.200
P99: 4.500
Mean: 2.450
Stdev (Jitter): 0.850

=== TTFT (s) ===
P50: 0.050
P90: 0.090
P99: 0.120
Mean: 0.065
Stdev (Jitter): 0.025
```

### Error Reporting

When errors occur, the tool provides a detailed breakdown:

```text
=== Error Breakdown ===
  timeout: 5 (50.0%)
  http_5xx: 3 (30.0%)
  connection_refused: 2 (20.0%)
```

### N/A Handling (Anti-Hallucination)

If a metric cannot be calculated (e.g., 100% failure rate), it displays `N/A` instead of misleading zeros:

```text
=== Latency (s) ===
P50: N/A
P90: N/A
P99: N/A

STATUS: FAILED (100% error rate)
```

### Per-Endpoint Breakdown (Multi-Model)

In mixed warfare mode, metrics are broken down by endpoint:

```text
=== Per-Endpoint Breakdown ===

  Flash:
    Total Requests: 75
    Successful: 73
    Failed: 2
    Latency P50/P90/P99: 1.234s / 2.100s / 3.500s
    Latency Stdev: 0.654s
    TTFT P50/P90/P99: 0.045s / 0.089s / 0.120s
    Total Tokens: 5840
    Errors: timeout: 2

  Thinker:
    Total Requests: 75
    Successful: 70
    Failed: 5
    Latency P50/P90/P99: 3.456s / 5.200s / 7.800s
    Latency Stdev: 1.234s
    TTFT P50/P90/P99: 0.120s / 0.250s / 0.400s
    Total Tokens: 4200
    Errors: timeout: 3, http_5xx: 2
```

Detailed JSON logs (including per-request metrics and error categories) are saved in the `results/` directory for analysis or correlation with Grafana/DCGM metrics.

## JSON Report

Each test run generates a detailed JSON report with:

- **Summary metrics**: P50/P90/P99, mean, stdev for latency and TTFT
- **Error breakdown**: Categorized error counts
- **Per-request details**: Individual timing, token counts, and error categories

```json
{
  "summary": {
    "total_requests": 150,
    "successful_requests": 145,
    "failed_requests": 5,
    "is_failed": false,
    "latency_p50": 2.1,
    "latency_stdev": 0.85,
    "error_breakdown": {
      "timeout": 3,
      "http_5xx": 2,
      "connection_refused": 0,
      "other": 0
    }
  },
  "details": [...]
}
```

## Prometheus Integration

Enable Prometheus metrics export by configuring the pushgateway:

```yaml
prometheus:
  enabled: true
  pushgateway_url: "localhost:9091"
  job_name: "llm_load_test"
  instance_name: "gpu_server_1"
  push_interval_seconds: 5
```

Exported metrics include:
- `llm_client_lag_p50/p90/p99_seconds`
- `llm_ttft_p50/p90/p99_seconds`
- `llm_tokens_per_second`
- `llm_request_rate_per_second`
- `llm_error_rate`
- `llm_contention_error_rate`
