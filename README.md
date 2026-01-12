# LLM Stress Test Tool (llama.cpp & Ollama)

A structured, async Python tool designed to stress-test local LLM servers (specifically `llama.cpp` and `Ollama`) equipped with NVIDIA GPUs.

It generates reproducible "noise" prompts to bypass the KV cache, forcing the GPU to re-compute attention for every request, simulating a worst-case load scenario.

## Features

- **High Concurrency**: Uses `asyncio` and `aiohttp` to simulate multiple users without blocking.
- **Cache Busting**: Generates deterministic but random "charabia" prompts to prevent KV cache hits.
- **Precise Metrics**:
    - **TTFT (Time To First Token)**: Reaction speed.
    - **End-to-End Latency**: Total generation time.
    - **Client-Side Throughput**: Effective tokens/sec received.
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
P99: 4.500

=== TTFT (s) ===
P50: 0.050
P90: 0.120
```

Detailed JSON logs (including per-request metrics) are saved in the `results/` directory for analysis or correlation with Grafana/DCGM metrics.
