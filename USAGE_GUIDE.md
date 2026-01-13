# LLM Load Testing Tool - Usage Guide

## Overview

This tool provides comprehensive load testing for LLM inference servers with support for:
- **Single Endpoint Testing**: Traditional load testing against one model
- **Mixed Warfare**: Simultaneous testing of multiple models to observe resource contention
- **Real-time Progress Monitoring**: Live console updates during test execution
- **Detailed Metrics**: TTFT, latency percentiles, throughput, error rates
- **Optional Prometheus Integration**: Push metrics to Prometheus Pushgateway (if needed)

## Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Single Endpoint Testing

Test a single llama.cpp server:

```bash
python main.py --config config/workload.yaml
```

**Default config (config/workload.yaml):**
```yaml
server:
  base_url: "http://127.0.0.1:8080/completion"
  model_alias: "llama-2-7b-chat"

workload:
  users: 20
  duration_seconds: 120
  ramp_up_seconds: 10
  seed: 42

prompts:
  min_tokens: 50
  max_tokens: 200
  prefix: "Please summarize the following text: "

prometheus:
  enabled: false  # Disable if you have Grafana connected to llama.cpp
```

### 3. Mixed Warfare Testing

Test two models simultaneously to observe GPU contention:

```bash
python main.py --config config/mixed_warfare.yaml
```

**Mixed Warfare config (config/mixed_warfare.yaml):**
```yaml
servers:
  - name: "Thinker"
    base_url: "http://127.0.0.1:8080/completion"
    model_alias: "llama-2-70b-chat"
    weight: 0.3  # 30% of requests

  - name: "Flash"
    base_url: "http://127.0.0.1:8081/completion"
    model_alias: "llama-2-7b-chat"
    weight: 0.7  # 70% of requests

workload:
  users: 20
  duration_seconds: 120
  ramp_up_seconds: 10
  seed: 42

prompts:
  min_tokens: 50
  max_tokens: 200
  prefix: "Please summarize the following text: "
```

## Understanding the Output

### Real-Time Progress Updates

Every 10 seconds during the test:
```
[Progress @ 30s] Active Users: 20 | Requests: 145 | RPS: 4.8 | TPS: 58.3 tok/s | P90 Latency: 2.45s | Errors: 0.0% | Time Left: 90s
```

### Final Test Summary

After test completion:
```
=== Test Summary ===
Total Requests: 580
Successful: 576
Failed: 4
Duration: 120.23s
RPS: 4.79
Global Throughput: 57.82 tokens/sec

=== Latency (s) ===
P50: 1.234
P90: 2.456
P99: 3.789

=== TTFT (s) ===
P50: 0.123
P90: 0.456
P99: 0.789

=== Automatic Analysis ===
PASS: Error rate is acceptable (0.7%)
WARNING: P90 Latency is high (2.46s). User experience degraded.
PASS: TTFT is excellent (0.12s).
```

### Per-Endpoint Breakdown (Mixed Warfare Only)

```
=== Per-Endpoint Breakdown ===

  Flash:
    Total Requests: 406
    Successful: 404
    Failed: 2
    Latency P50/P90/P99: 0.856s / 1.234s / 2.145s
    TTFT P50/P90/P99: 0.089s / 0.156s / 0.234s
    Total Tokens: 81200
    Throughput: 675.32 tokens/sec

  Thinker:
    Total Requests: 174
    Successful: 172
    Failed: 2
    Latency P50/P90/P99: 3.456s / 5.678s / 8.901s
    TTFT P50/P90/P99: 0.234s / 0.456s / 0.678s
    Total Tokens: 34400
    Throughput: 286.12 tokens/sec
```

## Advanced Features

### Configurable Workload Parameters

- **users**: Number of concurrent virtual users
- **duration_seconds**: Total test duration
- **ramp_up_seconds**: Time to gradually reach full concurrency (0 = immediate)
- **seed**: Deterministic random seed for reproducible tests

### Prompt Generation

- **min_tokens**: Minimum prompt length (approximate)
- **max_tokens**: Maximum generation length per request
- **prefix**: Optional prefix for all prompts

### Results Export

All test results are saved to `results/test_run_TIMESTAMP.json`:

```json
{
  "config": { ... },
  "summary": "...",
  "details": [
    {
      "request_id": "abc-123",
      "endpoint": "Flash",
      "start_ts": 1234567890.123,
      "end_ts": 1234567891.456,
      "ttft": 0.089,
      "latency": 1.333,
      "output_tokens": 200,
      "tps": 150.0,
      "error": null
    },
    ...
  ]
}
```

## Prometheus Integration (Optional)

If you need client-side metrics pushed to a separate Prometheus instance:

```yaml
prometheus:
  enabled: true
  pushgateway_url: "localhost:9091"
  job_name: "llm_load_test"
  instance_name: "local_test"
  push_interval_seconds: 5
  # Optional authentication
  # username: "admin"
  # password: "secret"
```

**Metrics pushed:**
- `llm_client_lag_p50/p90/p99_seconds`: Latency percentiles
- `llm_ttft_p50/p90/p99_seconds`: Time to first token percentiles
- `llm_tokens_per_second`: Throughput
- `llm_request_rate_per_second`: RPS
- `llm_error_rate`: Overall error rate
- `llm_contention_error_rate`: Timeout/503 errors indicating resource contention
- `llm_active_users`: Current concurrent users

## Interpreting Results for GPU Contention

When running Mixed Warfare tests to observe GPU contention:

1. **Compare TTFT between models**: Higher TTFT indicates the model is waiting for GPU resources
2. **Look for increased P90/P99 latency**: Tail latency degradation suggests resource competition
3. **Monitor contention_error_rate**: Timeouts and 503 errors indicate severe overload
4. **Compare throughput**: Check if combined TPS is less than sum of individual TPS

**Example Analysis:**
```
Flash (7B model):
  - TTFT P50: 0.089s (good)
  - Latency P90: 1.234s (acceptable)
  - Throughput: 675 tok/s

Thinker (70B model):
  - TTFT P50: 0.234s (slower, waiting for GPU)
  - Latency P90: 5.678s (high, resource contention)
  - Throughput: 286 tok/s (reduced)
```

This suggests "Thinker" is experiencing significant GPU contention from "Flash" requests.

## Best Practices

1. **Warm-up**: Run a short test first to warm up the server before collecting metrics
2. **Ramp-up**: Use `ramp_up_seconds` to avoid thundering herd at test start
3. **Realistic Workload**: Adjust `min_tokens`, `max_tokens`, and `users` to match your expected traffic
4. **Multiple Runs**: Run tests multiple times and average results for reliability
5. **Monitor Server-Side**: Use your existing Grafana dashboard to correlate client metrics with GPU utilization

## Troubleshooting

**Connection refused errors:**
- Verify llama.cpp server is running on the specified URL
- Check firewall settings

**High error rates:**
- Reduce number of concurrent users
- Increase server timeout settings
- Check server logs for issues

**Inconsistent results:**
- Use same `seed` value for reproducible tests
- Ensure server is not under external load
- Check for thermal throttling on GPU

## Architecture

```
main.py                         # Orchestrates the test
├── src/client/
│   └── api_client.py          # LoadTester: HTTP client with streaming support
├── src/generators/
│   └── prompt_factory.py      # PromptFactory: Deterministic prompt generation
└── src/metrics/
    ├── stats.py               # StatsCalculator: Metrics aggregation
    └── prometheus_exporter.py # PrometheusExporter: Optional metrics export
```

## Contributing

Found issues or have suggestions? This is your internal tool - modify as needed!
