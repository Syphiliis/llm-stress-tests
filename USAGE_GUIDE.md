# LLM Stress Test Tool - Usage Guide

## Overview

This repository is a Python async load generator for `llama.cpp` and Ollama inference endpoints.
It is designed to answer production-style questions such as:

- What happens to TTFT and latency when concurrency ramps up?
- How does the model behave during bursts or abrupt spikes?
- Does the GPU recover cleanly after a short overload window?
- Is the client itself queueing work before the server becomes the bottleneck?

The current implementation supports:

- workload modes: `closed_loop`, `open_loop`, `hybrid`
- load profiles: `constant`, `ramp`, `burst`, `spike`, `wave`, `cooldown`, `composite`
- arrival patterns: deterministic or Poisson
- user think time: `fixed`, `uniform`, `exponential`, `lognormal`
- system sampling: local GPU or remote GPU over SSH
- combined CSV export with request, summary, ping, system, and analysis rows

For the repository layout, see [STRUCTURE.md](./STRUCTURE.md).

## Installation

```bash
pip install -r requirements.txt
```

## Fast Start

Run the default single-endpoint scenario:

```bash
python main.py --config config/workload.yaml
```

Run the interactive launcher:

```bash
python cli.py
```

Run the test suite:

```bash
python3 -m unittest discover -s tests -v
python3 -m compileall .
```

## Picking the Right Mode

### `closed_loop`

Each virtual user waits for the previous response before sending the next request.

Use it when you want to model user-driven concurrency and observe how latency changes under growing parallelism.

### `open_loop`

Requests are launched according to a target rate, even if latency grows.

Use it when you want to preserve pressure during overload and avoid the classic closed-loop effect where throughput collapses simply because the client is waiting on responses.

### `hybrid`

This is still user-driven like closed-loop, but it supports dynamic user targets over time and realistic think time.

Use it when you want traffic that looks more human than a tight request loop.

## Picking the Right Profile

- `constant`: fixed load for the full run
- `ramp`: progressive increase or decrease
- `burst`: repeated short peaks over a base load
- `spike`: abrupt temporary jump
- `wave`: cyclic shape with `sine`, `triangle`, or `sawtooth`
- `cooldown`: progressive descent after a peak
- `composite`: multiple segments chained together

## Common Scenarios

### 1. Stable Concurrency Test

```yaml
server:
  name: "flash"
  base_url: "http://localhost:8080/completion"

workload:
  users: 20
  duration_seconds: 600
  mode: closed_loop
  max_in_flight: 64
  seed: 42

prompts:
  min_tokens: 128
  max_tokens: 1024
```

### 2. Open-Loop Spike

```yaml
workload:
  users: 8
  duration_seconds: 900
  mode: open_loop
  arrival_distribution: poisson
  max_in_flight: 128

load_profile:
  type: spike
  base_rps: 2
  peak_rps: 20
  start_at_seconds: 300
  hold_seconds: 45
```

### 3. Production-Like Wave With Think Time

```yaml
workload:
  users: 20
  duration_seconds: 1800
  mode: hybrid
  max_in_flight: 128

load_profile:
  type: wave
  min_users: 20
  max_users: 120
  period_seconds: 180
  shape: triangle

think_time:
  enabled: true
  distribution: lognormal
  mean_seconds: 1.2
  max_seconds: 8
```

### 4. Remote GPU Sampling

```yaml
system:
  enabled: true
  source: ssh
  ssh_host: "gpu-host.example.com"
  ssh_user: "ubuntu"
  ssh_port: 22
  interval_seconds: 10
```

### 5. Ollama Endpoint

```yaml
server:
  name: "ollama"
  base_url: "http://localhost:11434/api/generate"
  model_alias: "qwen2.5:7b"
```

The client detects the Ollama path and switches payload/stream parsing automatically.

## Example Config Files

- `config/workload.yaml`: default single-endpoint scenario
- `config/production_wave.yaml`: hybrid wave with remote GPU sampling
- `config/open_loop_spike_poisson.yaml`: open-loop Poisson spike
- `config/burst_cooldown.yaml`: burst and cooldown composite profile
- `config/comparison_flash_thinker.yaml`: sequential multi-endpoint comparison

## Reading the Output

Each run writes a combined CSV under `results/<timestamp>/combined_results.csv`.

Important record types:

- `request`: one row per request
- `summary`: aggregate metrics for the run
- `config`: serialized config used for the run
- `ping`: network snapshots
- `system`: CPU/RAM/GPU snapshots
- `ttft_vs_input`: TTFT bucketed by input size
- `tokens_vs_concurrency`: throughput bucketed by concurrency
- `reactivity`: early vs late TTFT drift
- `per_user_tps`: per-user throughput
- `comparison`: comparison rows when running sequential multi-model tests

Important columns:

- `ttft`: time to first token
- `latency`: end-to-end latency
- `queue_wait`: time spent waiting before a request could actually start
- `target_users`: intended concurrency at that moment
- `target_rps`: intended launch rate at that moment
- `in_flight`: observed in-process request count when the request started
- `think_time`: sampled pause after the previous request in user-driven modes

## How To Interpret Results

### For GPU saturation

Watch these together:

- `ttft_p50` and `ttft_p90`
- `latency_p90`
- `tokens_vs_concurrency`
- remote GPU utilization and memory usage

If TTFT rises early while GPU utilization stays high, the model is usually waiting on scarce compute.

### For client-side bottlenecks

Watch:

- `queue_wait`
- `in_flight`
- `max_in_flight`

If queue wait climbs before the server metrics degrade, the client-side concurrency cap is probably constraining the run.

### For recovery after spikes

Use:

- `spike` or `burst` profiles
- `reactivity` rows
- early vs late TTFT changes

If TTFT remains elevated long after the spike, the system is recovering slowly.

## Troubleshooting

### Connection refused

- verify the endpoint URL and port
- confirm the inference server is reachable from the load generator host
- check local firewall or SSH tunnel configuration

### High error rate

- lower `users`, `target_rps`, or `max_in_flight`
- increase client timeout settings
- inspect inference server logs for OOM, context, or batching failures

### No GPU metrics

- verify `nvidia-smi` exists on the target host
- if using SSH, verify key-based auth or non-interactive access works
- check `system.gpu_command` if your environment needs a custom query

### Results look too flat

- use `open_loop` instead of `closed_loop`
- add `burst`, `spike`, or `wave`
- enable `think_time` for more realistic user pacing

## Current Limits

- `hybrid` is still user-driven; it is not yet a full closed-loop plus open-loop overlay
- distributed multi-generator runs are not implemented
- remote CPU and RAM collection from the GPU host is not implemented yet

## Code Pointers

- [main.py](./main.py): non-interactive runner
- [cli.py](./cli.py): interactive launcher
- [src/engine/orchestrator.py](./src/engine/orchestrator.py): runtime orchestration
- [src/engine/load_profile.py](./src/engine/load_profile.py): load target scheduler
- [src/generators/think_time.py](./src/generators/think_time.py): think time distributions
- [src/metrics/stats.py](./src/metrics/stats.py): aggregation and verdicts
- [src/metrics/system_sampler.py](./src/metrics/system_sampler.py): local and SSH GPU sampling
