# LLM Stress Test Tool

Async Python load generator for `llama.cpp` and Ollama `/api/generate`, built to stress local or remote LLM inference servers under realistic production-like traffic.

It now supports:

- `closed_loop`, `open_loop`, and `hybrid` workload modes
- load profiles: `constant`, `ramp`, `burst`, `spike`, `wave`, `cooldown`, `composite`
- deterministic or Poisson arrivals
- user think time simulation
- local GPU sampling or remote GPU sampling over SSH
- combined CSV output with per-request, per-window, and summary records

See [STRUCTURE.md](./STRUCTURE.md) for the repository layout.

## What It Is Good At

- validating TTFT and latency drift when concurrency ramps up
- reproducing bursty traffic instead of only flat sustained load
- comparing multiple model endpoints on the same GPU host
- correlating client-side behavior with remote GPU utilization
- producing repeatable scenarios with a fixed seed

## Quick Start

```bash
git clone https://github.com/Syphiliis/llm-stress-tests.git
cd llm-stress-tests
pip install -r requirements.txt
python main.py --config config/workload.yaml
```

## Workload Model

### Modes

- `closed_loop`: each virtual user waits for a response before sending the next request
- `hybrid`: user-driven flow with think time and dynamic user count over time
- `open_loop`: request launches follow a target rate even when latency grows

### Load Profiles

- `constant`: fixed users or fixed RPS
- `ramp`: linear increase or decrease
- `burst`: periodic short peaks above a base load
- `spike`: abrupt temporary surge
- `wave`: cyclic pattern with `sine`, `triangle`, or `sawtooth`
- `cooldown`: progressive descent after a peak period
- `composite`: sequential list of segments

### Think Time

Supported distributions:

- `fixed`
- `uniform`
- `exponential`
- `lognormal`

This allows user behavior closer to real traffic than a pure tight request loop.

## Example Scenarios

- [config/workload.yaml](./config/workload.yaml): legacy/default single-endpoint scenario
- [config/production_wave.yaml](./config/production_wave.yaml): hybrid wave profile with SSH GPU sampling
- [config/open_loop_spike_poisson.yaml](./config/open_loop_spike_poisson.yaml): open-loop spike using Poisson arrivals
- [config/burst_cooldown.yaml](./config/burst_cooldown.yaml): burst + cooldown composite profile
- [config/comparison_flash_thinker.yaml](./config/comparison_flash_thinker.yaml): sequential comparison across two endpoints

## Configuration Examples

### llama.cpp

```yaml
server:
  name: "flash"
  base_url: "http://localhost:8080/completion"

workload:
  users: 20
  duration_seconds: 900
  mode: closed_loop
  max_in_flight: 64
  seed: 42

prompts:
  min_tokens: 128
  max_tokens: 1024
  prefix: "Stress test: "
```

### Ollama

```yaml
server:
  name: "ollama"
  base_url: "http://localhost:11434/api/generate"
  model_alias: "qwen2.5:7b"
```

The client auto-detects Ollama from the `/api/generate` path and sends the correct request payload.

### Open-Loop Spike

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

### Think Time

```yaml
think_time:
  enabled: true
  distribution: lognormal
  mean_seconds: 1.2
  max_seconds: 8
  after_error_multiplier: 1.5
```

### Remote GPU Sampling

```yaml
system:
  enabled: true
  source: ssh
  ssh_host: "24.124.32.70"
  ssh_user: "ubuntu"
  ssh_port: 22
  interval_seconds: 10
```

## Running Tests

```bash
python3 -m unittest discover -s tests -v
python3 -m compileall .
```

## CLI

Interactive mode is available via:

```bash
python cli.py
```

The CLI is useful for quickly switching between predefined configs or generating a temporary config for a remote target.

## Output

Each run writes a combined CSV under `results/<timestamp>/combined_results.csv`.

The CSV contains multiple record types:

- `request`: per-request latency, TTFT, target users/RPS, in-flight count, think time
- `summary`: aggregate latency, TTFT, throughput, error breakdown, stability score
- `ping`: network samples
- `system`: CPU/RAM/GPU snapshots
- `ttft_vs_input`: TTFT grouped by input size bucket
- `tokens_vs_concurrency`: throughput grouped by in-flight concurrency
- `reactivity`: early vs late TTFT drift
- `per_user_tps`: per-user throughput
- `comparison`: aggregate rows for sequential multi-model comparison runs
- `config`: serialized config used for the run

## Observability Notes

- `queue_wait` becomes especially useful in `open_loop` mode because it shows launch delay under client-side pressure.
- `target_users`, `target_rps`, and `in_flight` are exported so you can compare intended load versus actual in-process pressure.
- With `system.source: ssh`, GPU metrics are collected remotely; local CPU/RAM remain optional via `include_local_cpu_ram`.

## Current Limits

- `hybrid` is still user-driven; it is not yet a full overlay of closed-loop users plus independent open-loop burst injection.
- Distributed load generation across multiple client machines is not implemented yet.
- Remote CPU/RAM collection from the GPU host is not implemented; only remote GPU sampling is available over SSH.

## Main Entry Points

- [main.py](./main.py): non-interactive runner
- [cli.py](./cli.py): interactive config-driven launcher
- [src/engine/orchestrator.py](./src/engine/orchestrator.py): runtime coordinator
- [src/engine/load_profile.py](./src/engine/load_profile.py): time-based load target engine
- [src/generators/think_time.py](./src/generators/think_time.py): think time sampler
- [src/metrics/system_sampler.py](./src/metrics/system_sampler.py): local/SSH GPU sampling
