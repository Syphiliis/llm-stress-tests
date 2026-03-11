# Repository Structure

This document describes the current code layout and the role of each major module.

## Top-Level Files

- `main.py`: non-interactive entry point for running a scenario from YAML
- `cli.py`: interactive launcher that builds or selects configs before starting a run
- `run_remote.sh`: legacy helper for remote execution
- `requirements.txt`: Python runtime dependencies
- `README.md`: project overview and usage
- `USAGE_GUIDE.md`: additional usage notes
- `REVIEW.md`: review notes from earlier iterations

## Configuration

All reusable scenarios live under `config/`.

- `workload.yaml`: baseline single-endpoint config
- `flash_only.yaml`, `thinker_only.yaml`, `dual_warfare.yaml`, `mixed_warfare.yaml`: older predefined scenarios
- `comparison_flash_thinker.yaml`: sequential comparison mode
- `production_wave.yaml`: hybrid wave example with remote GPU sampling
- `open_loop_spike_poisson.yaml`: open-loop rate-driven spike example
- `burst_cooldown.yaml`: composite user-profile example
- `test-ci.yaml`, `test-mixed.yaml`: minimal configs used for quick validation

## Source Tree

### `src/clients/`

- `base.py`: abstract client interface
- `llama_cpp.py`: request/streaming client for `llama.cpp` and Ollama `/api/generate`
- `composite.py`: weighted client selection across multiple endpoints

### `src/config/`

- `schema.py`: Pydantic schema for workload, load profiles, think time, system metrics, and topology validation

### `src/engine/`

- `orchestrator.py`: central runtime coordinator
  - builds prompts
  - applies workload mode
  - schedules requests
  - manages background samplers
  - writes CSV reports
- `load_profile.py`: time-based load profile engine
  - constant
  - ramp
  - burst
  - spike
  - wave
  - cooldown
  - composite

### `src/generators/`

- `prompt_factory.py`: deterministic cache-busting prompt generation
- `token_scheduler.py`: prompt-size progression over time
- `think_time.py`: user think time sampler

### `src/metrics/`

- `stats.py`: request model, summary model, and aggregate calculations
- `system_sampler.py`: local or SSH-based GPU sampling plus optional local CPU/RAM
- `network.py`: ping collection
- `prometheus_exporter.py`: Pushgateway integration

### `src/utils/`

- `terminal.py`: colorized terminal helpers with fallback when `colorama` is unavailable

## Tests

The `tests/` directory contains focused unit tests around the core scheduling and validation logic.

- `test_llama_cpp_client.py`: payloads and stream parsing for llama.cpp/Ollama
- `test_config_schema.py`: config validation and edge cases
- `test_load_profile.py`: load profile scheduling semantics
- `test_think_time.py`: think time sampling rules
- `test_system_sampler.py`: local/SSH system sampler behavior
- `test_cli.py`: CLI config cleanup behavior

## Runtime Flow

1. `main.py` or `cli.py` loads YAML into `GlobalConfig`.
2. `LoadTestOrchestrator` builds:
   - the target client
   - the prompt generator
   - the token scheduler
   - the load profile scheduler
   - the think time sampler
   - optional ping/system/prometheus helpers
3. The orchestrator runs in:
   - `closed_loop`
   - `hybrid`
   - `open_loop`
4. Metrics are aggregated in `StatsCalculator`.
5. A combined CSV is written under `results/<timestamp>/`.

## Where To Extend Next

- If you want new traffic shapes, start in `src/engine/load_profile.py`.
- If you want richer user behavior, start in `src/generators/think_time.py` and `src/engine/orchestrator.py`.
- If you want more observability, start in `src/metrics/system_sampler.py` and `src/metrics/stats.py`.
- If you want new user-facing scenarios, add them in `config/`.
