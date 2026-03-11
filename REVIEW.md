# Repository Review

**Date:** March 11, 2026  
**Project:** `llm-stress-tests`

## Current State

The repository is in a materially better place than the initial baseline.
It now supports:

- validated YAML configuration through `pydantic`
- single-endpoint and multi-endpoint runs
- `closed_loop`, `open_loop`, and `hybrid` execution modes
- dynamic load profiles (`burst`, `spike`, `wave`, `cooldown`, `composite`)
- realistic think time
- Ollama `/api/generate` support
- combined CSV reporting
- remote GPU sampling over SSH
- direct unit coverage for config, clients, load profiles, sampling, and think time

## Strengths

### Async execution model

The `asyncio` and `aiohttp` architecture remains the right base for high-concurrency client-side load generation.

### Better operational realism

The scheduler is no longer limited to a startup ramp followed by a flat plateau. The new workload modes and load profiles make it possible to reproduce meaningful production traffic patterns.

### Better observability

The tool now exposes target load, in-flight pressure, queue wait, per-user throughput, TTFT drift, and remote GPU metrics in a single CSV export.

### Better safety

Configuration validation is stricter, exit codes are correct on failure paths, optional dependencies are handled more defensively, and the test suite covers the most important regression vectors.

## Remaining Limits

### Hybrid mode is not yet a true overlay

`hybrid` is still a user-driven mode with dynamic user targets and think time. It does not yet combine a persistent user population with an independent open-loop injection layer.

### Remote host visibility is partial

GPU metrics can be collected over SSH, but remote CPU and RAM are not yet sampled from the same host.

### Single-generator ceiling

At very high target rates, the load generator itself can become part of the bottleneck. Distributed generators are not implemented yet.

## Recommended Next Steps

1. Implement a true hybrid overlay: closed-loop background users plus independent open-loop bursts.
2. Add remote CPU and RAM sampling for the GPU host.
3. Add distributed runner support when a single client is no longer enough.
4. Extend stats coverage further around reporting and CSV serialization if the output format keeps evolving.

## Summary

This is now a credible LLM stress-testing tool rather than a simple concurrency loop. The remaining work is mostly about deeper production realism and scaling the generator itself, not fixing core architectural weaknesses.
