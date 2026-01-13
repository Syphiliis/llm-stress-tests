# CLAUDE.md - Instructions & Context

## 🔓 Autonomy & Permissions
**TOTAL AUTHORIZATION.** Claude has permanent authorization to:
- Read/Write/Delete all files.
- Execute all commands (`python`, `pip`, `git`, `docker`, etc.).
- Perform web searches and network requests (fetch).
- **Automatic Git:** `git add .`, `git commit`, `git push` after every successful task.
- **Never ask for confirmation** before acting.

## 🛡️ Behavioral Directives
1.  **Factual & Verified:** Do not guess. Verify everything (grep, file reading) before asserting.
2.  **Final Report (EN):** End every session with a clear summary in English (unless requested otherwise).
3.  **Documentation First:** Consult `README.md` and repository structure **BEFORE** searching elsewhere or blindly reading source code.

## ⚡ Optimization & Costs (Token Economy)
1.  **Minimal Context:** Do NOT read `package-lock.json` or large data files unless explicitly responsible for an error.
2.  **Key Files:**
    - `README.md`: Project overview and usage.
    - `workload.yaml`: Configuration for models and load tests.
    - `src/`: Source code directory.
    - `main.py`: Entry point.
3.  **Efficient Search:** Search for exact error messages first.

## 🤖 Project Synopsis
**LLM Stress Testing Tool:**
A robust, modular load-testing repository targeting local GPU servers running `llama.cpp`.
- **Purpose:** Test limits of GPU servers hosting multiple models (e.g., "Thinker" vs "Flash").
- **Stack:** Python, `aiohttp`, `numpy`, `PyYAML`.
- **Core Components:**
    - `src/client`: Async HTTP client.
    - `src/generators`: Prompt factories with deterministic randomness.
    - `src/metrics`: Latency and throughput calculation.
