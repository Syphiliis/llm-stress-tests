import asyncio
import argparse
import yaml
import logging
import os
import json
import csv
from colorama import init
from datetime import datetime as dt

from src.config.schema import GlobalConfig
from src.engine.orchestrator import LoadTestOrchestrator

# Initialize colorama
init(autoreset=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def main():
    parser = argparse.ArgumentParser(description="LLM Load Testing Tool")
    parser.add_argument("--config", default="config/workload.yaml", help="Path to configuration file")
    parser.add_argument("--output", default="results", help="Directory to save results")
    args = parser.parse_args()

    # Load Config from YAML
    try:
        with open(args.config, "r") as f:
            raw_config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {args.config}")
        return
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file: {e}")
        return

    # Validate and Parse Config with Pydantic
    try:
        config = GlobalConfig(**raw_config)
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        # Helpful error message for specific Pydantic errors could go here
        return

    # Create Output Interface
    run_ts = dt.now().strftime("%Y%m%d_%H%M%S")
    base_output = os.path.join(args.output, run_ts)
    os.makedirs(base_output, exist_ok=True)
    combined_csv_path = os.path.join(base_output, "combined_results.csv")

    def validate_config(cfg: GlobalConfig):
        if cfg.workload.duration_seconds != 1200:
            logger.warning("duration_seconds != 1200s (20 min). Requirement not enforced by code.")
        if cfg.workload.iterations < 1:
            logger.warning("iterations < 1; no consecutive runs will occur.")
        if cfg.prompts.strategy == "uniform":
            logger.warning("prompts.strategy is 'uniform'; progressive input sizing is not enabled.")
        if cfg.prompts.strategy == "staged" and not cfg.prompts.stages:
            logger.warning("prompts.strategy is 'staged' but stages are empty.")
        if cfg.prompts.strategy in ("linear", "exponential") and not cfg.prompts.ramp:
            logger.warning("prompts.strategy is ramped but prompts.ramp is not set; defaulting to min/max.")

    validate_config(config)

    async def execute_run(conf: GlobalConfig, out_dir: str, label: str, run_id: str, iteration: int, include_header: bool):
        orchestrator = LoadTestOrchestrator(conf)
        try:
            await orchestrator.run()
        except KeyboardInterrupt:
            logger.info("Test interrupted by user")
        except Exception as e:
            logger.error(f"Test failed with error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            orchestrator.save_report(
                out_dir,
                label=label,
                csv_path=combined_csv_path,
                include_header=include_header,
                run_id=run_id,
                iteration=iteration
            )
        return orchestrator.stats.calculate_summary()

    summaries = []
    total_iterations = config.workload.iterations

    for iter_idx in range(total_iterations):
        iter_label = f"iter{iter_idx + 1}"
        if config.comparison_mode and config.servers:
            for server in config.get_servers():
                single_config = config.model_copy(update={
                    "server": server,
                    "servers": None,
                    "comparison_mode": False
                })
                run_id = f"{server.name}_{iter_label}"
                summary = await execute_run(
                    single_config,
                    base_output,
                    label=f"{server.name}_{iter_label}",
                    run_id=run_id,
                    iteration=iter_idx + 1,
                    include_header=not os.path.exists(combined_csv_path)
                )
                summaries.append({"model": server.name, "iteration": iter_idx + 1, "summary": summary})
        else:
            model_name = config.server.name if config.server else "default"
            run_id = f"{model_name}_{iter_label}"
            summary = await execute_run(
                config,
                base_output,
                label=iter_label,
                run_id=run_id,
                iteration=iter_idx + 1,
                include_header=not os.path.exists(combined_csv_path)
            )
            summaries.append({"model": model_name, "iteration": iter_idx + 1, "summary": summary})

    # Comparison rows appended to combined CSV
    if config.comparison_mode and len(set(s["model"] for s in summaries)) > 1:
        def mean(values):
            vals = [v for v in values if v is not None]
            return sum(vals) / len(vals) if vals else None

        per_model = {}
        for entry in summaries:
            model = entry["model"]
            summ = entry["summary"]
            per_model.setdefault(model, []).append(summ)

        rows = []
        for model, sums in per_model.items():
            rows.append({
                "record_type": "comparison",
                "run_id": f"{model}_aggregate",
                "ts": dt.now().timestamp(),
                "label": "aggregate",
                "model": model,
                "metric_name": "latency_p90_mean",
                "metric_value": mean([s.latency_p90 for s in sums]),
            })
            rows.append({
                "record_type": "comparison",
                "run_id": f"{model}_aggregate",
                "ts": dt.now().timestamp(),
                "label": "aggregate",
                "model": model,
                "metric_name": "ttft_p50_mean",
                "metric_value": mean([s.ttft_p50 for s in sums]),
            })
            rows.append({
                "record_type": "comparison",
                "run_id": f"{model}_aggregate",
                "ts": dt.now().timestamp(),
                "label": "aggregate",
                "model": model,
                "metric_name": "tps_mean",
                "metric_value": mean([s.global_throughput_tokens_per_sec for s in sums]),
            })
            rows.append({
                "record_type": "comparison",
                "run_id": f"{model}_aggregate",
                "ts": dt.now().timestamp(),
                "label": "aggregate",
                "model": model,
                "metric_name": "error_rate_mean",
                "metric_value": mean([
                    (s.failed_requests / s.total_requests) if s.total_requests else None
                    for s in sums
                ]),
            })
            rows.append({
                "record_type": "comparison",
                "run_id": f"{model}_aggregate",
                "ts": dt.now().timestamp(),
                "label": "aggregate",
                "model": model,
                "metric_name": "stability_score_mean",
                "metric_value": mean([s.stability_score for s in sums]),
            })

        if rows:
            with open(combined_csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=LoadTestOrchestrator.CSV_COLUMNS)
                writer.writerows(rows)


if __name__ == "__main__":
    asyncio.run(main())
