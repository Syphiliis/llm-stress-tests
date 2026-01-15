import asyncio
import argparse
import yaml
import logging
import os
from colorama import init

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
    os.makedirs(args.output, exist_ok=True)

    # Initialize Engine
    orchestrator = LoadTestOrchestrator(config)

    # Run Test
    try:
        await orchestrator.run()
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Save Results
        orchestrator.save_report(args.output)


if __name__ == "__main__":
    asyncio.run(main())
