import logging
import argparse
from collections import defaultdict
from task import Task
import config.benchmark  # Ensure this import correctly references where the benchmarks are defined
from config.log import set_log_config

# Set up logging configuration
logger = logging.getLogger(__name__)
set_log_config()

def parse_benchmark_index(benchmark_name):
    """
    Converts a benchmark name to an index if applicable.
    Assumes benchmark names are in the format 'dataX' where X is the index.
    """
    if benchmark_name.startswith('data'):
        try:
            index = int(benchmark_name[4:])  # This extracts the number part from names like 'data1'
            return index
        except ValueError:
            logger.error(f"Invalid benchmark name format: {benchmark_name}")
            raise
    else:
        logger.error(f"Unknown benchmark name: {benchmark_name}")
        raise ValueError(f"Unknown benchmark name: {benchmark_name}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run a machine learning benchmark.")
    parser.add_argument('--benchmark', type=str, default="data1", help="Running Benchmark (See ./config/benchmark.py for details)")
    args = parser.parse_args()

    # Log the selected benchmark
    logger.info(f"Selected benchmark: {args.benchmark}")

    try:
        # Convert benchmark name to index and retrieve the corresponding benchmark
        dataset_index = parse_benchmark_index(args.benchmark)
        benchmark = config.benchmark.get_benchmark_by_index(dataset_index)  # Ensure this function is properly defined and imported
        global_args, train_args, algorithm = benchmark.get_args()
        
        # Log configuration details
        print('Training')
        logger.info("-- Training Start --")
        logger.info("Global args - Dataset: %s, Model: %s", global_args['dataset'], global_args['model'])
        
        # Initialize and run the task
        print('check')
        classification_task = Task(global_args=global_args, train_args=train_args, algorithm=algorithm)
        print(classification_task)
        logger.info(classification_task)
        classification_task.run()       
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
