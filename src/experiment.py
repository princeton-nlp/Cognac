import sys
import json
import yaml
import uuid
import logging
import argparse
import tempfile
from pathlib import Path


class ExperimentManager:
    """
    The only object needed to be instantiated throughout an experiment run.

    Functionalities:
    1. Create run dir
    2. Manage all logging
       a. Save command
       b. Save config/args
       c. Save predictions
       d. Save final eval stats
    3. Manage all metric calculations
    """
    def __init__(self,
            run_dir=None,
            name=None,
            override=True,
            num_runs=1,
            **kwargs
        ):
        """
        Create run dir and setup file paths.
        By default, one run folder will be created that contains the predictions file.
        If `num_runs` > 1, then the amount of sub dirs will be created.
        The sub dirs will be named `seed_{run_id}`.
        """
        self.num_runs = num_runs

        # Set up run dir path.
        if run_dir is None:
            run_dir = '/tmp/unnamed-experiments'
        if name is None:
            name = str(uuid.uuid4()).split('-')[-1]

        self.run_dir = Path(run_dir) / name

        self.config_path = self.run_dir / 'config.yaml'

        self.prediction_paths = [
            self.run_dir / f'seed_{run_id}' / 'predictions.jsonl'
            for run_id in range(num_runs)
        ]
        
        self.agg_result_path = self.run_dir / 'results.yaml'
        self.result_paths = [
            self.run_dir / f'seed_{run_id}' / 'results.yaml'
            for run_id in range(num_runs)
        ]

        # Create run dir and the needed paths.
        if not self.run_dir.exists() or override:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            for run_id in range(num_runs):
                seed_run_dir = self.run_dir / f'seed_{run_id}'
                seed_run_dir.mkdir(parents=True, exist_ok=True)
            print(f'Run dir `{self.run_dir}` created or existed.')
        else:
            print(f'Run dir `{self.run_dir}` not saved or overriden.')

        # Set up file loggers.
        self.file_loggers = []
        for run_id in range(self.num_runs):
            self.setup_logger(f'file_logger_{run_id}', run_id)
            self.file_loggers.append(logging.getLogger(f'file_logger_{run_id}'))

        # Set up other modules.
        self.metric_tracker = kwargs.get('metric_tracker', None)

    def setup_logger(self, logger_name, run_id=None):
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        handler = logging.FileHandler(self.prediction_paths[run_id], mode='w')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False

    def console(self, content):
        print(content)

    def log(self, content, run_id):
        assert run_id is not None, 'With `save=True`, `run_id` must be provided.'
        content_to_log = (
            json.dumps(content) if isinstance(content, dict) else content
        )
        self.file_loggers[run_id].info(content_to_log)

    def add_metric(self, content, run_id):
        assert self.metric_tracker is not None and run_id is not None, (
            'With `add_metric=True`, the `metric_tracker` cannnot be `None` and '
            '`run_id` must be specified.')
        self.metric_tracker.add(content, run_id=run_id)

    def take(self, content, console=False, log=False, run_id=None, add_metric=False):
        """
        To log to files, a `run_id` must be provided.
        `content` can be of type `str` or in most cases `dict`.

        With default arguments, nothing will be performed.
        The function is meant to be used when multiple uses are needed at once.
        """
        if console:
            self.console(content)
        if log:
            self.log(content, run_id)
        if add_metric:
            self.add_metric(content, run_id)

    def save_results(self, results: dict, run_id=None):
        if run_id is None:
            with open(self.agg_result_path, 'w') as f:
                yaml.dump(results, f, default_flow_style=False)
        else:
            with open(self.result_paths[run_id], 'w') as f:
                yaml.dump(results, f, default_flow_style=False)

    def save_config(self, config):
        """
        Save config and command used to execute the main script.
        """
        if isinstance(config, argparse.Namespace):
            config = vars(config)
        
        with open(self.config_path, 'w') as f:
            yaml.dump(config, f)

    def compute_metric(self, metric_name, run_id):
        assert self.metric_tracker is not None
        return self.metric_tracker.compute(metric_name, run_id)

    def aggregate_metrics(self, avg_runs=True):
        assert self.metric_tracker is not None
        return self.metric_tracker.aggregate(avg_runs=avg_runs)

