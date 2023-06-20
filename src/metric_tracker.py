from collections import defaultdict
import numpy as np


class MetricTracker:
    def __init__(self):
        self._state = defaultdict(lambda: defaultdict(list))

    @property
    def state(self):
        return {k: dict(v) for k, v in self._state.items()}

    def add(self, metric: dict, run_id: int):
        invalid_value_types = {str, list, dict}
        for name, value in metric.items():
            if type(value) in invalid_value_types:
                continue
            self._state[run_id][name].append(value)

    def compute(self, metric_name=None, run_id=None, ignore=('id', 'global_step')):
        """
        `metric_name` and `run_id` are expected to be passed in simultaneously,
        or they should both be `None`.
        """
        stats = defaultdict(dict)
        for run_id_, metrics in self._state.items():
            for name, values in metrics.items():
                if metric_name is not None and name != metric_name:
                    continue
                if name in ignore:
                    continue
                avg = sum(values) / len(values)
                stats[run_id_][name] = avg
                stats[run_id_][name + '_num_datapoints'] = len(values)
        stats = dict(stats)

        if run_id is not None and metric_name is not None:
            stats = stats[run_id][metric_name]
        return stats

    def aggregate(self, avg_runs=True):
        per_run_avg_stats = self.compute()

        if not avg_runs:
            return per_run_avg_stats
        else:
            run_avg_stats = defaultdict(list)
            metrics = list(per_run_avg_stats.values())
            for metric in metrics:
                for name, value in metric.items():
                    run_avg_stats[name].append(value)

            # Calculate mean/std.
            run_avg_stats = {
                k: {
                    'mean': float(np.mean(vs)),
                    'std': float(np.std(vs)),
                    'num_runs': len(vs),
                    'num_datapoints': run_avg_stats[k + '_num_datapoints'],
                }
                for k, vs in run_avg_stats.items() if not k.endswith('_num_datapoints')
            }
            return run_avg_stats
