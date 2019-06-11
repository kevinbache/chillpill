import abc
from typing import Any

import numpy as np
from tensorflow import keras

from chillpill import params


class TunerInterface(abc.ABC):
    """Interface for hyperparameter tuning.

    Used like this:

    from chillpill import params

    class MyParams(params.ParameterSet):
        num_layers=4
        num_neurons=32

    my_param_ranges = MyParams(
        num_layers=params.IntegerParameter(min_value=1, max_value=3),
        num_neurons=params.DiscreteParameter(np.logspace(2, 8, num=7, base=2)),
    )

    tuner = tuning.KerasHistoryRandomTuner(
        param_ranges=my_param_ranges,
        num_parameter_sets=10,
        metric_name_of_interest='val_acc'
    )

    tuning.run_tuning(tuner, train_fn)

    best_acc, best_params = tuner.get_best(do_max=True)
    """
    def __init__(self, param_ranges: params.ParameterSet, num_sets: int, *args, **kwargs):
        self.param_ranges = param_ranges
        self.num_sets = num_sets
        self.generated_param_sets = [None] * self.num_sets
        self.metric_of_interest_values = [None] * self.num_sets
        self._iter_index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._iter_index >= self.num_sets:
            raise StopIteration
        params = self._generate_params()
        params.set_index(self._iter_index)
        self.generated_param_sets[self._iter_index] = params

        self.result_values = []
        self.c = []
        self._iter_index += 1
        return params

    @abc.abstractmethod
    def _generate_params(self) -> params.ParameterSet:
        """New hyperparamter tuning regimes should override this method."""
        pass

    @abc.abstractmethod
    def report(self, results: Any, iter_index: int) -> float:
        pass

    def _get_best_index(self, do_max=True):
        if do_max:
            return np.asscalar(np.argmax(self.metric_of_interest_values))
        else:
            return np.asscalar(np.argmin(self.metric_of_interest_values))

    def get_best(self, do_max=True):
        index = self._get_best_index(do_max)
        return self.metric_of_interest_values[index], self.generated_param_sets[index],


class KerasHistoryRandomTuner(TunerInterface):
    """Implements random tuning and independently tracks a metric_name_of_interest from a Keras history object."""
    def __init__(
            self,
            param_ranges: params.ParameterSet,
            num_parameter_sets: int,
            metric_name_of_interest='val_acc',
    ):
        super().__init__(param_ranges, num_parameter_sets)
        self.metric_of_interest = metric_name_of_interest
        self.history_values = []

    def report(self, results: keras.callbacks.History, iter_index: int) -> float:
        metric_value_of_interest = results.history[self.metric_of_interest][-1]
        self.metric_of_interest_values[iter_index] = metric_value_of_interest
        return metric_value_of_interest

    def _generate_params(self) -> params.ParameterSet:
        return self.param_ranges.sample()


def run_tuning(tuner: TunerInterface, train_fn, verbose=True):
    """A thin, highest-level interface to a Tuner"""
    for param_set_index, param_set in enumerate(tuner):
        if verbose:
            print(f"Starting training round {param_set_index}.  ", end='')
        history = train_fn(param_set)
        metric_value = tuner.report(history, param_set.get_index())
        if verbose:
            print(f"Got metric value: {metric_value}")
