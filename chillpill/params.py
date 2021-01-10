import abc
import copy
import hashlib
import types
from typing import *

import numpy as np
import typing


class Samplable(abc.ABC):
    @abc.abstractmethod
    def sample(self):
        pass

    def to_ray_tune_search_dict(self):
        from ray import tune
        return tune.sample_from(self.sample)


class HasClassDefaults(abc.ABC):
    """Knows how to find class- and object-based data members on itself which aren't methods.

    HasClassDefault._get_member_names is like object.__dict__.keys() except it includes class members.
    """
    names_to_ignore = [
        '_abc_impl',
        '_abc_cache',
        '_abc_negative_cache',
        '_abc_negative_cache_version',
        '_abc_registry',
        'names_to_ignore',
        '_class_member_order',
        '_index',
        '_samplable_param_names',
        '_short_hash',
    ]

    # noinspection PyTypeChecker
    def _is_method(self, attribute_name):
        return isinstance(self.__getattribute__(attribute_name), (
            types.FunctionType,
            types.BuiltinFunctionType,
            types.MethodType,
            types.BuiltinMethodType,
        ))

    def _get_member_names(self):
        """This is like calling object.__dict__.keys() but __dict__ only includes instance members,
        not class members.

        This is not a foolproof way to get at the member names of a class, but it's good enough.
        """
        return [
            a for a in dir(self)
            if not a.startswith('__')
            and not self._is_method(a)
            and a not in self.names_to_ignore
        ]

    def _has_member(self, member_name: Text):
        return member_name in self._get_member_names()

    def __repr__(self):
        return "{}({})".format(
            self.__class__.__name__,
            ', '.join('{}: {}'.format(k, self.__dict__[k]) for k in self._get_member_names())
        )

    def _str_inner(self, obj):
        typestr = obj.__class__.__name__
        if isinstance(obj, ParameterSet):
            return str(obj).replace('\n', '\n  ')
        elif hasattr(obj, 'shape'):
            shape = ', '.join([str(e) for e in obj.shape]) or '[]'
            return f'{typestr}(shape={shape})'
        elif isinstance(obj, (list, tuple)):
            l = len(obj)
            # if isinstance(obj, list):
            #     empty_str = '[]'
            # else:
            #     empty_str = '()'
            # first_str = self._str_inner(obj[0]) if l else empty_str
            return f'{typestr}(len={l}, {str(obj)})'
        elif isinstance(obj, dict):
            # if not obj:
            #     return f'{typestr}(empty)'
            l = len(obj)
            # k0 = list(obj.keys())[0]
            # v0 = obj[k0]
            # first_str = f'{self._str_inner(k0)}: {self._str_inner(v0)}' if l else '{}'
            # return f'{typestr}([len={l}, first={first_str})'
            return f'{typestr}(len={l}, {str(obj)})'
        else:
            return str(obj)

    def __str__(self):
        d = {k: self._str_inner(self.__getattribute__(k)) for k in self._get_member_names()}
        strs = [f'{k}={v}' for k, v in d.items()]
        arg_str = ',\n  '.join(strs)
        return f'{self.__class__.__name__}(\n  {arg_str}\n)'

    @classmethod
    def _to_dict_inner(cls, v: Any):
        if v is None:
            return None
        elif hasattr(v, 'to_dict'):
            return v.to_dict()
        elif isinstance(v, (Mapping, MutableMapping)):
            return {k: cls._to_dict_inner(vv) for k, vv in v.items()}
        elif isinstance(v, list):
            return [cls._to_dict_inner(vv) for vv in v]
        elif isinstance(v, tuple):
            return tuple(cls._to_dict_inner(vv) for vv in v)
        elif isinstance(v, np.integer):
            return int(v)
        elif isinstance(v, np.floating):
            return float(v)
        elif isinstance(v, np.ndarray):
            return v.tolist()
        else:
            return v

    def to_dict(self):
        d = {}

        for k in sorted(self._get_member_names()):
            v = self.__getattribute__(k)
            d[k] = self._to_dict_inner(v)
            # if v is None:
            #     d[k] = None
            # elif hasattr(v, 'to_dict'):
            #     d[k] = v.to_dict()
            # elif isinstance(v, np.integer):
            #     return int(v)
            # elif isinstance(v, np.floating):
            #     return float(v)
            # elif isinstance(v, np.ndarray):
            #     d[k] = v.tolist()
            # else:
            #     d[k] = v
        return d

    @classmethod
    def _from_dict_inner(cls, d: Any, obj: Any):
        """
        Args:
            d: the dict or subsection of dict to be read into the obj
            obj: the hp object (or subsection of hp object) to read the dict into.

        Returns: Any
            obj with values from d stuffed into it
        """
        if np.issubdtype(type(d), np.integer):
            obj = int(d)
        elif np.issubdtype(type(d), np.floating):
            obj = float(d)
        elif isinstance(d, MutableMapping):
            for k, v in d.items():
                objv = v
                if hasattr(obj, k):
                    objv = getattr(obj, k)
                    if hasattr(objv, 'from_dict'):
                        objv = objv.from_dict(v)
                    else:
                        objv = cls._from_dict_inner(v, objv)

                setattr(obj, k, objv)
        else:
            obj = d
        return obj

    @classmethod
    def from_dict(cls, d: Dict):
        # todo: check if copy fixes class membership problem
        hp = copy.deepcopy(cls())
        return cls._from_dict_inner(d, hp)

    def __contains__(self, item):
        return item in self.keys()

    def keys(self):
        return self._get_member_names()

    def __getitem__(self, item):
        if item not in self.keys():
            raise ValueError(f'Keys for this class are: {self.keys()} but you tried to get {item}')
        return getattr(self, item)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __iter__(self):
        # return iter(self.items())
        d = {k: getattr(self, k) for k in self.keys()}
        return iter(d.items())


class ParameterSet(HasClassDefaults, Samplable):
    """Represents a set of Parameters.

    Subclass this and add default values as class members.

    This class is intended to be used in four cases:
        1) In development for defining expected hyperparameter types and default values
        2) For use by local hyperparamter tuning objets like the `KerasHistoryRandomTuner`
        3) For instantiating `HyperparamSearchSpec` objects which can be written to YAML
           to be processed by the Google Cloud AI Platform
        4) On a remote machine for instantiating parameters from passed arguments

    Examples:

    ##########################
    # 1) During development: #
    ##########################
    ```
    import numpy as np
    from chillpill import params

    class ModelHyperParams(params.ParameterSet):
        filter_size = 3
        num_hidden_layers = 2
        num_neurons_per_layer = 32
        dropout_rate = 0.5
        activation = 'relu'
        output_dir = '/tmp/output'
    ```

    ########################
    # 2) For local tuning: #
    ########################
    ```
    from chillpill import params, tuning

    def train_fn(params: ModelHyperParams):
        ...

    # instantiate the same param class you defined above, overriding some parameters with search ranges
    # the fact that the class is shared
    my_param_ranges = ModelHyperParams(
        filter_size=params.DiscreteParameter([3, 5, 7]),
        num_hidden_layers=params.IntegerParameter(min_value=1, max_value=3),
        num_neurons_per_layer=params.DiscreteParameter(np.logspace(2, 8, num=7, base=2)),
        dropout_rate=params.DoubleParameter(min_value=-0.1, max_value=0.9),
        activation = 'relu',
        output_dir = '/tmp/output',
    )

    tuner = tuning.KerasHistoryRandomTuner(
        param_ranges=my_param_ranges,
        num_parameter_sets=10,
        metric_name_of_interest='val_acc'
    )

    tuning.run_tuning(tuner, train_fn)

    best_acc, best_params = tuner.get_best(do_max=True)
    ```

    ################################################################
    # 3) Creating a HyperparamSearchSpec for distributed training: #
    ################################################################
    ```
    from chillpill import search

    spec = search.HyperparamSearchSpec(
        max_trials=10,
        max_parallel_trials=5,
        max_failed_trials=2,
        hyperparameter_metric_tag='val_acc',
    )

    my_param_ranges = ModelHyperParams(
        filter_size=params.DiscreteParameter([3, 5, 7]),
        num_hidden_layers=params.IntegerParameter(min_value=1, max_value=3),
        num_neurons_per_layer=params.DiscreteParameter(np.logspace(2, 8, num=7, base=2)),
        dropout_rate=params.DoubleParameter(min_value=-0.1, max_value=0.9),
        activation = 'relu',
        output_dir = '/tmp/output',
    )

    spec.add_parameters(my_param_ranges)
    spec.to_training_input_yaml('hps.yaml')
    ```
    --> a file like this:
    ```
        trainingInput:
          hyperparameters:
            algorithm: ALGORITHM_UNSPECIFIED
            enableTrialEarlyStopping: true
            goal: MAXIMIZE
            hyperparameterMetricTag: val_acc
            maxFailedTrials: 2
            maxParallelTrials: 5
            maxTrials: 10
            params:
            - {maxValue: 0.9, minValue: -0.1, parameterName: dropout_rate, type: DOUBLE}
            - discreteValues: [3, 5, 7]
              parameterName: filter_size
              type: DISCRETE
            - {maxValue: 3, minValue: 1, parameterName: num_hidden_layers, type: INTEGER}
            - discreteValues: [4, 8, 16, 32, 64, 128, 256]
              parameterName: num_neurons_per_layer
              type: DISCRETE
            resumePreviousJobId: null
    ```


    #####################################################################################
    # 4) From within a remote training script which passes in parameters via arguments: #
    #####################################################################################
    In a script invoked with args:
        --num_hidden_layers=3
        --num_neurons_per_layer=2
        --dropout_rate=0.2
        --learning_rate=0.4
        --activation=relu
    ```
    from chillpill import params
    from chillpill import simple_argparse
    params = ModelHyperParams.from_dict(simple_argparse.args_2_dict())
    assert(
      str(params) ==  \
      ModelHyperParams(activation: relu, dropout_rate: 0.2, filter_size: 3, num_hidden_layers: 2, num_neurons_per_layer: 2, output_dir: /tmp/output))
    )

    def build_model(params: ModelHyperParams):
        pass

    model = build_model(params)
    ```
    """
    def __init__(self, **kwargs):
        super().__init__()
        self._class_member_order = []

        # copy class attributes into the instance
        for k in self._get_member_names():
            self._class_member_order.append(k)
            self.__setattr__(k, self.__getattribute__(k))

        for k, v in kwargs.items():
            self.__setattr__(k, v)

        self._index = None

        # the names of the parameters which at least started life as Samplable.
        # i.e.: these are the parameters which will vary across a hyperparameter sweep
        self._samplable_param_names = self.get_samplable_param_names()

        self._short_hash = None

        self.__post_init__()

    def __post_init__(self):
        pass

    def sample(self):
        self._samplable_param_names = self.get_samplable_param_names()
        out = copy.deepcopy(self)
        for k, v in out.__dict__.items():
            if isinstance(v, Samplable):
                out.__setattr__(k, v.sample())
        return out

    def to_ray_tune_search_dict(self):
        def _to_ray_tune_search_dict_inner(d: Any):
            if not hasattr(d, 'items'):
                return d

            for k, v in d.items():
                if hasattr(v, 'to_ray_tune_search_dict'):
                    d[k] = v.to_ray_tune_search_dict()
                elif isinstance(v, Mapping):
                    d[k] = _to_ray_tune_search_dict_inner(v)
                elif isinstance(v, list):
                    d[k] = [_to_ray_tune_search_dict_inner(e) for e in v]
                elif isinstance(v, tuple):
                    d[k] = tuple(_to_ray_tune_search_dict_inner(e) for e in v)
                else:
                    # other members remain unchanged
                    pass
            return d

        self._samplable_param_names = self.get_samplable_param_names()
        d = self.to_dict()
        d = _to_ray_tune_search_dict_inner(d)

        return d

    # def to_ray_tune_samplable(self):
    #     return self.to_ray_tune_search_dict()

    def get_name_str(self):
        """Get a good default name for a run governed by these parameters."""
        return '-'.join(f'param={getattr(self, param)}' for param in self._samplable_param_names)

    def get_samplable_param_names(self) -> List[str]:
        out = []
        for name in self._get_member_names():
            v = getattr(self, name)
            if isinstance(v, Samplable):
                out.append(name)
        return out

    def get_index(self):
        """Used for differentiating parameter sets within a parameter search"""
        return self._index

    def set_index(self, index):
        self._index = index

    def get_short_hash(self, num_chars=4):
        # cache _short_hash because we don't want it to change as we 1) add more parameters or 2) sample from the set
        if self._short_hash is not None:
            return self._short_hash
        return self.reset_short_hash(num_chars)

    def reset_short_hash(self, num_chars=4):
        d = self.to_dict()
        h = hashlib.sha1(str(d).encode('utf-8'))
        self._short_hash = h.hexdigest()[-num_chars:]
        return self._short_hash


class Float(Samplable):
    def __init__(self, min_value: float, max_value: float):
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value

    def sample(self):
        return np.random.uniform(self.min_value, self.max_value)

    def to_ray_tune_search_dict(self):
        from ray import tune
        return tune.uniform(self.min_value, self.max_value)

    def __repr__(self):
        return f'{self.__class__.__name__}({self.min_value}, {self.max_value})'


class Integer(Samplable):
    def __init__(self, min_value: int, max_value: int):
        """min is inclusive, max is exclusive"""
        self.min_value = min_value
        self.max_value = max_value

    def sample(self):
        return np.random.randint(self.min_value, self.max_value)

    def to_ray_tune_search_dict(self):
        from ray import tune
        return tune.randint(self.min_value, self.max_value)

    def __repr__(self):
        return f'{self.__class__.__name__}({self.min_value}, {self.max_value})'


class Boolean(Samplable):
    def __init__(self, p_true: float):
        super().__init__()
        self.p_true = p_true

    def sample(self):
        return np.random.random() < self.p_true

    def __repr__(self):
        return f'{self.__class__.__name__}({self.p_true})'


class ProbabilityCapableParameter(Samplable):
    def __init__(self, possible_values: Union[List[Any], np.array], probs: Optional[Iterable[Any]] = None):
        super().__init__()
        self.possible_values = possible_values

        if probs:
            probs = np.array(list(probs))
            self.probs = list(probs / probs.sum())
        else:
            self.probs = None

    @classmethod
    def from_prob_dict(cls, d: Dict[Any, float]):
        return cls(list(d.keys()), probs=list(d.values()))

    def sample(self):
        return np.random.choice(self.possible_values, p=self.probs)

    def __repr__(self):
        return f'{self.__class__.__name__}({self.possible_values}, {self.probs})'


class Discrete(ProbabilityCapableParameter):
    def __init__(self, possible_values: Union[List[float], np.array], probs: Optional[Iterable[float]] = None):
        super().__init__(possible_values, probs)


class Categorical(ProbabilityCapableParameter):
    def __init__(self, possible_values: Union[List[Text], np.array], probs: Optional[Iterable[float]] = None):
        super().__init__(possible_values, probs)


if __name__ == '__main__':
    import numpy as np

    class ModelHyperParams(ParameterSet):
        num_hidden_layers = Integer(1, 4)
        num_neurons_per_layer = Discrete(np.logspace(2, 7, num=6, base=2, dtype=np.int))
        dropout_rate = Float(0.0, 0.99)
        activation = Categorical(['relu', 'sigmoid'])
        output_dir = '/tmp/output'
        filter_size = 3


    sample = ModelHyperParams().sample()
    print(sample)

    from chillpill import simple_argparse
    args = [
        '--num_hidden_layers=3',
        '--num_neurons_per_layer=2',
        '--dropout_rate=0.2',
        '--learning_rate=0.4',
        '--activation=relu',
        '--another_argument="also shows up"'
    ]

    params = ModelHyperParams.from_dict(simple_argparse.args_2_dict(args))
    print(params)
