ritalin
=======
`ritalin` treats your hyperparameters with grace and aplomb.

Installation
------------
`$ pip install --editable .` from within this directory.

Usage
-----
Examples:
  1. Running `ritalin/examples/local_hp_tuning.py`
  2. Setting the env variables required in `ritalin/examples/cloud_complex_hp_tuning/build_and_submit.sh` and then running `ritalin/examples/cloud_complex_hp_tuning/run_tuning_job.py`

The main class you should interact with in `ritalin` is `params.ParameterSet` which represents a set of (hyper)parameters.

```
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
    from ritalin import params

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
    from ritalin import params, tuning

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
    search = HyperparamSearchSpec(
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

    search.add_parameters(my_param_ranges)
    search.to_training_input_yaml('hps.yaml')
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
    from ritalin import params
    from ritalin import simple_argparse
    params = ModelHyperParams.from_dict(simple_argparse.args_2_dict())
    assert(
      str(params) ==  \
      ModelHyperParams(activation: relu, dropout_rate: 0.2, filter_size: 3, num_hidden_layers: 2, num_neurons_per_layer: 2, output_dir: /tmp/output))
    )

    def build_model(params: ModelHyperParams):
        pass

    model = build_model(params)
```


Authors
-------
`ritalin` was written by Kevin Bache.
