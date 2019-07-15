chillpill
=======
`chillpill` offers a surprisingly powerful implementation of (Hyper)`ParameterSet`s which requires the minimal 
possible amount of work.

Installation
------------
`$ pip install --editable .` from within this directory.


Examples
--------
See https://github.com/kevinbache/chillpill_examples for complete working examples of
1. [Local hyperparameter tuning](https://github.com/kevinbache/chillpill_examples/tree/master/chillpill_examples/local_hp_tuning)
2. [Cloud-Based Tuning from Train Function](https://github.com/kevinbache/chillpill_examples/tree/master/chillpill_examples/cloud_hp_tuning_from_train_fn)  
3. [Cloud-Based Tuning from Container](https://github.com/kevinbache/chillpill_examples/tree/master/chillpill_examples/cloud_hp_tuning_from_container)  


Usage
-----
Define your (hyper)ParameterSet classes by inheriting from `params.ParameterSet` and defining default values as 
class members.
```python
from chillpill import params

class ModelHyperParams(params.ParameterSet):
    filter_size = 3
    num_hidden_layers = 2
    num_neurons_per_layer = 32
    dropout_rate = 0.5
    activation = 'relu'
    output_dir = '/tmp/output'
```
 
The `params.ParameterSet`'s `__init__` method will copy these class members into any instantiated subclasses so they
won't be shared between instantiated objects.   

`ParameterSet`s can also be instantiated with parameter ranges for conducting (hyper)parameter searches.

For example, continuing the code from above:

```python
my_param_ranges = ModelHyperParams(
    filter_size=params.DiscreteParameter([3, 5, 7]),
    num_hidden_layers=params.IntegerParameter(min_value=1, max_value=3),
    num_neurons_per_layer=params.DiscreteParameter(np.logspace(2, 8, num=7, base=2)),
    dropout_rate=params.DoubleParameter(min_value=-0.1, max_value=0.9),
    activation = 'relu',
)
```

Reusing the same class for defining (hyper)Parameter search ranges and default definitions means that you
can refactor a parameter name (e.g. `filter_size` above) one time in your IDE and it changes everywhere you 
touch it in your code.

These parameter ranges can be sampled:
```python
hps = my_param_ranges.sample()
print(hp)
```

Which yields a fully instantiated version of your hyperparameters.
`ModelHyperParams(activation: relu, dropout_rate: 0.6031449000058329, filter_size: 3, num_hidden_layers: 2, num_neurons_per_layer: 32, output_dir: /tmp/output)`

For local tuning of Keras models, check out the `KerasHistoryRandomTuner`.  For example:
```python    
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

See https://github.com/kevinbache/chillpill_examples/tree/master/chillpill_examples/local_hp_tuning for a complete example of local hyperparameter tuning for Keras models.

`ParameterSet`s can also be used in conjunction with `search.HyperparamSearchSpec` to conduct full hyperparameter 
searches using Google's Cloud Machine Learning Engine.  For example: 

```python
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

This creates `hps.yaml`, a file like this:
```yaml
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
which can be used for 
[distributed hyperparameter tuning](https://cloud.google.com/blog/products/gcp/hyperparameter-tuning-cloud-machine-learning-engine-using-bayesian-optimization) 
using Google Cloud Machine Learning Engine.  See `chillpill/examples/cloud_complex_hp_tuning/run_tuning_job.py` 
for a complete example.  

Finally, you can rebuild this (Hyper)`ParameterSet` object on the cluster using `chillpill/simple_argparse.py` module and 
`ParameterSet.from_dict()`.  For example, in a python script invoked with the arguments:
```bash
python train.py \
    --num_hidden_layers=3 \
    --num_neurons_per_layer=2 \
    --dropout_rate=0.2 \
    --learning_rate=0.4 \
    --activation=relu
```
You could instantiate the (Hyper)`ParameterSet` class you defined above like so:

```
from chillpill import params
from chillpill import simple_argparse
params = ModelHyperParams.from_dict(simple_argparse.args_2_dict())

def build_model(params: ModelHyperParams):
    pass

model = build_model(params)
...
```

That's the basic idea of chillpill.  One (Hyper)`ParameterSet` class which does everything you need with a minimal amount 
of typing and refactoring headaches.

Alternatives
------------
`ParameterSet` (Hyper)Parameter classes are the best: better than dict-based hyperparameter sets, 
traditional classes, and dataclasses.

### Dicts
This is better than using a dict for your hyperparameters.
1)  Parameter values can be type checked in your training code (i.e.: you can be sure your parameter object has all the
    needed parameter values
2)  It's easy to rename hyperparameters by refactoring in your IDE
    (this fails with dicts that map parameter names to values)

### Traditional Classes
It's also better than defining a custom Hyperparameter class.
Take this for example:

```python
class ModelHyperParams:
    def __init__(
            self,
            filter_size = 3,
            num_hidden_layers = 2,
            num_neurons_per_layer = 32,
            dropout_rate = 0.5,
            activation = 'relu',
            output_dir = '/tmp/output'
    ):
        self.filter_size = filter_size
        self.num_hidden_layers = num_hidden_layers
        self.num_neurons_per_layer = num_neurons_per_layer
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.output_dir = output_dir
```

This is fine but you had to write out every parameter name three times and if you want to change one you've got to
change it in three places.

### Dataclasses
Finally, Python 3.7's `dataclass`es provide perhaps the most similar analog to our `ParameterSet`s, but they pose two
problems which hold them back.
1) They require Python 3.7.
2) They require type annotations along with default values.  While requiring type annotations is usually a good
       thing in Python, they're often redundant for `ParameterSet`s on which the types can be easily inferred from
       default values.  This makes for unnecessary extra typing.
3) They pose problems for subclassing.  For example:
```python
from dataclasses import dataclass
from typing import Any

@dataclass
class Parent:
    pass

class Child(Parent):
    name: Any = 'asdf'
    value: Any = 42

c = Child(name='ffffff', value=1234)
print(c)
```

Yields the following error: `TypeError: __init__() got an unexpected keyword argument 'name'`.  A (hyper)ParameterSet
superclass should be able to be subclassed so that useful methods can be implemented in parents.

Summary
-------
Overall, `ParameterSet`s offer (Hyper)Parameter objects which are powerful but still as simple to use as it is 
possible to be in python.

Authors
-------
`chillpill` was written by Kevin Bache.
