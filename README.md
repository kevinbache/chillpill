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

You should implement your hyperparameters as a subclass of `ParameterSet`.

Alternatives
------------
```python
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

This is better than using a dict for your hyperparameters.
  1) Parameter values can be type checked in your training code (i.e.: you can be sure your parameter object has all the
     needed parameter values
  2) It's easy to rename hyperparameters by refactoring in your IDE
     (this fails with dicts that map parameter names to values)

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

Finally, Python 3.7's `dataclass`es provide perhaps the most similar analog to our `ParameterSet`s, but they pose two
problems which hold them back.
    1) They require type annotations along with default values.  While requiring type annotations is usually a good
       thing in Python, they're often redundant for `ParameterSet`s on which the types can be easily inferred from
       default values.
    2) They pose problems for subclassing[^bignote].

[^bignote]: For example:
    ```python
    from dataclasses import dataclass
    from typing import Any

    @dataclass
    class WithoutExplicitTypes:
        pass


    class Child(WithoutExplicitTypes):
        name: Any = 'asdf'
        value: Any = 42

    c = Child(name='ffffff', value=1234)
    print(c)
    ```

    Yields the following error: `TypeError: __init__() got an unexpected keyword argument 'name'`.  A (hyper)ParameterSet
    superclass should be able to be subclassed so that useful methods can be implemented in parents.

Overall, `PameterSet` let's your define your parameter classes with the minimum required amount of typing.



Authors
-------
`ritalin` was written by Kevin Bache.
