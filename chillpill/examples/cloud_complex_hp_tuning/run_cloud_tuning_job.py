"""This module runs a distributed hyperparameter tuning job on Google Cloud AI Platform."""
import subprocess
from pathlib import Path

import numpy as np

from chillpill import params
from chillpill import search
from chillpill.examples.cloud_complex_hp_tuning import train

if __name__ == '__main__':
    # Create a Cloud AI Platform Hyperparameter Search object
    search = search.HyperparamSearchSpec(
        max_trials=10,
        max_parallel_trials=5,
        max_failed_trials=2,
        hyperparameter_metric_tag='val_acc',
    )

    # Add parameter search ranges for this problem.
    my_param_ranges = train.MyParams(
        activation=params.CategoricalParameter(['relu', 'tanh']),
        num_layers=params.IntegerParameter(min_value=1, max_value=3),
        num_neurons=params.DiscreteParameter(np.logspace(2, 8, num=7, base=2)),
        dropout_rate=params.DoubleParameter(min_value=-0.1, max_value=0.9),
        learning_rate=params.DiscreteParameter(np.logspace(-6, 2, 17, base=10)),
        batch_size=params.IntegerParameter(min_value=1, max_value=128),
    )
    search.add_parameters(my_param_ranges)

    this_dir = Path(__file__).resolve().parent

    # Dump search spec and parameter ranges to a yaml file.
    search.to_training_input_yaml(this_dir / 'hps.yaml')

    # Call a bash script to build a docker image for this repo, submit it to the docker registry defined in the script
    # and run a training job on the Cloud AI Platform using this container and these hyperparameter ranges.
    subprocess.call([this_dir / 'build_submit_run.sh'])
