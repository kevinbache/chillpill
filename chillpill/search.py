"""This module implements Google Cloud's Hyperparameter Spec YAML as a set of Python objects.
See https://cloud.google.com/ml-engine/reference/rest/v1/projects.jobs#hyperparameterspec
"""
import abc
import enum
import time
from typing import Text, Optional, Union, List, Dict, Callable, Any, Type
import yaml

import numpy as np

from chillpill import params, gcloud_run_instructions


def snake_2_camel(name):
    words = name.split('_')
    return words[0] + ''.join(w.capitalize() for w in words[1:])


def convert_keys(obj: Any, converter: Callable):
    if isinstance(obj, list):
        return [convert_keys(e, converter) for e in obj]
    elif isinstance(obj, dict):
        return {converter(k): convert_keys(v, converter) for k, v in obj.items()}
    else:
        return obj


def convert_keys_to_camel(d: Dict) -> Dict:
    return convert_keys(d, snake_2_camel)


def _numpy_scalar_to_python_scalar(number: np.number):
    if float(number).is_integer():
        return int(number)
    else:
        return float(number)


class ParameterSearchGoalType(enum.Enum):
    MAXIMIZE = 0,
    MINIMIZE = 1


class SearchAlgorithm(enum.Enum):
    ALGORITHM_UNSPECIFIED = 0,
    GRID_SEARCH = 1,
    RANDOM_SEARCH = 2


class ScaleType(enum.Enum):
    """
        From https://cloud.google.com/ml-engine/reference/rest/v1/projects.jobs#ScaleType
        NONE	                By default, no scaling is applied.
        UNIT_LINEAR_SCALE	    Scales the feasible space to (0, 1) linearly.
        UNIT_LOG_SCALE	        Scales the feasible space logarithmically to (0, 1).
                                The entire feasible space must be strictly positive.
        UNIT_REVERSE_LOG_SCALE	Scales the feasible space "reverse" logarithmically to (0, 1).
                                The result is that values close to the top of the feasible space are spread out
                                more than points near the bottom. The entire feasible space must be strictly positive.
    """
    NONE = 0
    UNIT_LINEAR_SCALE = 1
    UNIT_LOG_SCALE = 2
    UNIT_REVERSE_LOG_SCALE = 3


class HyperparamSearchSpec(params.HasClassDefaults):
    """A Pythonic wrapper around Google Cloud's HyperParameterSpec YAML format."""
    def __init__(
            self,
            max_trials: int = 10,
            max_parallel_trials: Optional[int] = 2,
            max_failed_trials: Optional[int] = 0,
            hyperparameter_metric_tag: Text = 'val_acc',
            goal=ParameterSearchGoalType.MAXIMIZE,
            resume_previous_job_id: Optional[Text] = None,
            enable_trial_early_stopping: Optional[bool] = True,
            algorithm=SearchAlgorithm.ALGORITHM_UNSPECIFIED,
    ):
        self.goal = goal
        self.hyperparameter_metric_tag = hyperparameter_metric_tag
        self.max_trials = max_trials
        self.max_parallel_trials = max_parallel_trials
        self.max_failed_trials = max_failed_trials
        self.resume_previous_job_id = resume_previous_job_id
        self.enable_trial_early_stopping = enable_trial_early_stopping
        self.algorithm = algorithm
        self.params = []

    ###########################################
    # low level methods which expose ScaleType
    def _add_integer_parameter(self, parameter_name: Text, min_value: int, max_value: int, scale_type=ScaleType.NONE):
        self.params.append(IntegerSpecParameter(parameter_name, min_value, max_value, scale_type))

    def _add_double_parameter(self, name: Text, min_value: float, max_value: float, scale_type=ScaleType.NONE):
        self.params.append(DoubleSpecParameter(name, min_value, max_value, scale_type))

    def _add_discrete_parameter(self, name: Text, possible_values: Union[List[float], np.array]):
        self.params.append(DiscreteSpecParameter(name, possible_values))

    def _add_categorical_parameter(self, name: Text, possible_values: List[Text]):
        self.params.append(CategoricalSpecParameter(name, possible_values))

    ############################################################
    # high level methods which work with hp.SamplableParameters
    def _add_parameter(self, name: Text, parameter: params.SamplableParameter):
        self.params.append(SpecParameterFactory.spec_param_from_param(name, parameter))

    def add_parameters(self, parameters: params.ParameterSet):
        for name, attribute in parameters.__dict__.items():
            if isinstance(attribute, params.SamplableParameter):
                self._add_parameter(name, attribute)

    @classmethod
    def _to_dict_inner(cls, obj: Any) -> Any:
        if isinstance(obj, list):
            return [cls._to_dict_inner(e) for e in obj]
        elif isinstance(obj, dict):
            return {k: cls._to_dict_inner(v) for k, v in obj.items()}
        elif isinstance(obj, SpecParameter):
            return obj.to_dict()
        elif isinstance(obj, enum.Enum):
            return obj.name
        else:
            return obj

    def _to_dict(self) -> Dict:
        d = self._to_dict_inner(self.__dict__)
        return d

    def to_training_input_yaml(self, filename=None) -> Text:
        s = yaml.dump(self.to_training_input_dict())
        if filename is not None:
            with open(filename, 'w') as f:
                f.write(s)

        return s

    def to_training_input_dict(self) -> Dict:
        return {
            'trainingInput': {
                'hyperparameters': convert_keys_to_camel(self._to_dict()),
            }
        }

    def run_from_trian_fn(
            self,
            train_fn: Callable[[params.ParameterSet], None],
            train_params_type: Type[params.ParameterSet],
            cloud_staging_bucket: Text,
            gcloud_project_name: Text,
            do_error_if_cloud_staging_bucket_does_not_exist=False,
            additional_package_root_dirs: Optional[List[Text]] = None,
            runtime_version: Text = '1.13',
            python_version: Text = '3.5',
            job_name: Optional[Text] = None,
            static_args: Optional[Dict] = None,
            region: Text = 'us-central1',
            verbose=True,
    ):
        ri = gcloud_run_instructions.TrainFnPackageBasedRunInstructions(
            train_fn=train_fn,
            train_params_type=train_params_type,
            cloud_staging_bucket=cloud_staging_bucket,
            cloud_project=gcloud_project_name,
            do_error_if_cloud_staging_bucket_does_not_exist=do_error_if_cloud_staging_bucket_does_not_exist,
            additional_package_root_dirs=additional_package_root_dirs,
            runtime_version=runtime_version,
            python_version=python_version,
            region=region,
            verbose=verbose,
        )
        return self._run_job_from_run_instructions(
            gcloud_project_name=gcloud_project_name,
            run_instructions=ri,
            job_name=job_name,
            static_args=static_args,
            region=region,
        )

    def run_from_container(
            self,
            container_image_uri: Text,
            gcloud_project_name: Text,
            job_name: Optional[Text] = None,
            static_args: Optional[Dict] = None,
            region: Text = 'us-central1',
    ):
        ri = gcloud_run_instructions.ContainerBasedRunInstructions(container_image_uri=container_image_uri)
        self._run_job_from_run_instructions(
            run_instructions=ri,
            gcloud_project_name=gcloud_project_name,
            job_name=job_name,
            static_args=static_args,
            region=region,
        )

    def _run_job_from_run_instructions(
            self,
            run_instructions: gcloud_run_instructions.JobSpecModifier,
            gcloud_project_name: Text,
            job_name: Optional[Text]=None,
            static_args: Optional[Dict]=None,
            region: Text='us-central1',
    ):
        # https://cloud.google.com/ml-engine/docs/tensorflow/training-jobs
        # https://cloud.google.com/ml-engine/reference/rest/v1/projects.jobs#TrainingInput

        if job_name is None:
            job_name = f'my_job_{str(int(time.time()))}'

        job_spec = self.to_training_input_dict()
        job_spec['job_id'] = job_name
        job_spec['trainingInput']['region'] = region

        # if this is a container-based run, just add the container uri
        # if this is a package-based run, tar up the package, add the runner module, upload the package,
        # point the job_spec toward the package
        run_instructions.modify_job_spec_inplace(job_spec)

        if static_args:
            job_spec['trainingInput']['args'] = [f'--{k}={v}' for k, v in static_args.items()]

        project_id = f'projects/{gcloud_project_name}'

        print(job_spec)

        from googleapiclient import discovery
        cloudml = discovery.build('ml', 'v1')
        request = cloudml.projects().jobs().create(body=job_spec, parent=project_id)

        response = request.execute()

        msg = f'''
        Job [{job_name}] submitted successfully.
        Your job is still active. You may view the status of your job with the command

          $ gcloud ai-platform jobs describe {job_name}

        or continue streaming the logs with the command

          $ gcloud ai-platform jobs stream-logs {job_name}

        You can see the results of your hyperparameter optimization at: 
          https://console.cloud.google.com/mlengine/jobs/{job_name}/
        '''
        print(msg)

        return response


class SearchType(enum.Enum):
    ALGORITHM_UNSPECIFIED = 0,
    GRID_SEARCH = 1,
    RANDOM_SEARCH = 2


class ParameterRangeType(enum.Enum):
    DOUBLE = 0
    INTEGER = 1
    DISCRETE = 2
    CATEGORICAL = 3


class SpecParameter(params.HasClassDefaults):
    """Represents a parameter in a HyperparameterSearchSpec"""
    def __init__(
            self,
            parameter_name: Text,
            type: ParameterRangeType,
            scale_type=ScaleType.NONE
    ):
        self.parameter_name = parameter_name
        self.type = type
        self.scale_type = scale_type

    def to_dict(self):
        # TODO: _get_member_names can probably be __dict__
        d = {k: self.__getattribute__(k) for k in self._get_member_names()}
        if d['scale_type'] == ScaleType.NONE:
            del d['scale_type']
        d['type'] = d['type'].name
        return d

    @classmethod
    @abc.abstractmethod
    def from_hyperparameter(cls, name: Text, parameter: params.SamplableParameter):
        raise NotImplementedError("Don't instantiate this class directly.  Use it's children.")


class DoubleSpecParameter(SpecParameter):
    def __init__(
            self,
            parameter_name: Text,
            min_value: float,
            max_value: float,
            scale_type=ScaleType.NONE,
    ):
        super().__init__(parameter_name, ParameterRangeType.DOUBLE, scale_type)
        self.max_value = max_value
        self.min_value = min_value

    def sample(self):
        if self.scale_type is not ScaleType.NONE:
            raise NotImplementedError("sample is only implemented for ScaleType.NONE")
        return np.random.uniform(self.min_value, self.max_value)

    @classmethod
    def from_hyperparameter(cls, name: Text, parameter: params.Double):
        return cls(
            parameter_name=name,
            min_value=parameter.min_value,
            max_value=parameter.max_value,
        )


class IntegerSpecParameter(SpecParameter):
    def __init__(
            self,
            parameter_name: Text,
            min_value: int,
            max_value: int,
            scale_type=ScaleType.NONE,
    ):
        super().__init__(parameter_name, ParameterRangeType.INTEGER, scale_type)
        self.max_value = max_value
        self.min_value = min_value

    def sample(self):
        if self.scale_type is not ScaleType.NONE:
            raise NotImplementedError("sample is only implemented for ScaleType.NONE")
        return np.random.randint(self.min_value, self.max_value)

    @classmethod
    def from_hyperparameter(cls, name: Text, parameter: params.Integer):
        return cls(
            parameter_name=name,
            min_value=parameter.min_value,
            max_value=parameter.max_value,
        )


class DiscreteSpecParameter(SpecParameter):
    def __init__(
            self,
            parameter_name: Text,
            discrete_values: Union[List[float], np.array],
    ):
        super().__init__(parameter_name, ParameterRangeType.DISCRETE)
        if not self._is_sorted(discrete_values):
            discrete_values = sorted(discrete_values)

        if not self._is_sorted(discrete_values):
            raise ValueError("Values must be separated by at least 1e-10.")

        self.discrete_values = discrete_values

    @staticmethod
    def _is_sorted(l):
        return all(l[i] <= l[i + 1] for i in range(len(l) - 1))

    @staticmethod
    def _is_separated(l):
        return all(l[i + 1] - l[i] > 1e-10 for i in range(len(l) - 1))

    def sample(self):
        return np.random.choice(self.discrete_values)

    @classmethod
    def from_hyperparameter(cls, name: Text, parameter: params.Discrete):
        return cls(
            parameter_name=name,
            discrete_values=list([_numpy_scalar_to_python_scalar(e) for e in parameter.possible_values]),
        )


class CategoricalSpecParameter(SpecParameter):
    def __init__(
            self,
            parameter_name: Text,
            categorical_values: List[Text],
    ):
        super().__init__(parameter_name, ParameterRangeType.CATEGORICAL)
        self.categorical_values = categorical_values

    def sample(self):
        return np.random.choice(self.categorical_values)

    @classmethod
    def from_hyperparameter(cls, name: Text, parameter: params.Categorical):
        return cls(
            parameter_name=name,
            categorical_values=parameter.possible_values,
        )


class SpecParameterFactory:
    """Translates hp.SamplableParameters into SpecParameters"""
    hp_to_spec_types = {
        params.Integer: IntegerSpecParameter,
        params.Double: DoubleSpecParameter,
        params.Discrete: DiscreteSpecParameter,
        params.Categorical: CategoricalSpecParameter,
    }

    @classmethod
    def spec_param_from_param(cls, name: Text, parameter: params.SamplableParameter):
        return cls.hp_to_spec_types[parameter.__class__].from_hyperparameter(name, parameter)


if __name__ == '__main__':
    search = HyperparamSearchSpec(
        max_trials=10,
        max_parallel_trials=5,
        max_failed_trials=2,
        hyperparameter_metric_tag='val_acc',
    )

    class ModelHyperParams(params.ParameterSet):
        num_hidden_layers = params.Integer(1, 4)
        num_neurons_per_layer = params.Discrete(np.logspace(2, 7, num=6, base=2, dtype=np.int))
        dropout_rate = params.Double(0.0, 0.99)
        activation = params.Categorical(['relu', 'sigmoid'])
        output_dir = '/tmp/output'
        filter_size = 3
    search.add_parameters(ModelHyperParams())
    print(search.to_training_input_dict())

    run_instructions = \
        gcloud_run_instructions.ContainerBasedRunInstructions('gcr.io/kb-experiment/chillpill:cloud_hp_tuning_example')

    search._run_job_from_run_instructions(
        job_name=f'my_job_{str(int(time.time()))}',
        run_instructions=run_instructions,
        gcloud_project_name='kb-experiment',
        # static_args={'bucket_id': 'kb-bucket'}
    )
