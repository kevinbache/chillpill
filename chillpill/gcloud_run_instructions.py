import abc
import enum
from google.cloud import storage
from io import StringIO
import os
from pathlib import Path
import six
import tarfile
import tempfile
from typing import Text, Optional, Dict


class MachineType:
    """https://cloud.google.com/ml-engine/reference/rest/v1/projects.jobs#ReplicaConfig"""
    pass


class PreconfiguredMachineType(MachineType, enum.Enum):
    """see https://cloud.google.com/ml-engine/reference/rest/v1/projects.jobs#ReplicaConfig"""
    standard = 0
    large_model = 1
    complex_model_s = 2
    complex_model_m = 3
    complex_model_l = 4
    standard_gpu = 5
    complex_model_m_gpu = 6
    complex_model_l_gpu = 7
    standard_p100 = 8
    complex_model_m_p100 = 9
    standard_v100 = 10
    large_model_v100 = 11
    complex_model_m_v100 = 12
    complex_model_l_v100 = 13
    cloud_tpu = 14

    def __str__(self):
        return self.name


class GceMachineType(enum.Enum):
    n1standard4 = ('n1-standard-4', 0)
    n1standard8 = ('n1-standard-8', 1)
    n1standard16 = ('n1-standard-16', 2)
    n1standard32 = ('n1-standard-32', 3)
    n1standard64 = ('n1-standard-64', 4)
    n1standard96 = ('n1-standard-96', 5)
    n1highmem2 = ('n1-highmem-2', 6)
    n1highmem4 = ('n1-highmem-4', 7)
    n1highmem8 = ('n1-highmem-8', 8)
    n1highmem16 = ('n1-highmem-16', 9)
    n1highmem32 = ('n1-highmem-32', 10)
    n1highmem64 = ('n1-highmem-64', 11)
    n1highmem96 = ('n1-highmem-96', 12)
    n1highcpu16 = ('n1-highcpu-16', 13)
    n1highcpu32 = ('n1-highcpu-32', 14)
    n1highcpu64 = ('n1-highcpu-64', 15)
    n1highcpu96 = ('n1-highcpu-96', 16)

    def __init__(self, name, id):
        self._name = name
        self.id = id

    def __str__(self):
        return self._name


class AcceleratorType(enum.Enum):
    """ref: https://cloud.google.com/ml-engine/reference/rest/v1/AcceleratorType"""
    ACCELERATOR_TYPE_UNSPECIFIED = 0
    NVIDIA_TESLA_K80 = 1
    NVIDIA_TESLA_P100 = 2
    NVIDIA_TESLA_V100 = 3
    NVIDIA_TESLA_P4 = 4
    NVIDIA_TESLA_T4 = 5
    TPU_V2 = 6


class AcceleratorConfig:
    def __init__(self, accelerator_type: AcceleratorType, accelerator_count: int):
        self.accelerator_type = accelerator_type
        self.accelerator_count = accelerator_count


class GceMachineWithAccelerators(MachineType):
    """Not all accelerator configs work with all machine types.
    See https://cloud.google.com/ml-engine/docs/tensorflow/using-gpus#compute-engine-machine-types-with-gpu for details.
    """
    def __init__(self, gce_machine_type: GceMachineType, accelerator_config: Optional[AcceleratorConfig]):
        self.gce_machine_type = gce_machine_type
        self.accelerator_config = accelerator_config


class JobSpecModifier(abc.ABC):
    @abc.abstractmethod
    def modify_job_spec_inplace(self, job_spec: Dict):
        pass


class TrainFnPackageBasedRunInstructions(JobSpecModifier):
    TRAIN_MODULE_NAME = 'chillpill_added_train_module.py'
    SOURCE_TARFILE_NAME = 'source.tar.gz'

    def __init__(
            self,
            local_source_root_dir: Text,
            train_function_import_string: Text,
            train_params_type_import_string: Text,
            cloud_staging_bucket: Text,
            runtime_version: Text = '1.13',
            python_version: Text = '3.5',
            verbose=True,
    ):
        """

        Args:
            local_source_root_dir:
                The root of your local source code files.  This directory should contain a Python package, meaning it
                has a setup.py file and your package subdirectory.
                Example structure:
                    local_source_root_dir/
                        setup.py
                        my_package/
                            model.py
                            train.py
                            etc.
            train_function_import_string:
                This should be the path to your training function.  Your training function should accept one input
                which should be a subclass of chillpill.params.ParameterSet (and which is referenced by the
                train_params_type_import_string parameter below).
                e.g.: 'my_package.path.to.train.function.my_train_fn'
            train_params_type_import_string:
                This should be the path to the parameter class which is the type that's passed to your train function.
                This is used for recreating the training params from the command line arguments that CMLE passes to
                the train module.  MyParams should be a subclass of chillpill.params.ParameterSet.
                e.g.: 'my_package.path.to.params.definition.MyParams'
            cloud_staging_bucket:
                The cloud bucket to which to upload the training package.
            runtime_version:
                CMLE runtime version.
                This determines package verions which will be available to your training application.
                See https://cloud.google.com/ml-engine/docs/tensorflow/runtime-version-list
            python_version:
                Which python version to use to run your module.  Since chillpill requires Python 3.5 you probably
                should too.
            verbose:
                If True, print some status messages as you go.
        """
        self.local_package_root_dir = local_source_root_dir
        self.train_function_import_string = train_function_import_string
        self.train_params_type_import_string = train_params_type_import_string
        self.cloud_bucket_path = cloud_staging_bucket
        self.runtime_version = runtime_version
        self.python_version = python_version
        self.verbose = verbose

    def modify_job_spec_inplace(self, job_spec: Dict):
        # make tarfile
        tarfile_path = self._make_tarfile_of_source_directory(self.local_package_root_dir)
        if self.verbose:
            print(f"Created tarfile, {str(tarfile_path)} of source code at {self.local_package_root_dir}")
        self._add_train_module_to_tarfile(tarfile_path)

        # upload tarfile
        if self.verbose:
            cloud_tarfile = str(Path(self.cloud_bucket_path) / tarfile_path.name)
            print(f"Uploading tarfile from {str(tarfile_path)} to {cloud_tarfile}")
        cloud_tarfile_url = self._upload_file(tarfile_path)

        # modify job_spec appropriately
        job_spec['trainingInput']['packageUris'] = [cloud_tarfile_url]
        job_spec['trainingInput']['runtimeVersion'] = self.runtime_version
        job_spec['trainingInput']['pythonVersion'] = self.python_version

    def _make_tarfile_of_source_directory(self, local_source_root_dir: Text) -> Path:
        with tempfile.TemporaryDirectory() as tempdir:
            output_path = Path(tempdir) / self.SOURCE_TARFILE_NAME
            with tarfile.open(output_path, "w:gz") as tar:
                tar.add(local_source_root_dir, arcname=os.path.basename(local_source_root_dir))
        return output_path

    def _add_train_module_to_tarfile(self, tarfile_path: Path):
        # https://stackoverflow.com/questions/2239655/how-can-files-be-added-to-a-tarfile-with-python-without-adding-the-directory-hi'''

        contents = f"""
from typing import Text
from importlib import import_module
from chillpill import simple_argparse


def get_obj(import_str: Text):
    module_import_str, obj_name = import_str.rsplit('.', 1)
    module = import_module(module_import_str)
    obj = getattr(module, obj_name)
    return obj


if __name__ == '__main__':
    train_fn_import_str = '{self.train_function_import_string}'
    train_param_type_import_str = '{self.train_params_type_import_string}'

    train_fn = get_obj(train_fn_import_str)
    train_param_type = get_obj(train_param_type_import_str)

    # get a parameter dictionary
    hp = train_param_type.from_dict(simple_argparse.args_2_dict())
    print(hp)
    train_fn(hp)
"""
        with tarfile.open(tarfile_path, "w|gz") as tar:
            # todo: stick inside directory
            tar.addfile(tarfile.TarInfo(self.TRAIN_MODULE_NAME), StringIO.StringIO(contents))

    def _upload_file(self, file_path: Path):
        # https://github.com/GoogleCloudPlatform/getting-started-python/blob/master/3-binary-data/bookshelf/storage.py
        """
        Uploads a file to a given Cloud Storage bucket and returns the public url
        to the new object.
        """
        client = storage.Client()
        bucket = client.bucket(self.cloud_bucket_path)
        blob = bucket.blob(file_path.name)
        blob.upload_from_filename(str(file_path))

        url = blob.public_url

        if isinstance(url, six.binary_type):
            url = url.decode('utf-8')

        return url


class ContainerBasedRunInstructions(JobSpecModifier):
    def __init__(self, container_image_uri: Text):
        """A container which has your train module as it's CMD or ENTRYPOINT"""
        self.container_image_uri = container_image_uri

    def modify_job_spec_inplace(self, job_spec: Dict):
        job_spec['trainingInput']['masterConfig'] = {'imageUri': self.container_image_uri}


