import abc
import contextlib
import enum
import shutil
import time

from chillpill import params, packages
from google.cloud import storage
import io
from pathlib import Path
import tarfile
import tempfile
from typing import Text, Optional, Dict, Tuple, List, Callable, Any, Type, Union


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
    """Interface for objects which know how to modify a job_spec."""
    @abc.abstractmethod
    def modify_job_spec_inplace(self, job_spec: Dict):
        pass


class TrainFnPackageBasedRunInstructions(JobSpecModifier):
    """
    Instructions which create a hyperparameter search run based on a train function within a Python package.
    The training package gets a new module added which loads the passed command-line parameters into a parameter
    object and passes it to the train function.  See self._add_train_module_to_tarfile() for details.
    """
    TRAIN_MODULE_NAME = 'chillpill_added_train_module.py'
    SOURCE_TARFILE_NAME_TEMPLATE = 'source_{}.tar'

    def __init__(
            self,
            train_fn: Callable[[params.ParameterSet], None],
            train_params_type: Type[params.ParameterSet],
            cloud_staging_bucket: Text,
            cloud_project: Text,
            do_error_if_cloud_staging_bucket_does_not_exist=False,
            additional_package_root_dirs: Optional[List[Text]] = None,
            region: Text = 'us-central',
            runtime_version: Text = '1.13',
            python_version: Text = '3.5',
            verbose=True,
    ):
        """
        
        Args:
            train_fn:
                The function object which you use for training.  Should accept one parameter which is a  
                subclass of params.ParameterSet. 
            train_params_type: 
                The type of the parameters argument to your train_fn.  
                For instance, if your parameters are defined like so:
                    class MyParameters(params.ParameterSet):
                        ...
                    then you would pass MyParameters to this argument.
            cloud_staging_bucket:
                The bucket where you'd like your packages to be staged.  e.g.: 'my-bucket'. 
            cloud_project:
                The name of your gcloud project.
            do_error_if_cloud_staging_bucket_does_not_exist:
                If True, then throw an error if cloud_staging_bucket does not exist.
            additional_package_root_dirs:
                A list of additional package directories on this local machine which are necessary for your training
                application.  If packages are available on pip or publicly on github, just list them as requirements
                in your train package's setup.py.  Use this for packages which aren't available publicly.  Each package
                will be tarred up, uploaded to the staging bucket, and installed on the machine which runs your
                training application.
            region:
                The region in which you'd like to run your training application.  GPUs are not available in all
                regions.  See https://cloud.google.com/compute/docs/gpus/ for details.
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
        local_train_package_root_dir = packages.find_package_root(train_fn)
        train_function_import_string = packages.get_import_string_of_type(train_fn)
        train_params_type_import_string = packages.get_import_string_of_type(train_params_type)
        package_name = packages.get_package_name(train_fn)

        self.local_package_root_dir = local_train_package_root_dir
        if additional_package_root_dirs is None:
            additional_package_root_dirs = []
        self.num_source_dirs_uploaded = 0

        self.additional_package_root_dirs = additional_package_root_dirs
        self.train_function_import_string = train_function_import_string
        self.train_params_type_import_string = train_params_type_import_string
        self.package_name = package_name
        self.train_module_import_str = f'{self.package_name}.{str(Path(self.TRAIN_MODULE_NAME).stem)}'
        self.project = cloud_project
        self.region = region
        self.cloud_staging_bucket = cloud_staging_bucket
        self.runtime_version = runtime_version
        self.python_version = python_version
        self.verbose = verbose

        self._create_bucket_if_not_exists(
            bucket_name=self.cloud_staging_bucket,
            project=self.project,
            region=region,
            do_error_if_not_exist=do_error_if_cloud_staging_bucket_does_not_exist,
        )

    @staticmethod
    def _create_bucket_if_not_exists(
            bucket_name: Text,
            project: Text,
            region: Text = 'us-central1',
            do_error_if_not_exist=False,
    ):
        client = storage.Client()
        b = storage.Bucket(client, name=bucket_name)
        if not b.exists():
            if do_error_if_not_exist:
                raise ValueError(f"Bucket {bucket_name} does not exist.")
            b.create(client, project=project, location=region)

        return b

    def _create_and_upload_package_tar(
            self,
            local_package_root: Union[Text, Path],
            do_add_train_module=False,
            do_delete_tempdir=True,
    ):
        tarfile_path, tmp_dir_path = self._make_tarfile_of_source_directory(local_package_root)
        if self.verbose:
            print(f"Created tarfile, {str(tarfile_path)} of source code at {local_package_root}")
        if do_add_train_module:
            self._add_train_module_to_tarfile(tarfile_path)

        # upload tarfile
        if self.verbose:
            cloud_tarfile = str(Path(self.cloud_staging_bucket) / tarfile_path.name)
            print(f"Uploading tarfile from {str(tarfile_path)} to {cloud_tarfile}")
        cloud_tarfile_url = self._upload_file(tarfile_path)

        if do_delete_tempdir:
            if self.verbose:
                print(f"Removing temp directory: {str(tmp_dir_path)}")
            shutil.rmtree(tmp_dir_path)

        return cloud_tarfile_url

    def modify_job_spec_inplace(self, job_spec: Dict):
        # make tarfile
        package_uris = []
        package_url = self._create_and_upload_package_tar(self.local_package_root_dir, do_add_train_module=True)
        package_uris.append(package_url)

        # upload additional packages
        for additional_dir in self.additional_package_root_dirs:
            package_url = self._create_and_upload_package_tar(additional_dir, do_add_train_module=False)
            package_uris.append(package_url)

        # modify job_spec appropriately
        #   ref: https://cloud.google.com/ml-engine/reference/rest/v1/projects.jobs
        job_spec['trainingInput']['packageUris'] = package_uris
        job_spec['trainingInput']['pythonModule'] = self.train_module_import_str
        job_spec['trainingInput']['runtimeVersion'] = self.runtime_version
        job_spec['trainingInput']['pythonVersion'] = self.python_version

    def _make_tarfile_of_source_directory(self, local_source_root_dir: Text) -> Tuple[Path, Path]:
        tmp_dir_path = Path(tempfile.mkdtemp())
        self.num_source_dirs_uploaded += 1
        output_path = tmp_dir_path / self.SOURCE_TARFILE_NAME_TEMPLATE.format(self.num_source_dirs_uploaded)
        with tarfile.open(output_path, "w") as tar:
            def filter_fn(tar_info: tarfile.TarInfo):
                if tar_info.name.endswith('.egg-info') \
                        or tar_info.name.endswith('.pyc') \
                        or '__pycache__' in tar_info.name:
                    return None
                else:
                    return tar_info
            tar.add(local_source_root_dir, arcname='', filter=filter_fn)
        return output_path, tmp_dir_path

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
        with tarfile.open(str(tarfile_path), "a") as tarf:
            with contextlib.closing(io.BytesIO(contents.encode())) as fobj:
                filename = str(Path(self.package_name) / self.TRAIN_MODULE_NAME)
                tarinfo = tarfile.TarInfo(filename)
                tarinfo.size = len(fobj.getvalue())
                tarinfo.mtime = time.time()
                tarf.addfile(tarinfo, fileobj=fobj)

    def _upload_file(self, file_path: Path):
        # https://github.com/GoogleCloudPlatform/getting-started-python/blob/master/3-binary-data/bookshelf/storage.py
        """
        Uploads a file to a given Cloud Storage bucket and returns the public url
        to the new object.
        """
        client = storage.Client()
        bucket = client.bucket(self.cloud_staging_bucket)
        blob = bucket.blob(file_path.name)
        blob.upload_from_filename(str(file_path))

        package_location_on_cloud = f'gs://{str(Path(self.cloud_staging_bucket) / file_path.name)}'
        return package_location_on_cloud


class ContainerBasedRunInstructions(JobSpecModifier):
    def __init__(self, container_image_uri: Text):
        """A container which has your train module as it's CMD or ENTRYPOINT"""
        self.container_image_uri = container_image_uri

    def modify_job_spec_inplace(self, job_spec: Dict):
        job_spec['trainingInput']['masterConfig'] = {'imageUri': self.container_image_uri}


