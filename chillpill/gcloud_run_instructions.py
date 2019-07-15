import abc
import contextlib
import enum
import time

from google.cloud import storage
import io
import os
from pathlib import Path
import shutil
import tarfile
import tempfile
from typing import Text, Optional, Dict, Tuple, List


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


# 'https://cloud.google.com/storage/docs/access-control/using-iam-permissions#storage-add-bucket-iam-python'
def add_bucket_iam_member(bucket_name, roles, member):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    policy = bucket.get_iam_policy()

    for role in roles:
        policy[role].add(member)

    bucket.set_iam_policy(policy)

    print('Added {} with role {} to {}.'.format(
        member, role, bucket_name))


# 'https://cloud.google.com/storage/docs/access-control/using-iam-permissions#storage-add-bucket-iam-python'
# 'https://github.com/GoogleCloudPlatform/python-docs-samples/blob/master/storage/cloud-client/acl.py'
def print_bucket_roles(bucket_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    iam_policies = bucket.get_iam_policy()

    for iam_policy in iam_policies:
        members = iam_policies[iam_policy]
        print('Policy Role: {}, Members: {}'.format(iam_policy, members))

    for acl_entry in bucket.acl:
        print('Acl Role: {}, Entity: {}'.format(acl_entry['role'], acl_entry['entity']))


def get_cloud_ml_service_account(project):
    from googleapiclient import discovery
    cloudml = discovery.build('ml', 'v1')
    r = cloudml.projects().getConfig(name=f'projects/{project}').execute()
    return r['serviceAccount']


# create bucket
# 'https://github.com/GoogleCloudPlatform/python-docs-samples/blob/master/storage/cloud-client/snippets.py'
def create_bucket(bucket_name):
    """Creates a new bucket."""
    storage_client = storage.Client()
    bucket = storage_client.create_bucket(bucket_name)
    print('Bucket {} created'.format(bucket.name))


# https://stackoverflow.com/questions/42576366/google-cloud-storage-python-api-create-bucket-in-specified-location
def create_bucket_in_location(bucket_name, location):
    b = storage.Bucket(storage.Client(), name=bucket_name)
    b.location = location
    b.create()


def create_bucket_if_not_exists(bucket_name: Text, project: Text, region: Text ='us-central1'):
    client = storage.Client()
    b = storage.Bucket(client, name=bucket_name)
    if not b.exists():
        b.create(client, project=project, location=region)

    # bucket = client.get_bucket(bucket_name)
    # if not bucket.exists():
    #     bucket = client.create_bucket(bucket_name)
    # bucket.location = region

    # # https://cloud.google.com/storage/docs/access-control/create-manage-lists#storage-set-acls-python
    # # Reload fetches the current ACL from Cloud Storage.
    # b.acl.reload()
    # # You can also use `group`, `domain`, `all_authenticated` and `all` to
    # # grant access to different types of entities. You can also use
    # # `grant_read` or `grant_write` to grant different roles.
    # b.default_object_acl.user(owner_email).grant_owner()
    # ml_service_email = get_cloud_ml_service_account(project)
    # b.default_object_acl.user(ml_service_email).grant_owner()
    # b.default_object_acl.save()

    return b


class TrainFnPackageBasedRunInstructions(JobSpecModifier):
    TRAIN_MODULE_NAME = 'chillpill_added_train_module.py'
    SOURCE_TARFILE_NAME_TEMPLATE = 'source_{}.tar'

    def __init__(
            self,
            local_train_package_root_dir: Text,
            train_function_import_string: Text,
            train_params_type_import_string: Text,
            cloud_staging_bucket: Text,
            # cloud_staging_bucket_owner_email: Text,
            package_name: Text,
            project: Text,
            additional_package_root_dirs: Optional[List[Text]]=None,
            region: Text='us-central',
            runtime_version: Text = '1.13',
            python_version: Text = '3.5',
            verbose=True,
    ):
        """

        Args:
            local_train_package_root_dir:
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
        self.local_package_root_dir = local_train_package_root_dir
        if additional_package_root_dirs is None:
            additional_package_root_dirs = []
        self.num_source_dirs_uploaded = 0

        self.additional_package_root_dirs = additional_package_root_dirs
        self.train_function_import_string = train_function_import_string
        self.train_params_type_import_string = train_params_type_import_string
        self.package_name = package_name
        self.train_module_import_str = f'{self.package_name}.{str(Path(self.TRAIN_MODULE_NAME).stem)}'
        self.project = project
        self.region = region
        self.cloud_staging_bucket = cloud_staging_bucket
        # self.cloud_staging_bucket_owner_email = cloud_staging_bucket_owner_email
        self.runtime_version = runtime_version
        self.python_version = python_version
        self.verbose = verbose

        # 'https://cloud.google.com/ml-engine/docs/tensorflow/packaging-trainer'
        #     '''
        #     --staging-bucket specifies the Cloud Storage location where you want to stage your training and
        #     dependency packages. Your GCP project must have access to this Cloud Storage bucket, and the bucket
        #     should be in the same region that you run the job. See the available regions for AI Platform services.
        #     If you don't specify a staging bucket, AI Platform stages your packages in the location specified in
        #     the job-dir parameter.
        #     '''
        create_bucket_if_not_exists(
            bucket_name=self.cloud_staging_bucket,
            # owner_email=self.cloud_staging_bucket_owner_email,
            project=self.project,
            region=region
        )

        # create_bucket(self.cloud_staging_bucket)
        # create_bucket_in_location(self.cloud_staging_bucket, location=region)

        print_bucket_roles(self.cloud_staging_bucket)
        roles = [
            'roles/storage.legacyObjectReader',
            'roles/storage.legacyObjectOwner',
            'roles/storage.legacyBucketReader',
            'roles/storage.legacyBucketWriter',
            'roles/storage.legacyBucketOwner',
        ]
        add_bucket_iam_member(
            self.cloud_staging_bucket,
            roles,
            f"serviceAccount:{get_cloud_ml_service_account(self.project)}")
        print_bucket_roles(self.cloud_staging_bucket)
        print('done')
        # end __init__

    def create_and_upload_package_tar(self, local_package_root: Text, do_add_train_module=False):
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
        return cloud_tarfile_url

    def modify_job_spec_inplace(self, job_spec: Dict):
        # make tarfile
        package_urls = []
        package_url = self.create_and_upload_package_tar(self.local_package_root_dir, do_add_train_module=True)
        package_urls.append(package_url)

        # upload additional packages
        for additional_dir in self.additional_package_root_dirs:
            package_url = self.create_and_upload_package_tar(additional_dir, do_add_train_module=False)
            package_urls.append(package_url)

        # if self.verbose:
        #     print(f"Removing temp directory: {str(tmp_dir_path)}")
        # shutil.rmtree(tmp_dir_path)

        # modify job_spec appropriately
        #   ref: https://cloud.google.com/ml-engine/reference/rest/v1/projects.jobs
        job_spec['trainingInput']['packageUris'] = package_urls
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

            # # s = io.StringIO()
            # # s.write(contents)
            # # s.seek(0)
            # # tarinfo = tarfile.TarInfo(name=str(Path(self.package_name) / self.TRAIN_MODULE_NAME))
            # # tarinfo.size = len(s.buf)
            #
            # tarf.addfile(tarinfo=tarinfo, fileobj=s)

    def _upload_file(self, file_path: Path):
        # https://github.com/GoogleCloudPlatform/getting-started-python/blob/master/3-binary-data/bookshelf/storage.py
        """
        Uploads a file to a given Cloud Storage bucket and returns the public url
        to the new object.
        """
        # from google.auth import compute_engine
        # cred = compute_engine.Credentials()

        client = storage.Client()
        bucket = client.bucket(self.cloud_staging_bucket)
        blob = bucket.blob(file_path.name)
        blob.upload_from_filename(str(file_path))

        # # Reload fetches the current ACL from Cloud Storage.
        # blob.acl.reload()
        #
        # # You can also use `group`, `domain`, `all_authenticated` and `all` to
        # # grant access to different types of entities. You can also use
        # # `grant_read` or `grant_write` to grant different roles.
        # blob.acl.user(blob_owner_email).grant_owner()
        # blob.acl.save()

        package_location_on_cloud = f'gs://{str(Path(self.cloud_staging_bucket) / file_path.name)}'
        return package_location_on_cloud


class ContainerBasedRunInstructions(JobSpecModifier):
    def __init__(self, container_image_uri: Text):
        """A container which has your train module as it's CMD or ENTRYPOINT"""
        self.container_image_uri = container_image_uri

    def modify_job_spec_inplace(self, job_spec: Dict):
        job_spec['trainingInput']['masterConfig'] = {'imageUri': self.container_image_uri}


