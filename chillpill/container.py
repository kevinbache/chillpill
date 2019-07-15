import tarfile
from pathlib import Path
from typing import Text


def find_package_root(location_within_package: Text):
    """
    Find the root of the package which

    Args:
        location_within_package:
            Any file location within the package.

    Returns:

    """

    SETUP_NAME = 'setup.py'

    p = Path(location_within_package).expanduser().resolve()
    if p.is_file():
        p = p.parent

    while True:
        if (p / SETUP_NAME).is_file():
            return p
        else:
            parent = p.parent
            if parent == p:
                """
                For reference: 
                p = Path('/')
                p.parent      # ==> PosixPath('/')
                """
                raise ValueError(f"Couldn't find a file named {SETUP_NAME} in any of "
                                 f"the parent directories of {location_within_package}")
            p = parent


def make_targz_from_directory(dir_to_tar: Path, tarfile_name: Text, error_if_containing_dir_dne=False):
    output_dir = Path(tarfile_name).parent
    if not output_dir.is_dir():
        if error_if_containing_dir_dne:
            raise ValueError(f"The output directory, {output_dir}, does not exist.")
        else:
            output_dir.mkdir(parents=True)

    with tarfile.open(tarfile_name, "w:gz") as f:
        f.add(dir_to_tar, arcname=dir_to_tar.stem)


def make_targz_of_package(any_location_within_package: Text, output_tarfile_fullname: Text):
    p = find_package_root(any_location_within_package)
    make_targz_from_directory(p, output_tarfile_fullname)


if __name__ == '__main__':
    package_root = Path('/Users/bache/projects/chillpill')
    p = find_package_root("/Users/bache/projects/chillpill/chillpill/examples/local_hp_tuning.py")
    assert p == package_root

    p = find_package_root("/Users/bache/projects/chillpill/chillpill/examples/")
    assert p == package_root

    targz_file = '/tmp/mytarfile/pkg.tar.gz'
    make_targz_of_package("/Users/bache/projects/chillpill/chillpill/examples/local_hp_tuning.py", targz_file)
    print(Path(targz_file).is_file())
