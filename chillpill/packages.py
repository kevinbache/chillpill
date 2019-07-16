import inspect
from pathlib import Path
from typing import Any, Optional

SETUP_NAME = 'setup.py'


def find_module_root(obj: Any) -> Optional[Path]:
    """Find the root of a package, that is, the directory next to its setup.py, which contains that object."""
    p = Path(inspect.getfile(obj))
    while p.parent != p:
        prev = p
        p = p.parent
        for f in p.iterdir():
            if f.is_file() and f.name == SETUP_NAME:
                return prev
    raise ValueError(f"The passed object which is located at {inspect.getfile(obj)} "
                     f"is not contained within a Python package.")


def find_package_root(obj: Any) -> Optional[Path]:
    """Find the root of a package, that is, the directory containing its setup.py, from any object in that pkg."""
    out = find_module_root(obj)
    if out is None:
        return out
    else:
        return out.parent


def get_import_string_of_type(obj: Any):
    """Get the import string for the type of an object.

    Example:
        class MyClass:
            pass

        my_obj = MyClass()
        get_import_string_of_type(my_obj)
        #   --> 'path.to.module.MyClass'

    This also works correctly if the class is not defined in the same module as the member.
    """
    mod = inspect.getmodule(obj).__name__
    if isinstance(obj, type):
        name = obj.__name__
    else:
        name = obj.__class__.__name__
    return f'{mod}.{name}'


def get_package_name(obj: Any):
    """Get the name of the package containing the given object."""
    outs = inspect.getmodule(obj).__name__.split('.', 1)
    return outs[0]


