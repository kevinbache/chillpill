from typing import Text
from importlib import import_module
from chillpill import simple_argparse


def get_obj(import_str: Text):
    module_import_str, obj_name = import_str.rsplit('.', 1)
    module = import_module(module_import_str)
    obj = getattr(module, obj_name)
    return obj


if __name__ == '__main__':
    train_fn_import_str = 'chillpill.examples.cloud_complex_hp_tuning.train.train_fn'
    train_param_type_import_str = 'chillpill.examples.cloud_complex_hp_tuning.train.MyParams'

    train_fn = get_obj(train_fn_import_str)
    train_param_type = get_obj(train_param_type_import_str)

    # get a parameter dictionary
    hp = train_param_type.from_dict(simple_argparse.args_2_dict())
    print(hp)
    train_fn(hp)

