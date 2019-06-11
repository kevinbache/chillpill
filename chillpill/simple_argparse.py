"""This module implements simple, dynamic argument parsing for Python scripts.

Use this when argparse is too verbose.
"""
from collections import OrderedDict
import sys
from typing import Text, Optional, Dict, List


def _parse(args: List[Text]) -> Dict:
    """Simple key value arg parser which doesn't need to know ahead of time what the argument names will be.

    Args should be a raw list of the arguments passed to a script, as in:
        import sys
        _parse(sys.argv[1:])
    """
    eq = '='
    out = OrderedDict()
    k = None
    err = ValueError('Got unparsable arguments: {}'.format(args))

    for arg in args:
        if eq in arg:
            first, second = arg.split(eq, maxsplit=1)
            if not first.startswith('-'):
                raise err
            first = first.replace('-', '')
            out[first] = second
            k = None
        else:
            if k is None:
                if arg.startswith('-'):
                    k = arg.replace('-', '')
                else:
                    raise err
            else:
                if arg.startswith('-'):
                    out[k] = None
                    k = arg.replace('-', '')
                else:
                    out[k] = arg
                    k = None
    if k is not None:
        out[k] = None

    return out


def _convert_values_to_types(v: Optional[Text]):
    """Converts strings to ints and floats where possible."""
    if v is None:
        return v

    try:
        return int(v)
    except ValueError:
        pass

    try:
        return float(v)
    except ValueError:
        pass

    if isinstance(v, Text):
        # remove quotes
        v = _remove_bounding_text(v, '"')
        v = _remove_bounding_text(v, "'")

    return v


def _remove_bounding_text(string: Text, remove: Text):
    if string[0] == string[-1] == remove:
        string = string[1:-1]
    return string


def _convert_numeric_values(d: Dict):
    return OrderedDict([(k, _convert_values_to_types(v)) for k, v in d.items()])


def _remove_quotes(d: Dict):
    return OrderedDict([(k, _convert_values_to_types(v)) for k, v in d.items()])


def args_2_dict(args=None):
    if args is None:
        args = sys.argv[1:]
    parsed = _parse(args)
    return _convert_numeric_values(parsed)


if __name__ == '__main__':
    args = [
        '--eq1=1',
        '--split', 'value',
        '--eq2.0=2.0',
        '--single',
        '--single2',
        '--split2', 'value2',
        '--quotes="handle quotes"'
    ]

    print()
    print("Parsed:")
    parsed = _parse(args)
    for k, v in parsed.items():
        print("{:>8}: {}".format(k, v))

    print()
    print("Converted to numerics:")
    converted = _convert_numeric_values(parsed)
    for k, v in converted.items():
        print("{:>8}: {:<20} {:<20}".format(k, str(v), str(type(v))))
