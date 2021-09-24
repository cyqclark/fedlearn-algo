__doc__ = "support functions"

import orjson
import numpy as np


DTYPE = 'float32'

if DTYPE=='float32':
    NP_DTYPE = np.float32
elif DTYPE=='float16':
    NP_DTYPE = np.float16
else:
    raise(f"DTYPE {DTYPE} is not supported yet!")

data_type = {
        -1: 'undefined',
        0: 'plaintext',
        1: 'riac'
        }

spliter = '|spliter|'
divider = '|divider|'

def serialize(data_dict, *args, **kwargs):
    data_dict.update(kwargs)
 
    msg = []
    for k, v in data_dict.items():
        if isinstance(v, np.ndarray):
            v = v.astype(DTYPE) # fp32 is sufficient enough
            _shape = v.shape
            _data = v.ravel().tobytes().decode('ISO-8859-1')
            msg += ['ndarr', divider, k, divider, _data, divider, ",".join([str(v) for v in _shape]) ]
        elif isinstance(v, int):
            msg += ['int', divider, k, divider, str(v)]
        else:
            raise(f"data type {type(v)} for {k} is not supported!")
        msg.append(spliter)

    msg = "".join(msg)
    # print(type(msg), 'msg')
    return  msg


def deserialize(msg, *args, **kwargs):
    data_dict = {}
    # print(type(msg), 'deser')
    for item_str in  msg.split(spliter):
        item = item_str.split(divider)
        if item[0] == "ndarr":
            _, _name, _arr, _shape = item
            # print('des component: ', _name)
            _arr = np.frombuffer(_arr.encode('ISO-8859-1'), dtype=NP_DTYPE)
            _shape = tuple(map(int, _shape.split(',')))
            data_dict[_name] = _arr.astype(DTYPE ).reshape(_shape)
        elif item[0] == 'int':
            _, _name, _v = item
            # print('des component: ', _name)
            data_dict[_name] = int(_v)
    return data_dict
