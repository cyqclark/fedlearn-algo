# Copyright 2021 Fedlearn authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

__doc__ = "support functions"

import json
import orjson
import numpy as np
#import utils.HE.riac as RIAC


DTYPE = 'float32'
NP_DTYPE = np.float32
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

def serialize_v1(np_tensor, *args, **kwargs):
   d = {
           'data': np_tensor.tolist(),
           'shape': np_tensor.shape,
       }
   d.update(kwargs)
   return orjson.dumps(d, option=orjson.OPT_SERIALIZE_NUMPY), None

def deserialize_v1(msg, *args, **kwargs):
    msg_dict = orjson.loads(msg)
    data = np.asarray(msg_dict['data']).astype(DTYPE).reshape(msg_dict['shape'])
    del msg_dict['data']
    del msg_dict['shape']
    return data, msg_dict 

   
#def serialize_v0(item, **kwargs):
#    #header = [str(v) for v in RIAC.get_size(item)]
#
#    msg_body = []
#    data_type = -1
#    data = np.asarray(item)
#    item_size = data.shape 
#    data = data.flatten().tolist()
#    for x in data:
#        if isinstance(x, (int, float)):
#            msg_body.append( str(x) )
#            if data_type == -1:
#                data_type = 0
#            elif data_type != 0:
#                raise NotImplementedError(f"Only support serialization of a single data type yet!")
#        elif isinstance(x, RIAC.IterativeAffineCiphertext):
#            msg_body.append( x.serialize_string() )
#            if data_type == -1:
#                data_type = 1
#            elif data_type != 1:
#                raise NotImplementedError(f"Only support serialization of a single data type yet!")
#        else:
#            raise NotImplementedError(f"Not implemented yet for data type {type(x)}!")
#
#    kwargs['data_type'] = data_type
#    kwargs['data_shape'] = item_size
#    kwargs['data'] = msg_body 
#    return orjson.dumps(kwargs), data_type
#
#
#def deserialize_v0(msg) -> np.ndarray:
#    """
#    Deserialize a mssage, could be an array of RIAC encrypted objects or plaintext objects
#    """
#    msg_dict = orjson.loads(msg)
#    _type = msg_dict['data_type']
#    _shape = msg_dict['data_shape']
#    data = []
#    for x in msg_dict['data']:
#        if _type == 0:
#            data.append( float(x) )
#        elif _type == 1:
#            data.append( RIAC.IterativeAffineCiphertext.deserialize_string(x) )
#        else:
#            raise NotImplementedError(f"Not implemented yet for data type {_type}!")
#    data = np.asarray(data).reshape(_shape).astype('float32')
#    del msg_dict['data']
#    return data, msg_dict 
#
#
#if __name__ == "__main__":
#    # testing code
#    x = -1 * np.load('/app/src/python/client_side/data_face.npy').astype('float32')[0]
#    key = RIAC.generate_keypair()
#    enc_x = RIAC.encrypt(key, x)
#
#    ser_x, _= serialize(enc_x, online=1)
#    _x, dd = deserialize(ser_x)
#
#    y = RIAC.decrypt(key, _x)
#    print('x: ', np.asarray(x).flatten()[:10])
#    print('y: ', np.asarray(y).flatten()[:10])
#    print('diff: ', np.max(np.abs(np.asarray(x) - np.asarray(y))))
#            
