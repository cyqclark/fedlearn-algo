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

__doc__ = "tensorflow related util function"

import pickle

import numpy

import tensorflow as tf

###########################################
# conversion between numpy array and bytes
###########################################

def serialize_numpy_array_dict(x: numpy.ndarray) -> dict:
    """
    Convert a numpy array to bytes from numpy internal function
    """
    data = x.tobytes()
    dtype = x.dtype.name
    shape = x.shape
    return {"data": data,
            "dtype": dtype,
            "shape": shape}

def serialize_numpy_array_dict_to_btyes(x: numpy.ndarray) -> bytes:
    """
    Convert serialized numpy array dict to byte using pickle
    """
    d = serialize_numpy_array_dict(x)
    return pickle.dumps(d)

def serialize_numpy_array_pickle(x: numpy.ndarray) -> bytes:
    """
    Convert a numpy array to bytes directly from pickle
    """
    return pickle.dumps(x)

def serialize_numpy_array(x: numpy.array, method: str = "numpy") -> bytes:
    """
    Serialize a numpy array
    """
    if method == "numpy":
        return serialize_numpy_array_dict_to_btyes(x)
    elif method == "pickle":
        return serialize_numpy_array_pickle(x)
    else:
        raise ValueError("Unsupported value of method!")

def deserialize_numpy_array_dict(d: dict) -> numpy.ndarray:
    """
    Create numpy array from serialized dict
    """
    return numpy.frombuffer(d["data"], dtype=d["dtype"]).reshape(d["shape"])

def deserialize_numpy_array_from_dict_bytes(x: bytes) -> dict:
    """
    Create numpy array dict from serialized numpy array dict bytes
    """
    d = pickle.loads(x)
    return deserialize_numpy_array_dict(d)

def deserialize_numpy_array_pickle(x: bytes) -> numpy.ndarray:
    """
    Create numpy array directly from serialized byte using pickle
    """
    return pickle.loads(x)

def deserialize_numpy_array(x: bytes, method: str = "numpy") -> numpy.ndarray:
    if method == "numpy":
        return deserialize_numpy_array_from_dict_bytes(x)
    elif method == "pickle":
        return deserialize_numpy_array_pickle(x)
    else:
        raise ValueError("Unsupported value of method!")

###########################################
# conversion between TF tensor and bytes
###########################################

def serialize_tf_tensor(tensor, method="numpy") -> bytes:
    """
    Serialize a tf tensor to bytes
    """
    return serialize_numpy_array(tensor.numpy(), method=method)

def deserialize_tf_tensor(x: bytes, method="numpy"):
    """
    Deserialize a tf tensor from bytes
    """
    ndarray = deserialize_numpy_array(x, method=method)
    return tf.convert_to_tensor(ndarray)


###########################################
# conversion between TF tensor and bytes
###########################################

def list_array_to_bytes(x):
    return None

def bytes_to_list_array(x):
    return None


###########################################
# Tests
###########################################

def test():
    x = numpy.random.randn(5, 3)

    for methodi in "numpy pickle".split():
        data = serialize_numpy_array(x, method=methodi)
        x1 = deserialize_numpy_array(data, method=methodi)
        assert numpy.array_equal(x, x1)

    tfx = tf.convert_to_tensor(x)
    for methodi in "numpy pickle".split():
        data = serialize_tf_tensor(tfx, method=methodi)
        tfx1 = deserialize_tf_tensor(data, method=methodi)
        tf.debugging.assert_equal(tfx, tfx1)
    return None
