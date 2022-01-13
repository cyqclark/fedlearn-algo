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

import numpy
from scipy import stats
from collections import Counter
from enum import Enum, unique
from sklearn.preprocessing import LabelEncoder,MaxAbsScaler
import sklearn.preprocessing as preprocessing
from typing import Dict, Optional, List, Tuple, Set, Sequence, Callable
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
import numpy


class HyperParameters():
    def __init__(self):
        # learning rate
        self.lr = 0.001
        # maximum iteration
        self.max_iter = 100 
        self.batch_size = None

class InputModule(nn.Module):
    def __init__(self, catefeat: Dict, numfeat: Dict):
        """
        Input Module of the SplitNN
        
        """
        super(InputModule, self).__init__()

        ##add user attribute embedding
        dim_categorical = 0
        ##sub-channel of categorical feature
        temp_attr_emb = {}
        self.cate_featurelist = []
        self.attremb_dim = 5
        for temp_name in catefeat.keys():
            self.cate_featurelist.append(temp_name)
            feat_info = catefeat[temp_name]
            vocab_size = feat_info.dim
            dim_categorical += attremb_dim
            temp_attr_emb[temp_name] = nn.Embedding(vocab_size, self.attremb_dim)
        if len(temp_attr_emb) == 0:
            self.embedding = None
        else:
            self.embedding = nn.ModuleDict(temp_attr_emb)
        # sub-channel of numerical feature
        dim_numerical = len(numfeat)

        self.num_featurelist = []
        for temp_name in numfeat.keys():
            self.num_featurelist.append(temp_name)

        self.fc_num = nn.Linear(dim_numerical, dim_numerical)
        self.output_dim = dim_numerical + dim_categorical

    def get_embedding(self, user_attr, temp_name):
        return self.embedding[temp_name](torch.from_numpy(user_attr))

    def forward(self, cate_feat, num_feat):
        embed_userattr = []
        category_concate = None
        num_output = None
        feat_concate = None
        num_catefeat = len(self.cate_featurelist)
        num_numfeat = len(self.num_featurelist)

        for i in range(num_catefeat):
            feat_name = self.cate_featurelist[i]
            feature = cate_feat[feat_name]
            embed_userattr.append(self.get_embedding(feature, feat_name))

        ##attribute feature concatenate
        if num_catefeat > 0:
            category_concate = torch.cat(tuple(attr_embed for attr_embed in embed_userattr), 1)
            feat_concate = category_concate

        if num_numfeat > 0:
            num_concate = numpy.concatenate(tuple(num_feat[temp_name] for temp_name in self.num_featurelist), 1).astype(numpy.float32)
            num_output = self.fc_num(torch.from_numpy(num_concate))
            if feat_concate is not None:
                feat_concate = torch.cat((feat_concate, num_output),1)
            else:
                feat_concate = num_output

        return feat_concate


class CategoricalFeature:
    def __init__(self, name: str, encoder: LabelEncoder):
        self.name = name
        self.encoder = encoder
        self.dim = len(list(encoder.classes_))

class NumericalFeature:
    def __init__(self, name: str, scaler: MaxAbsScaler):
        self.name = name
        self.scaler = scaler

@unique
class FeatureType(Enum):
    Categorical = 'Categorical'  # 0
    Numerical = 'Numerical'  # 1

def check_integer(lst: List) -> bool:
    for l in lst:
        if l % 1 > 0:
            return False
    return True

def data_transform(feature, category_encoder, numerical_scaler) -> Tuple[Dict, Dict]:
    """

    """
    cate_feat = {}
    num_feat = {}
    n = len(feature)
    for temp_name in category_encoder.keys():
        class_list = list(category_encoder[temp_name].encoder.classes_)
        feature.loc[~feature[temp_name].isin(class_list), temp_name] = numpy.inf
        cate_feat[temp_name] = category_encoder[temp_name].encoder.transform(feature[temp_name].values)

    for temp_name in numerical_scaler.keys():
        num_feat[temp_name] = numerical_scaler[temp_name].scaler.transform(numpy.asarray(feature[temp_name]).reshape(n,1))
    '''
    for i in range(d):
        temp_name = feature_name[i]
        if data_type[i] == FeatureType.Categorical:
            # data token
            class_list = list(cate_encoders[temp_name].classes_)
            feature[~feature[:,i].isin(class_list), i] = numpy.inf
            cate_feat[temp_name] = category_encoder[feature_name].transform(feature[:,i])
        elif data_type[i] == FeatureType.Numerical:
            num_feat[temp_name] = numerical_scaler[feature_name].transform(feature[:,i])
    '''
    return cate_feat, num_feat


def data_preprocessing(feature, feature_name) -> Tuple[Dict, Dict]:
    """

    """
    data_type = feature_type_eval(feature, feature_name)
    category_features = {}
    numerical_features = {}
    n = len(feature)
    for temp_name in feature_name:
        if data_type[temp_name] == FeatureType.Categorical:
            # data tokenization code
            encoder = LabelEncoder()
            value_list = list(numpy.unique(feature[temp_name].values)) + [numpy.inf]
            temp_encoder = encoder.fit(value_list)
            category_features[temp_name] = CategoricalFeature(name=temp_name, encoder=temp_encoder)
        else:
            # data normalization
            maxabs_scaler = preprocessing.MaxAbsScaler().fit(numpy.asarray(feature[temp_name]).reshape(n,1))
            numerical_features[temp_name] = NumericalFeature(name=temp_name, scaler=maxabs_scaler)
    return category_features, numerical_features


def feature_type_eval(feature, feature_name) -> Dict:
    """
    Detecting feature type, i.e. categorical or numerical.
    Pre-processing will be conducted based on feature type.
    :param feature: feature Dataframe
                    feature_name
    :return: list of feature type
    """
    feature_type = {}
    for temp_name in feature_name:
        c = Counter(feature[temp_name])
        keys = [k for k in c]
        is_integer = check_integer(keys)
        if is_integer and len(keys) <= 100:
            feature_type[temp_name] = FeatureType.Categorical
        else:
            feature_type[temp_name] = FeatureType.Numerical
    return feature_type


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