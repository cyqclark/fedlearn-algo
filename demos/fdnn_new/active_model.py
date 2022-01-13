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
from utils import CategoricalFeature, NumericalFeature
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import InputModule
from typing import Dict, List
import math
import torch.nn.functional as F
from table_data import TableData


class InputModule(nn.Module):
    def __init__(self, cate_feat: Dict, num_feat: Dict, is_cuda = False):
        """
        Input Module of the SplitNN
        :param cate_feat: dictionary of the categorical feature encoders
        :param num_feat: dictionary of the numerical features encoders
        """
        super(InputModule, self).__init__()

        ## add user attribute embedding
        dim_categorical = 0
        ## sub-channel of categorical feature
        temp_attr_emb = {}
        self.cate_feature_list = []
        self.emb_dim = 5
        for temp_name in cate_feat.keys():
            self.cate_feature_list.append(temp_name)
            feat_info = cate_feat[temp_name]
            vocab_size = feat_info.dim
            dim_categorical += self.emb_dim
            temp_attr_emb[temp_name] = nn.Embedding(vocab_size, self.emb_dim)

        if len(temp_attr_emb) == 0:
            self.embedding = None
        else:
            self.embedding = nn.ModuleDict(temp_attr_emb)

        # sub-channel of numerical feature
        dim_numerical = len(num_feat)
        self.num_feature_list = []
        for temp_name in num_feat.keys():
            self.num_feature_list.append(temp_name)

        self.fc_num = nn.Linear(dim_numerical, dim_numerical)
        self.output_dim = dim_numerical + dim_categorical

    def get_embedding(self, feat, feat_name):
        return self.embedding[feat_name](torch.from_numpy(feat))

    def forward(self, cate_feat: Dict, num_feat: Dict):
        """
        model forward processing
        :param cate_feat dictionary of categorical features
        :param num_feat dictionary of numerical features
        """
        embedding_list = []

        num_catefeat = len(self.cate_feature_list)
        num_numfeat = len(self.num_feature_list)

        for i in range(num_catefeat):
            feat_name = self.cate_feature_list[i]
            feature = cate_feat[feat_name]
            embedding_list.append(self.get_embedding(feature, feat_name))

        # concatenate categorical embeddings
        feature_concatenation = None
        if num_catefeat > 0:
            feature_concatenation = torch.cat(tuple(embed for embed in embedding_list), 1)

        if num_numfeat > 0:
            temp = numpy.concatenate(tuple(num_feat[temp_name] for temp_name in self.num_feature_list), 1).astype(numpy.float32)
            numerical_output = self.fc_num(torch.from_numpy(temp))
            if feature_concatenation is not None:
                feature_concatenation = torch.cat((feature_concatenation, numerical_output), 1)
            else:
                feature_concatenation = numerical_output

        return feature_concatenation


class ActiveModel(nn.Module):
    def __init__(self, data: TableData, remote_input_dim: Dict):
        super(ActiveModel, self).__init__()
        cate_feat = data.cate_encoders
        num_feat = data.num_scalers
        self.local_feature_module = InputModule(cate_feat, num_feat)
        #
        total_dim = 0
        temp = {}
        self.remote_clients = []
        self.remote_input = None
        for key in remote_input_dim.keys():
            dim = remote_input_dim[key]
            temp[key] = nn.Linear(dim, dim)
            total_dim += dim
            self.remote_clients.append(key)
        total_dim += self.local_feature_module.output_dim

        self.fc1 = nn.Linear(total_dim, math.ceil(total_dim/2))
        self.fc2 = nn.Linear(math.ceil(total_dim/2), 1)
        self.sigmoid = nn.Sigmoid()
       
    def forward(self, cate_input: Dict, num_input: Dict, remote_input: Dict):

        output_local = self.local_feature_module(cate_input, num_input)
        self.remote_input = remote_input
        output_remote = torch.cat(tuple(self.remote_input[key] for key in self.remote_clients), 1)
        concate = torch.cat((output_remote, output_local), 1)
        output = self.fc1(concate)
        output = self.fc2(output)
        output = self.sigmoid(output)
        return output




