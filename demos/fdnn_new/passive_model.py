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
import torch
import numpy
from torch import nn
import math
from typing import Dict
from table_data import TableData


class InputModule(nn.Module):
    def __init__(self, catefeat: Dict, numfeat: Dict, is_cuda = False):
        """
        Input Module of the SplitNN
        :param catefeat: dictionary of the categorical feature encoders
        :param numfeat: dictionary of the numerical features encoders
        """
        super(InputModule, self).__init__()

        ## add user attribute embedding
        dim_categorical = 0
        ## sub-channel of categorical feature
        temp_attr_emb = {}
        self.cate_feature_list = []
        self.emb_dim = 5
        for temp_name in catefeat.keys():
            self.cate_feature_list.append(temp_name)
            feat_info = catefeat[temp_name]
            vocab_size = feat_info.dim
            dim_categorical += self.emb_dim
            temp_attr_emb[temp_name] = nn.Embedding(vocab_size, self.emb_dim)

        if len(temp_attr_emb) == 0:
            self.embedding = None
        else:
            self.embedding = nn.ModuleDict(temp_attr_emb)

        # sub-channel of numerical feature
        dim_numerical = len(numfeat)
        self.num_feature_list = []
        for temp_name in numfeat.keys():
            self.num_feature_list.append(temp_name)

        self.fc_num = nn.Linear(dim_numerical, dim_numerical)
        self.output_dim = dim_numerical + dim_categorical

    def get_embedding(self, user_attr, temp_name):
        return self.embedding[temp_name](torch.from_numpy(user_attr))

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


class OutputModule(nn.Module):
    def __init__(self, feat_dim):
        super(OutputModule, self).__init__()
        self.fc = nn.Linear(feat_dim, math.ceil(feat_dim/2))
        self.out_dim = math.ceil(feat_dim/2)

    def forward(self, input):
        return self.fc(input)


class CoreModule(nn.Module):
    def __init__(self, feat_dim):
        super(CoreModule, self).__init__()
        self.fc = nn.Linear(feat_dim, feat_dim)
        self.out_dim = feat_dim
        
    def forward(self, input):
        return self.fc(input)


class PassiveModel(nn.Module):
    def __init__(self, data):
        super(PassiveModel, self).__init__()
        cate_encoders = data.cate_encoders
        num_scalars = data.num_scalers
        self.input_module = InputModule(cate_encoders, num_scalars)
        self.model = CoreModule(self.input_module.output_dim)
        self.output_module = OutputModule(self.model.out_dim)
        self.out_dim = self.output_module.out_dim
        return

    def forward(self, data):
        cate_feat = data.cate_data
        num_feat = data.num_data
        output = self.input_module(cate_feat, num_feat)
        output = self.model(output)
        output = self.output_module(output)
        return output
