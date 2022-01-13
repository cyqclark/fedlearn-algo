import torch
import torch.nn as nn
from typing import Dict


class InputModule(nn.Module):
    def __init__(self, catefeat: Dict, numfeat: Dict):
        """
        Input Module of the SplitNN
        :param catefeat: dictionary of the categorical features
        :param numfeat: dictionary of the numberical features
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
            num_concate = numpy.concatenate(tuple(num_feat[temp_name] for temp_name in self.num_featurelist), 1).astype(
                numpy.float32)
            num_output = self.fc_num(torch.from_numpy(num_concate))
            if feat_concate is not None:
                feat_concate = torch.cat((feat_concate, num_output), 1)
            else:
                feat_concate = num_output

        return feat_concate
