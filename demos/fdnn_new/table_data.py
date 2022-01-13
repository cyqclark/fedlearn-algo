import pandas
import numpy
from typing import List, Dict
from utils import NumericalFeature, CategoricalFeature
import utils


class MiniBatchData:
    def __init__(self, cate_data={}, num_data={}, label=None):
        self.cate_data = cate_data
        self.num_data = num_data
        self.label = label


class TableData:
    def __init__(self):
        self.data_path = None
        self.data = {'features': None, 'feature_names': None, 'label': None}
        self.uid = None
        self.cate_data = {}
        self.num_data = {}
        self.label = None
        self.cate_encoders = {}
        self.num_scalers = {}
        self.feature_category = None
        self.sample_number = 0

    def load_data(self, path: str, feature_names: List, label: str=None):
        df = pandas.read_csv(path)
        self.uid = df.loc[:, ["uid"]]
        self.data['features'] = df[feature_names]
        self.sample_number = df.shape[0]
        self.data['feature_names'] = feature_names
        if label is None:
            self.data['label'] = None
        else:
            self.data['label'] = df[label].values.tolist()
            self.label = numpy.asarray(self.data['label'])
            
    def data_shuffle(self, index):
        if len(self.cate_data) > 0:
            for cate_feat in self.cate_data:
                self.cate_data[cate_feat] = self.cate_data[cate_feat][index]
        if len(self.num_data) > 0:
            for num_feat in self.num_data:
                self.num_data[num_feat] = self.num_data[num_feat][index]
        if self.label is not None:
            self.label = self.label[index]

    def get_minibatch(self, index):
        cate_data = {}
        num_data = {}
        label = None
        if len(self.cate_data) > 0:
            for cate_feat in self.cate_data:
                cate_data[cate_feat] = self.cate_data[cate_feat][index]
        if len(self.num_data) > 0:
            for num_feat in self.num_data:
                num_data[num_feat] = self.num_data[num_feat][index]
        if self.label is not None:
            label = self.label[index]
        return MiniBatchData(cate_data, num_data, label)
    
    def load_data_category_file(self, path: str):
        self.feature_category = pandas.read_csv(path)

    def data_preprocessing(self):
        self.cate_encoders, self.num_scalers = utils.data_preprocessing(self.data['features'], self.data['feature_names'])

    def data_transform(self):
        self.cate_data, self.num_data = utils.data_transform(self.data['features'], self.cate_encoders, self.num_scalers)