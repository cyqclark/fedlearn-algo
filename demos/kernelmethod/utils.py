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

import pandas as pd
import os
import sklearn.preprocessing as preprocessing
from kernelmethod import KernelMapping


class DataError(ValueError):
    pass


class Data:
    def __init__(self, data_path: str = None):
        self.data_path = data_path

        if os.path.exists(data_path) is False:
            raise DataError("%s is not a valid path" % data_path)

        self.data_frame = pd.read_csv(data_path)
        self.feature = None
        self.normalized_feature = None
        self.transformed_feature = None
        self.label = None
        self.scaler = None
        self.is_normalize = False
        self.normalize_type = None
        return

    def normalize_transform(self, scaler=None):
        """
        Feature normalization.
        """
        if scaler is None:
            if self.scaler is None:
                raise DataError('scalar is not properly set up.')
        else:
            self.scaler = scaler

        if self.feature is None:
            raise DataError('sample feature is not properly set up.')

        if self.scaler is not None:
            self.normalized_feature = self.scaler.transform(self.feature)
            self.is_normalize = True

    def normalize_fit(self, normalize_type: str = 'standardscaler'):
        """
        Generate feature normalization parameters.
        """
        if self.feature is None:
            raise DataError('data is not properly set up.')

        if normalize_type not in ['standardscaler', 'minmaxscaler']:
            raise DataError('%s is not supported as normalization type.')

        self.is_normalize = True
        self.normalize_type = normalize_type

        if normalize_type == 'standardscaler':
            self.scaler = preprocessing.StandardScaler()
        elif normalize_type == 'minmaxscaler':
            self.scaler = preprocessing.MinMaxScaler()

        self.scaler = self.scaler.fit(self.feature)

    def kernel_transform(self, kernel_mapping: KernelMapping):
        """
        Kernelized feature transformation.
        """
        if self.is_normalize is True:
            self.transformed_feature = kernel_mapping.transform(self.normalized_feature)
        else:
            self.transformed_feature = kernel_mapping.transform(self.feature)
