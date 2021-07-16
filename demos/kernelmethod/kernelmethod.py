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


class KernelMappingError(ValueError):
    pass


class KernelMappingParam:
    def __init__(self, scale: float, feature_dim: int, map_dim: int, kernel_type: str = 'rbf', seed: int = 0):
        if scale <= 0:
            raise KernelMappingError("scale value should be larger than 0.")
        if feature_dim <= 0:
            raise KernelMappingError("input data feature dimension should be larger than 0.")
        if map_dim <= 0:
            raise KernelMappingError("mapped data feature dimension should be larger than 0.")

        self.kernel_type = kernel_type
        self.map_dim = map_dim
        self.feature_dim = feature_dim
        self.scale = scale
        self.seed = seed
        return


def generate_mapping(scale: float, feature_dim: int, map_dim: int, seed: int = 0):
    """
    generate kernel mapping matrix and bias vector
    """
    numpy.random.seed(seed)
    bias = numpy.random.uniform(0, 2 * numpy.pi, map_dim)
    matrix = numpy.sqrt(2 * scale) * numpy.random.normal(size=(feature_dim, map_dim))
    return matrix, bias


class KernelMethodParam:
    def __init__(self):
        self.matrix = None
        self.bias = None
        self.kernel_mapping_param = None
        self.model_param = None
        return


class KernelMapping:
    def __init__(self, param: KernelMappingParam):
        self.map_dim = param.map_dim
        self.feature_dim = param.feature_dim
        self.scale = param.scale
        self.seed = param.seed
        self.matrix = None
        self.bias = None
        self.matrix, self.bias = generate_mapping(self.scale,
                                                  self.feature_dim,
                                                  self.map_dim,
                                                  self.seed)
        return

    def transform(self, data):
        """
        data kernel transformation.
        :param data: n by d numpy array
        :return: mapped data matrix
        """
        num, dim = data.shape
        if dim != self.feature_dim:
            raise KernelMappingError("dimension mismatch.")

        if self.matrix is None or self.bias is None:
            raise KernelMappingError("kernel parameters are not initialized.")

        feature_trans = numpy.cos(numpy.dot(data, self.matrix) + self.bias)
        feature_trans *= numpy.sqrt(2.) / numpy.sqrt(self.map_dim)

        return feature_trans
