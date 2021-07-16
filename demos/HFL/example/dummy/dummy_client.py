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

import os, sys
import numpy as np
import time

import argparse
from typing import Any

root_path = os.getcwd()

sys.path.append(root_path)
sys.path.append(os.path.join(root_path, 'demos/HFL'))

from demos.HFL.base_client import Client
from demos.HFL.common.param_util import (
    Params,
    ParamsRes,
    TrainArgs,
    EvalArgs,
    TrainRes,
    EvalRes
)

class DummyClient(Client):
    def __init__(self, config = None):
        self._init_params()

    def __get_dummy_weights(self, num_layers = 1):
        return {f'layer_{i}': np.ones((1024, 1024)).astype('float') for i in range(num_layers)}

    def _init_params(self):
        W = self.__get_dummy_weights(num_layers = 6)

        self.params = Params(
            weights=list(W.values()),
            names=list(W.keys()),
            weight_type='float'
        )

    def get_params(self) -> ParamsRes:
        return ParamsRes(
            self.params,
            response_messages={'dummy', 1})

    def set_params(self, params: Params) -> None:
        self.params = params

    def train(self, trainArgs: TrainArgs) -> TrainRes:
        print('train dummy client...')

        return TrainRes(
            self.params,
            num_samples=100,
            metrics={'Acc': 0.5, 'Loss': 0.3, 'F1': 0.8})

    def evaluate(self, evalAgrs: EvalArgs) -> EvalArgs:
        pass

    def inference(self, inputArgs: Any) -> Any:
        pass 