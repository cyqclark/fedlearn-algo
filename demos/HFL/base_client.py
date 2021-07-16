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

__doc__ = "Abstract class for Client"
import os,sys
from  abc import ABC, abstractmethod
from typing import Any, Union

root_path = os.getcwd()

sys.path.append(root_path)
sys.path.append(os.path.join(root_path,'demos/HFL'))
#sys.path.append(os.path.join(root_path,'demos/HFL/common'))

from demos.HFL.common.param_util import(
    Params, 
    ParamsRes, 
    TrainArgs, 
    TrainRes, 
    EvalArgs,
    EvalRes,
    NLPInferArgs
)


class Client(ABC):
    """
    Define basic functions for local model such as parameters retrieval/updating,
    model training/evaluation and inference.
    """
    @abstractmethod
    def get_params(self)->ParamsRes:
        pass
    
    @abstractmethod 
    def set_params(self, params:Params)->None:
        pass
    

    @abstractmethod
    def train(self, trainArgs:TrainArgs)->TrainRes:
        pass
    

    @abstractmethod
    def evaluate(self, evalAgrs:EvalArgs)->EvalRes:
        pass

    @abstractmethod
    def inference(self,  inputArgs:Union[NLPInferArgs])->Any:
        pass

