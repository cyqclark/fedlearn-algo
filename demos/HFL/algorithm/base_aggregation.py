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

import os,sys

root_path = os.getcwd()
from typing import List,Tuple,Dict,Union
from abc import abstractmethod, ABC

sys.path.append(root_path)
sys.path.append(os.path.join(root_path,'demos/HFL'))


from demos.HFL.common.param_util import(
    Params, 
    ParamsRes, 
    TrainArgs, 
    TrainRes, 
    EvalArgs,
)


class Aggregator(ABC):
    def __init__(self,
                config:Dict[str,Union[str,int,float]]):
        self.config = config        
    
    @abstractmethod
    def aggregate(self, 
                    trainRes_list: List[TrainRes]
                 )->Params:
        ...
    def __call__(self,
                trainRes: List[TrainRes])->Params:
        return self.aggregate(trainRes)        

