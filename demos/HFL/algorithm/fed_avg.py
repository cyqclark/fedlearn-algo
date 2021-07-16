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
    TrainRes
)

from demos.HFL.algorithm.base_aggregation import Aggregator

class FedAvg(Aggregator):
    def aggregate(self, 
                    trainRes_list: List[TrainRes]
                 )->Params:
    
        """
        Fed Avg algorithm for HFL
        
        Parameters
        ---------
        trainRes_list: List[TrainRes]
            A list of TrainRes, each corresponds to one client's model parameters and training metrics

        Returns
        -------
            Params: Parameters of global model 
        """        
        w_names = trainRes_list[0].params.names
        total_samples = sum(tr.num_samples for tr in trainRes_list)        
        weights =    [tr.num_samples/total_samples for tr in trainRes_list]
        

        ave_params = [([w*weights[idx] for w in tr.params.weights]) 
                            for idx,tr in enumerate(trainRes_list)]


        ave_params =  [sum([ data[layer_idx] for data in ave_params]) 
                        for layer_idx in  range(len(ave_params[0]))]                    
                         
                         
        return Params(
            names=w_names,
            weights=ave_params,
            weight_type='float')         
