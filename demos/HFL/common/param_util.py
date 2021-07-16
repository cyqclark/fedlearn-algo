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

"""
High level inputs and outputs definition for functions in HFL framework.
"""
from dataclasses import dataclass
from typing import Optional, Dict, List, Union

import numpy as np



Scalar = Union[bool, bytes, float, int, str]

Metrics = Dict[str, Scalar]


@dataclass
class Params:
    """Model Parameters.
       
       weights: List[np.ndarray]  | list of model parameter data
       names: List[str]           |  list of model parameter names
       weight_type: str           |  type option ['float']
       
    """
    names: List[str]
    weights: List[np.ndarray]
    weight_type: str

@dataclass
class ParamsRes:
    """Client response to return weights.
       params: Params
       response_messages: Dict[str, Scalar]
    """

    params: Params
    response_messages: Dict[str, Scalar]



@dataclass
class TrainArgs:
    """Train args for a client.
       
       params: Params
       config: Dict[str, Scalar]
    """

    params: Params
    config: Dict[str, Scalar]


@dataclass
class TrainRes:
    """Training response from a client.
       
       
       params: Params
       num_samples: int
       metrics: Optional[Metrics] = None
    
    """

    params: Params
    num_samples: int
    metrics: Optional[Metrics] = None


@dataclass
class EvalArgs:
    """Evaluate  args for a client."""

    params: Params
    config: Dict[str, Scalar]


@dataclass
class EvalRes:
    """Evaluate response from a client."""

    loss: float
    num_samples: int
    metrics: Optional[Metrics] = None

@dataclass
class NLPInferArgs:
    '''Inference  input for a NLP task client '''
    inputs: List[str]
    params: Optional[Params] = None

@dataclass
class NLPInferRes:
    ''' Inference response for NLP task client '''
    inputs:List[str]
    outputs:List[Union[np.ndarray,List[Scalar]]]

