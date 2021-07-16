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

import copy
import pickle
from typing import Dict

from core.entity.common.machineinfo import MachineInfo

class ResponseMessage:
    """
    The message that client machine sends to server machine
    """
    def __init__(self,  sender: MachineInfo, receiver: MachineInfo, body: Dict, phase_id: str):
        self.body = body
        self.phase_id = phase_id
        self.client_info = sender
        self.server_info = receiver

    def copy(self):
        return ResponseMessage(self.client_info,
                               self.server_info,
                               copy.deepcopy(self.body),
                               self.phase_id)

    def serialize_body(self):
        """
        A sample serialization function for body in ResponseMessage
        """
        self.body = serialize_body(self.body)
        return None

    def deserialize_body(self):
        """
        A sample deserialization function for body in ResponseMessage
        """
        self.body = deserialize_body(self.body)
        return None


class RequestMessage:
    """
    The message that master machine sends to client machine
    """
    def __init__(self, sender: MachineInfo, receiver: MachineInfo, body: Dict, phase_id: str):
        self.phase_id = phase_id
        self.body = body
        self.server_info = sender
        self.client_info = receiver
    
    def copy(self):
        return RequestMessage(self.server_info,
                              self.client_info,
                              copy.deepcopy(self.body),
                              self.phase_id)

    def serialize_body(self):
        """
        A sample serialization function for body in RequestMessage
        """
        self.body = serialize_body(self.body)
        return None

    def deserialize_body(self):
        """
        A sample deserialization function for body in RequestMessage
        """
        self.body = deserialize_body(self.body)
        return None


def serialize_body(body: Dict):
    """
    A default serialize function that serialize items in body.
    This function is implemented using pickle.
    """
    serialized_body = {}
    for key, value in body.items():
        serialized_body[key] = pickle.dumps(value)
    return serialized_body


def deserialize_body(body: Dict):
    """
    A default deserialize function that deserialize items in body.
    This function is implemented using pickle.
    """
    deserialized_body = {}
    for key, value in body.items():
        deserialized_body[key] = pickle.loads(value)
    return deserialized_body