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

from typing import Dict
from core.master.master import Master
from core.client.client import Client
from core.entity.common.machineinfo import MachineInfo
from core.entity.common.message import RequestMessage, ResponseMessage



'''
def train_pipeline(server: Master, clients: Dict[MachineInfo, Client]):
    """Training initialization. Send initialization signal to all clients."""
    requests = server.init_train_control()
    responses = {}
    for client_info, client in clients.items():
        request = requests[client_info]
        client = clients[client_info]
        responses[client_info] = client.process_request(request)
    requests = server.training_control(responses=responses)

    """Training loop. Synchronize """
    while server.is_continue():
        responses = {}
        for client_info, client in clients.items():
            request = requests[client_info]
            client = clients[client_info]
            responses[client_info] = client.process_request(request)
        requests = server.training_control(responses)

    """Training process finish. Send finish signal to all clients."""
    requests = server.finish_control()
    for client_info, request in requests.items():
        client = clients[client_info]
        responses[client] = client.process_request(request)
    server.end_training_session(responses)

    return


def inference_pipeline(server: Master, clients: Dict[MachineInfo, Client]):
    responses = Dict[MachineInfo, ResponseMessage]
    requests = server.init_inference_control()
    for client_info, client in clients.items():
        responses[client_info] = client.inference_init(responses)
    requests = server.control(responses)

    for client_info, client in clients.items():
        request = requests[client_info]
        responses[client_info] = client.inference(request)

    server.post_inference_control()
    return
'''
