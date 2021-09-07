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
Demo script for secure two party comparison
"""
import os
import random
import sys
import time

import numpy

import client
import coordinator
import EncodedNumber

sys.path.append(os.getcwd())
from core.entity.common.machineinfo import MachineInfo

def test_comparison():
    active = coordinator.ActiveWrapper(1., None, None)
    passive = client.PassiveWrapper(1., None)

    active_number = numpy.random.rand()
    passive_number = numpy.random.rand()
    active.set_raw_number(active_number)
    passive.set_raw_number(passive_number)

    message_1st = active.init_request()

    message_2nd = passive.init_response(message_1st)

    message_3th = active.second_request(message_2nd)

    message_4th = passive.second_response(message_3th)

    result = active.parse_final(message_4th)

    print("Raw number:")
    print("Active: %.6f"%active_number)
    print("Passve: %.6f"%passive_number)
    print("Secure comparison result: Passive > Active ? %s"%str(result))


def test_comparison_grpc():

    active_client_info = MachineInfo("127.0.0.1", "8001", "0")
    passive_client_info = MachineInfo("127.0.0.1", "8002", "0")

    active = coordinator.ActiveWrapper(1.,
                                       active_client_info=active_client_info,
                                       passive_client_info=[passive_client_info])
    passive = client.PassiveWrapper(1.,
                                    client_info = passive_client_info)
    client_map = {passive_client_info: passive,
                }
    active_number = numpy.random.rand()
    passive_number = numpy.random.rand()
    active.set_raw_number(active_number)
    passive.set_raw_number(passive_number)

    phase = "0"

    init_requests = active.create_init_request()

    responses = {}
    for client_info, reqi in init_requests.items():
        c = client_map[client_info]
        responses[client_info] = c.control_flow_client(reqi.phase_id, reqi)

    while True:
        phase = active.get_next_phase(phase)
        print("Phase %s start..."%phase)
        requests = active.control_flow_coordinator(phase, responses)
        responses = {}
        if active.is_inference_continue():
            for client_info, reqi in requests.items():
                c = client_map[client_info]
                responses[client_info] = c.control_flow_client(reqi.phase_id, reqi.copy())
        else:
            break

    result = active.result

    print("Raw number:")
    print("Active: %.6f"%active_number)
    print("Passve: %.6f"%passive_number)
    print("Secure comparison result: Passive > Active ? %s"%str(result))

    return None


if __name__ == "__main__":
    print("Local demo")
    test_comparison()
    print("Remote demo")
    test_comparison_grpc()