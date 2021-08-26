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
Production wrapper code for 1 out of n OT
"""
import json
import os
import sys

import rsa

import EncodedNumber, util

sys.path.append(os.getcwd())
from core.server.server import Server
from core.entity.common.message import RequestMessage, ResponseMessage
from ot_core import Alice1_nOT, Bob

class ActiveWrapper(Bob, Server):
    def __init__(self,
                 raw_number,
                 active_client_info,
                 passive_client_info,
                 rand_message_bit=256):

        # Bob initialization
        Bob.__init__(self, rand_message_bit)
        self.dict_functions = {"0": self.create_init_request,
                               "1": self.second_request_grpc,
                               "2": self.parse_final_grpc}
        self.reset_auto_machine()
        self.set_raw_number(raw_number)

        # coordinator initialization
        self.inference_finish = False
        self.coordinator_info = active_client_info
        self.client_info = passive_client_info
        self.remote = False
        return None
        
    def set_raw_number(self, raw_number):
        self.raw_number = raw_number
        self.encoded_number = EncodedNumber.OTEncodedNumber(raw_number)
        self.secret = self.encoded_number.encoded_number_array_decimal
        return None

    def reset_auto_machine(self):
        self.current_state = -1
        return None

    def control_flow_coordinator(self,
                            phase_num,
                            responses):
        """
        The main control flow of coordinator. This might be able to work in a generic
        environment.
        """
        # update phase id
        for _, resi in responses.items():
            resi.phase_id = phase_num
        # if phase has preprocessing, then call preprocessing func
        if phase_num in self.dict_functions:
            requests = self.dict_functions[phase_num](responses)
        else:
            import pdb
            pdb.set_trace()
        return requests

    def get_next_phase(self, phase):
        if phase == "-1":
            return "0"
        elif phase == "0":
            return "1"
        elif phase == "1":
            self.inference_finish = True
            print("Finish!")
            return "2"
        else:
            raise ValueError("Invalid phase!")

    def check_ser_deser(self, message):
        if self.remote:
            if isinstance(message, ResponseMessage):
                message.deserialize_body()
            elif isinstance(message, RequestMessage):
                message.serialize_body()
        return None

    def make_request(self, response, body, phase_id):
        request = RequestMessage(self.coordinator_info,
                                 response.client_info,
                                 {str(key): value for key, value in body.items()},
                                 phase_id=phase_id)
        self.check_ser_deser(request)
        return request

    def create_init_request(self):
        """
        Send start comparison request
        """
        requests = {clienti: RequestMessage(self.coordinator_info, clienti, {}, "0")
                    for clienti in self.client_info}
        return requests

    def init_request(self, message=None):
        """
        Send start comparison request
        """
        return "start"

    def second_request_grpc(self, response):
        request = {}
        for machine_info, res in response.items():
            message = res.body["body"]
            message = self.second_request(message)
            reqi = self.make_request(res,
                                     body={"body": message},
                                     phase_id="1")
            request[machine_info] = reqi
        return request

    def second_request(self, message):
        """
        Receive first response and send second request
        """
        # deserialization
        message = json.loads(message)

        message["key"] = util.create_rsa_key(message["key"])
        self.receive_alice_key_with_rand_message_array(message)
        request = self.send_selected_message_array(self.secret)
        
        # serialization
        return json.dumps(request)

    def parse_final_grpc(self, response):
        request = {}
        for machine_info, res in response.items():
            message = res.body["body"]
            message = self.parse_final(message)
            reqi = self.make_request(res,
                                     body={"body": message},
                                     phase_id="2")
            request[machine_info] = reqi
        return request

    def parse_final(self, message):
        """
        Receive second response and parse the final result
        """

        # deserialization
        message = json.loads(message)

        self.receive_secret(message, self.secret)
        self.result = self.parse_result()
        return self.result

    def init_inference_control(self):
        return None

    def is_inference_continue(self):
        return not self.inference_finish

    def post_inference_session(self):
        return None

    # training part
    def init_training_control(self):
        return None

    def is_training_continue(self):
        return None
    
    def post_training_session(self):
        return None