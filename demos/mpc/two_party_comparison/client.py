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
from core.client.client import Client
from core.entity.common.message import RequestMessage, ResponseMessage
from ot_core import Alice1_nOT, Bob


class PassiveWrapper(Alice1_nOT, Client):
    def __init__(self,
                 raw_number,
                 client_info,
                 rsa_key=None,
                 rsa_key_size=512,
                 rand_message_bit=16):
        Alice1_nOT.__init__(self, rsa_key, rsa_key_size, rand_message_bit)
        #self.reset_auto_machine()
        self.set_raw_number(raw_number)
        
        self.dict_functions = {"0": self.init_response_grpc,
                               "1": self.second_response_grpc,
                              }
        self.client_info = client_info
        # no preprocessing or postprocessing in this demo training code
        self.preprocessing_func = {}
        self.postprocessing_func = {}
        return None

    def reset_auto_machine(self):
        self.current_state = -1
        return None

    def set_raw_number(self, raw_number):
        self.raw_number = raw_number
        self.encoded_number = EncodedNumber.OTEncodedNumber(raw_number)
        self.secret = self.encoded_number.compose_secret(
            self.encoded_number.encoded_number_array_decimal)
        return None

    def control_flow_client(self,
                            phase_num,
                            request):
        """
        The main control flow of client. This might be able to work in a generic
        environment.
        """
        # if phase has preprocessing, then call preprocessing func
        response = request
        if phase_num in self.preprocessing_func:
            response = self.preprocessing_func[phase_num](response)
        if phase_num in self.dict_functions:
            response = self.dict_functions[phase_num](response)
        # if phase has postprocessing, then call postprocessing func
        if phase_num in self.postprocessing_func:
            response = self.postprocessing_func[phase_num](response)
        return response

    def auto_receive(self, message):
        """
        Auto receive machine, experimental.
        """
        if self.current_state == -1:
            self.current_state = 0
        elif self.current_state == 0:
            self.current_state = 1
        elif self.current_state == 1:
            print("Finish!")
            return None
        return self.control_map[self.current_state](message)

    def init_response_grpc(self, request):
        body = request.body["body"] if "body" in request.body else ""
        response = self.init_response(body)
        return self.make_response(request, body={"body": response})

    def init_response(self, message=None):
        """
        Receive start request and send response
        """
        response = self.send_key_with_rand_message()
        # serialization
        response["key"] = util.extract_rsa_key(response["key"])
        return json.dumps(response)

    def second_response_grpc(self, request):
        message = request.body["body"]
        response = self.second_response(message)
        return self.make_response(request, body={"body": response})

    def second_response(self, message):
        """
        Receive second request and send response
        """
        # deserialization
        message = json.loads(message)
        
        self.receive_bob_selected_message(message)
        response = self.send_message_with_secret(self.secret)

        # serialization
        return json.dumps(response)

    def make_response(self, request, body):
        response = ResponseMessage(self.client_info,
                                   request.server_info,
                                   body,
                                   phase_id=request.phase_id)
        return response

    # training part
    def train_init(self):
        return None
    
    def inference_init(self):
        return None

