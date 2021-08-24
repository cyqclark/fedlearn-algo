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
                 rand_message_bit=256):

        # Bob initialization
        Bob.__init__(self, rand_message_bit)
        self.control_map = {0: self.init_request,
                            1: self.second_request,
                            2: self.parse_final}
        self.reset_auto_machine()
        self.set_raw_number(raw_number)

        # Client initialization
        self.dict_functions = {}
        return None
        
    def set_raw_number(self, raw_number):
        self.raw_number = raw_number
        self.encoded_number = EncodedNumber.OTEncodedNumber(raw_number)
        self.secret = self.encoded_number.encoded_number_array_decimal
        return None

    def reset_auto_machine(self):
        self.current_state = -1
        return None

    def auto_send(self, message):
        """
        Auto send machine, experimental.
        """
        if self.current_state == -1:
            self.current_state = 0
        elif self.current_state == 0:
            self.current_state = 1
        elif self.current_state == 1:
            self.current_state = 2
        elif self.current_state == 2:
            print("Finish!")
            return None
        return self.control_map[self.current_state](message)

    

    def init_request(self, message=None):
        """
        Send start comparison request
        """
        return "start"

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

    def get_next_phase(self, phase):
        return None

    def is_inference_continue(self):
        return None

    def post_inference_session(self):
        return None

    # training part
    def init_training_control(self):
        return None

    def is_training_continue(self):
        return None
    
    def post_training_session(self):
        return None