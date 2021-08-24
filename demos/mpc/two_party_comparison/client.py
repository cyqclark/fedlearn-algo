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
                 rsa_key=None,
                 rsa_key_size=512,
                 rand_message_bit=16):
        Alice1_nOT.__init__(self, rsa_key, rsa_key_size, rand_message_bit)
        self.control_map = {0: self.init_response,
                            1: self.second_response,
                            }
        self.reset_auto_machine()
        self.set_raw_number(raw_number)
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

    def init_response(self, message=None):
        """
        Receive start request and send response
        """
        response = self.send_key_with_rand_message()
        # serialization
        response["key"] = util.extract_rsa_key(response["key"])
        return json.dumps(response)

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

    def make_response(request, body):
        response = ResponseMessage(self.machine_info,
                                   request.server_info,
                                   body,
                                   phase_id=request.phase_id)
        return None

    # training part
    def train_init(self):
        return None
    
    def inference_init(self):
        return None

