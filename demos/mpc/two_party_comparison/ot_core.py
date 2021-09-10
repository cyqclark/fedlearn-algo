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

"1-out-of-N OT based on 1-out-of-2 OT and RSA cryptosystem"

import random

import gmpy2
import rsa

import ot_config

MAX_BIT = ot_config.MAX_BIT
CHUNK_SIZE = ot_config.CHUNK_SIZE



class Alice1_nOT(object):
    def __init__(self,
                 rsa_key=None,
                 rsa_key_size=512,
                 rand_message_bit=16):

        self.rsa_key_size = rsa_key_size
        self.key_num = MAX_BIT//CHUNK_SIZE
        self.rand_message_bit = rand_message_bit

        # tmp code
        self.key_type = "RSA"
        self.key_size = rsa_key_size
        if rsa_key is None:
            self.key_dict = self.create_keys()
        else:
            assert isinstance(rsa_key, dict), "rsa_key should be dictonary!"
            self.key_dict = rsa_key
        return None

    def create_keys(self):
        if self.key_type == "RSA":
            return get_rsa_keys(self.key_num, self.key_size)
        else:
            raise NotImplementedError("Unsupported key type!")
    
    def send_key(self):
        return [self.key_dict[ki]["public_key"] for ki in range(self.key_num)]
    
    def send_rand_message(self):
        self.alice_rand_message_array = [self.send_single_rand_message() for _ in range(self.key_num)]
        return self.alice_rand_message_array

    def send_key_with_rand_message(self):
        message1 = self.send_key()
        message2 = self.send_rand_message()
        return {"key": message1,
                "rand_message": message2}

    def send_single_rand_message(self):
        self.alice_rand_message = [random.getrandbits(self.rand_message_bit) for _ in range(2**CHUNK_SIZE)]
        return self.alice_rand_message
    
    def receive_bob_selected_message(self, messages):
        self.alice_decrypt_array = []
        for i in range(self.key_num):
            message = messages[i]
            private_key = self.key_dict[i]["private_key"]
            alice_rand_message = self.alice_rand_message_array[i]
            tmp = [(message - messagei) for messagei in alice_rand_message]
            alice_decrypt = [gmpy2.powmod(tmpi, private_key.d, private_key.n) for tmpi in tmp]
            self.alice_decrypt_array.append(alice_decrypt)
        return None
    
    def send_message_with_secret(self, secret_array):
        messages = []
        for i in range(self.key_num):
            secret = secret_array[i]
            alice_decrypt = self.alice_decrypt_array[i]
            message = [int(Alice_decrypt_i) + Alice_i for Alice_decrypt_i, Alice_i
     in zip(alice_decrypt, secret)]
            messages.append(message)
        return messages
    
    
    
class Bob(object):
    def __init__(self, rand_message_bit=256):
        self.rand_message_bit = rand_message_bit
        return None
    
    
    def receive_rsa_key(self, message):
        self.public_key = message
        # duplicate
        self.key_dict = {i: messagei for i, messagei in enumerate(message)}
        self.key_num = len(self.key_dict.keys())
        return None
    
    
    def receive_alice_rand_message_array(self, message_array):
        self.alice_rand_message_array = message_array
        return None

    def receive_alice_key_with_rand_message_array(self, message):
        self.receive_rsa_key(message["key"])
        self.receive_alice_rand_message_array(message["rand_message"])
        return None
    
    def send_selected_message_array(self, indices):
        # get random k
        self.bob_k = [random.getrandbits(self.rand_message_bit) for i in range(self.key_num)]
        messages = []
        for i in range(self.key_num):
            idx = indices[i]
            pubkey = self.key_dict[i]
            message = self.alice_rand_message_array[i][idx] + gmpy2.powmod(
                self.bob_k[i], pubkey.e, pubkey.n)
            message = int(message) - pubkey.n if message > pubkey.n else int(message)
            messages.append(message)
        return messages
    
    def receive_secret(self, messages, indcies):
        self.received_secret_array = []
        for i in range(self.key_num):
            message = messages[i]
            idx = indcies[i]
            received_secret = message[idx] - self.bob_k[i]
            self.received_secret_array.append(received_secret)
        return self.received_secret_array

    def parse_result(self):
        for si in self.received_secret_array:
            if si == 2:
                pass
            elif si == 0:
                return False
            elif si == 1:
                return True
            else:
                raise ValueError("Unknown secret!")
        return False
        #return sum(self.received_secret_array) > 0


def get_rsa_keys(key_num, key_size):
        d = dict()
        for i in range(key_num):
            keys = rsa.newkeys(key_size)
            d[i] = {"public_key": keys[0],
                    "private_key": keys[1]}
        return d