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
Encoded number for ot comparison
"""
import numpy

import ot_config

MAX_BIT=ot_config.MAX_BIT
CHUNK_SIZE=ot_config.CHUNK_SIZE

class OTEncodedNumber(object):
    def __init__(self,
                 raw_number,
                 precision=2**32
                 ):
        assert isinstance(raw_number, (int, float)), "Only support int/float type"
        self.precision = precision
        self.raw_number = raw_number
        self.encoded_number = self.encoding(self.raw_number)
        self.encoded_number_array_binary = self.break_down_encoded_number(
            self.encoded_number)
        self.encoded_number_array_decimal = self.bin_bit_to_decimal(
            self.encoded_number_array_binary)
        return None

    def encoding(self, raw_number):
        encoded_number = int(raw_number * self.precision)
        positive = True
        if encoded_number >= 0:
            encoded_number = str(bin(encoded_number))[2:]
        else:
            positive = False
            encoded_number = str(bin(encoded_number))[3:]
        if len(encoded_number) > MAX_BIT - 1:
            raise ValueError("Current encoded number only supports %i bits but got %s bits"%(
                MAX_BIT, len(encoded_number)))
        if positive:
            return "1" + "0" * (MAX_BIT - len(encoded_number) - 1) + encoded_number
        else:
            return "0" * (MAX_BIT - len(encoded_number)) + encoded_number

    def break_down_encoded_number(self, encoded_number):
        return [encoded_number[i: i+CHUNK_SIZE] for i in range(0, len(encoded_number), CHUNK_SIZE)]

    def bin_bit_to_decimal(self, bin_bit):
        return [int(bi, 2) for bi in bin_bit]

    def compose_secret(self, bin_decimal):
        secrets = []
        for di in bin_decimal:
            si = []
            for i in range(2**CHUNK_SIZE):
                if i < di:
                    si.append(1)
                elif i > di:
                    si.append(0)
                else:
                    si.append(2)
            secrets.append(si)
        #return [[0 if i <= di else 1 for i in range(2**CHUNK_SIZE)] for di in bin_decimal]
        return secrets



def nextPowerOf2_32bit(n):
 
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n += 1
    return n