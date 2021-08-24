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

import unittest   # The test framework

def test_comparison():

    active = coordinator.ActiveWrapper(1.)
    passive = client.PassiveWrapper(1.)

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

    return None


if __name__ == "__main__":
    test_comparison()