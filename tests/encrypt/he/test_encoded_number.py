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
Test script
"""
import os
import random
import sys

sys.path.append(os.getcwd()) # this might cause some issues
from core.encrypt.he import EncodedNumber

import unittest   # The test framework

class TestPaillier(unittest.TestCase):
    
    def test_encoded_int(self):
        plaintext = random.randint(-1e6, 1e6)
        enc = EncodedNumber.EncodedScaler(plaintext, 2, 50)
        self.assertEqual(enc.get_plaintext(), plaintext)
        self.assertEqual(enc.plaintext, plaintext)

        encoded_number = enc.get_encoded_number()

        enc1 = EncodedNumber.EncodedScaler.convert_from_encoded_number(
            encoded_number=encoded_number, base=2, exponent=50)
        
        self.assertEqual(enc1.plaintext, plaintext)
        self.assertEqual(enc1.get_plaintext(), plaintext)

    def test_encoded_float(self):
        scale = 1e5
        plaintext = (random.random() - 0.5) * scale
        enc = EncodedNumber.EncodedScaler(plaintext, 2, 50)
        self.assertEqual(enc.get_plaintext(), plaintext)
        self.assertEqual(enc.plaintext, plaintext)

        encoded_number = enc.get_encoded_number()

        enc1 = EncodedNumber.EncodedScaler.convert_from_encoded_number(
            encoded_number=encoded_number, base=2, exponent=50)
        
        self.assertEqual(enc1.plaintext, plaintext)
        self.assertEqual(enc1.get_plaintext(), plaintext)

        


if __name__ == '__main__':
    unittest.main()
