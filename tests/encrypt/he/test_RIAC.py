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

import numpy

sys.path.append(os.getcwd()) # this might cause some issues
from core.encrypt.he import RandomizedIterativeAffine as RIAC

import unittest   # The test framework

_ASSERT_DECIMAL = 1e-6
_RELL_ERROR = 1e-6

class TestRIAC(unittest.TestCase):
    
    def __init__(self, *args, **kwargs):
        super(TestRIAC, self).__init__(*args, **kwargs)
        self.encrypt_key= RIAC.generate_keypair(key_round=2)
    
    def test_encrypt_int(self):
        plaintext = random.randint(-1e6, 1e6)
        ciphertext = self.encrypt_key.encrypt(plaintext)
        self.assertEqual(self.encrypt_key.decrypt(ciphertext), plaintext)

    def test_encrypt_float(self):
        plaintext = 1e4 * (random.random() - 0.5)
        ciphertext = self.encrypt_key.encrypt(plaintext)
        self.assertEqual(self.encrypt_key.decrypt(ciphertext), plaintext)

    def test_sum_int(self):
        p1 = random.randint(-1e6, 1e6)
        p2 = random.randint(-1e6, 1e6)
        c1 = self.encrypt_key.encrypt(p1)
        c2 = self.encrypt_key.encrypt(p2)
        self.assertEqual(self.encrypt_key.decrypt(c1 + c2), p1 + p2)
        
    def test_sum_float(self):
        # known issue p1 = 1.1e5, p2 = -2.1;e6
        p1 = 1e4 * (random.random() - 0.5)
        p2 = 1e4 * (random.random() - 0.5)
        c1 = self.encrypt_key.encrypt(p1)
        c2 = self.encrypt_key.encrypt(p2)
        self.assertEqual(self.encrypt_key.decrypt(c1 + c2), p1 + p2)
    
    def test_scalar_mult_int(self):
        plaintext = random.randint(-1e6, 1e6)
        n = random.randint(-1e6, 1e6)
        ciphertext = self.encrypt_key.encrypt(plaintext)
        self.assertEqual(self.encrypt_key.decrypt(n * ciphertext), n * plaintext)

    def test_scalar_mult_float(self):
        plaintext = 1e4 * (random.random() - 0.5)
        n = random.randint(-1e6, 1e6)
        ciphertext = self.encrypt_key.encrypt(plaintext)
        self.assertEqual(self.encrypt_key.decrypt(n * ciphertext), n * plaintext)

    def test_mult_add(self):
        plaintext1 = 1e4 * (random.random() - 0.5)
        plaintext2 = 1e4 * (random.random() - 0.5)
        n1 = random.randint(-1e6, 1e6)
        n2 = random.randint(-1e6, 1e6)
        ciphertext1 = self.encrypt_key.encrypt(plaintext1)
        ciphertext2 = self.encrypt_key.encrypt(plaintext2)
        decrypted = self.encrypt_key.decrypt(ciphertext1 * n1 * n2 + ciphertext2)
        result = plaintext1 * n1 * n2 + plaintext2
        if abs(decrypted - result) > _ASSERT_DECIMAL:
            if abs(decrypted/result - 1) < _RELL_ERROR:
                print("WARNING in test_mult_add")
                print("Abs diff %.8f is larger than %.8f,"%(abs(decrypted - result), _ASSERT_DECIMAL))
                print("but rel diff %.8f is smaller than %.8f"%(abs(decrypted/result - 1), _RELL_ERROR))
            else:
                self.assertAlmostEqual(decrypted, result, _ASSERT_DECIMAL, "Assert almost equal failed!")
        return None

    def test_mean_int(self):
        n = 1000
        plaintexts = numpy.random.randint(-1e6, 1e6, n)
        ciphertexts = numpy.array([self.encrypt_key.encrypt(pi) for pi in plaintexts],
                                   dtype=object)
        ciphertexts_mean = numpy.mean(ciphertexts)
        plaintexts_mean = numpy.mean(plaintexts)
        self.assertAlmostEqual(self.encrypt_key.decrypt(ciphertexts_mean),
                               plaintexts_mean,
                               delta=1e-6)

    def test_mean_float(self):
        n = 1000
        plaintexts = 1e4 * numpy.random.randn(n)
        ciphertexts = numpy.array([self.encrypt_key.encrypt(pi) for pi in plaintexts],
                                   dtype=object)
        ciphertexts_mean = numpy.mean(ciphertexts)
        plaintexts_mean = numpy.mean(plaintexts)
        self.assertAlmostEqual(self.encrypt_key.decrypt(ciphertexts_mean),
                               plaintexts_mean,
                               delta=1e-6)
        return None

    


if __name__ == '__main__':
    unittest.main()
