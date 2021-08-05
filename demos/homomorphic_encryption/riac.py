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
A demo of RIAC usage.
"""

# a temp fix for path problem
import sys
path0 = sys.path[0]
if path0.split("/")[-1] == "homomorphic_encryption":
    sys.path.append("/".join(path0.split("/")[:-2]))

import random

import core.encrypt.he.RandomizedIterativeAffine as RIAC

def demo_encrypt_decrypt():
    """
    This demo code shows how to use RIAC in encryption decryption
    """
    print("\n============================Encryption Decryption Demo==================================")
    key = RIAC.generate_keypair() # get encryption key

    print("\nInteger demo: ")
    plaintext = random.randint(-1e6, 1e6) # get a random plaintext in int
    ciphertext = key.encrypt(plaintext) # encryption
    decrypted = key.decrypt(ciphertext) # decryption
    print("Plaintext: %i; decrypted ciphertext: %.6f; diff=%.6f"%(plaintext, decrypted, plaintext-decrypted))

    print("\nFloat demo: ")
    plaintext = 1e6 * (random.random() - 0.5) # get a random plaintext in float
    ciphertext = key.encrypt(plaintext) # encryption
    decrypted = key.decrypt(ciphertext) # decryption
    print("Plaintext: %.6f; decrypted ciphertext: %.6f; diff=%.6f"%(plaintext, decrypted, plaintext-decrypted))
    return None

def demo_add():
    """
    In this demo function we show how to use RIAC in addition
    """
    key = RIAC.generate_keypair() # get encryption key

    print("\n============================Addition Demo==================================")
    # sum of two integers
    print("\nAddition of two integers: ")
    
    plaintext1 = random.randint(-1e6, 1e6) # get a random plaintext in int
    ciphertext1 = key.encrypt(plaintext1) # encryption
    
    plaintext2 = random.randint(-1e6, 1e6) # get a random plaintext in int
    ciphertext2 = key.encrypt(plaintext2) # encryption

    decrypted = key.decrypt(ciphertext1 + ciphertext2) # addition and decryption
    print("Plaintext: %i + %i = %i; decrypted ciphertext: %.6f; diff=%.6f"%(
        plaintext1, plaintext2, plaintext1 + plaintext2,
        decrypted, plaintext1 + plaintext2 - decrypted))


    # sum of an integer and a float
    print("\nAddition of an integer and a float: ")
    
    plaintext1 = random.randint(-1e6, 1e6) # get a random plaintext in int
    ciphertext1 = key.encrypt(plaintext1) # encryption
    
    plaintext2 = 1e6 * (random.random() - 0.5) # get a random plaintext in float
    ciphertext2 = key.encrypt(plaintext2) # encryption

    decrypted = key.decrypt(ciphertext1 + ciphertext2) # addition and decryption
    print("Plaintext: %i + %.6f = %.6f; decrypted ciphertext: %.6f; diff=%.6f"%(
        plaintext1, plaintext2, plaintext1 + plaintext2,
        decrypted, plaintext1 + plaintext2 - decrypted))


    # sum of two floats
    print("\nAddition of two floats: ")
    
    plaintext1 = 1e6 * (random.random() - 0.5) # get a random plaintext in float
    ciphertext1 = key.encrypt(plaintext1) # encryption
    
    plaintext2 = 1e6 * (random.random() - 0.5) # get a random plaintext in float
    ciphertext2 = key.encrypt(plaintext2) # encryption

    decrypted = key.decrypt(ciphertext1 + ciphertext2) # addition and decryption
    print("Plaintext: %.6f + %.6f = %.6f; decrypted ciphertext: %.6f; diff=%.6f"%(
        plaintext1, plaintext2, plaintext1 + plaintext2,
        decrypted, plaintext1 + plaintext2 - decrypted))
    return None

def demo_scalar_multiplication():
    """
    In this demo function we show how to use RIAC in scalar multiplication
    """
    key = RIAC.generate_keypair() # get encryption key

    print("\n============================Scalar Multiplication Demo==================================")
    
    plaintext1 = random.randint(-1e6, 1e6) # get a random plaintext in int
    ciphertext1 = key.encrypt(plaintext1) # encryption
    
    m = 1e6 * (random.random() - 0.5) # get a random plaintext in float

    decrypted = key.decrypt(ciphertext1 * m) # scalar multiplication and decryption
    print("Plaintext: %i * %.6f = %.6f; decrypted ciphertext: %.6f; diff=%.6f"%(
        plaintext1, m, plaintext1 * m, decrypted, plaintext1 * m - decrypted))
    return None


if __name__ == "__main__":
    demo_encrypt_decrypt()
    demo_add()
    demo_scalar_multiplication()