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
Randomized Iterative Affine Cipher
The main code is partially grabbed from:
https://github.com/lyyanjiu1jia1/RandomizedIterativeAffineCipher
but with some optimizations.

# TODO:
# add capacity check.
"""

import math
import random

import numpy

from core.encrypt.he import util
from core.encrypt.he.EncodedNumber  import EncodedScaler

DEFAULT_KEYSIZE = 2048
ENCODING_BASE = 2
ENCODING_EXPONENT = 50

class IterativeAffineKey(object):
    """
    Key class for randomized iterative affine homomorphic encryption scheme.
    """
    def __init__(self, n, a, g, x, encode_precision=2**100):
        assert len(n) == len(a), "Length of n must be aligned with length of a"
        self.n = n
        self.a = a
        self.g = g
        self.x = x
        self.h = g * x % self.n[0]
        self.key_round = len(a)
        self.precision = encode_precision
        self.a_inv = self.mod_inverse()
        return None

    def encrypt(self, plaintext):
        return self.raw_encrypt(int(self.precision*plaintext))

    def decrypt(self, ciphertext):
        if ciphertext == 0:
            return 0
        else:
            return float(self.raw_decrypt(ciphertext) / self.precision)

    def random_encode(self, plaintext):
        y = random.SystemRandom().getrandbits(160)
        return y * self.g % self.n[0], (plaintext + y * self.h) % self.n[0]

    def decode(self, ciphertext):
        intermediate_result = (ciphertext.cipher2 - self.x * ciphertext.cipher1) % self.n[0]
        if intermediate_result / self.n[0] > 0.9:
            intermediate_result -= self.n[0]
        return intermediate_result / ciphertext.multiple ** ciphertext.mult_times

    def raw_encrypt(self, plaintext):
        plaintext = self.random_encode(plaintext)
        ciphertext = IterativeAffineCiphertext(plaintext[0], plaintext[1], self.n[-1])
        for i in range(self.key_round):
            ciphertext = self.raw_encrypt_round(ciphertext, i)
        return ciphertext

    def raw_encrypt_round(self, plaintext, round_index):
        return IterativeAffineCiphertext(
            plaintext.cipher1,
            (self.a[round_index] * plaintext.cipher2) % self.n[round_index],
            plaintext.n_final
        )

    def raw_decrypt(self, ciphertext):
        plaintext1 = ciphertext.cipher1
        plaintext2 = ciphertext.cipher2
        for i in range(self.key_round):
            plaintext1, plaintext2 = self.raw_decrypt_round(plaintext1, plaintext2, i)
        encoded_result = IterativeAffineCiphertext(
            cipher1=plaintext1,
            cipher2=plaintext2,
            n_final=ciphertext.n_final,
            multiple=ciphertext.multiple,
            mult_times=ciphertext.mult_times
        )
        return self.decode(encoded_result)

    def raw_decrypt_round(self, ciphertext1, ciphertext2, round_index):
        cur_n = self.n[self.key_round - 1 - round_index]
        cur_a_inv = self.a_inv[self.key_round - 1 - round_index]
        plaintext1 = ciphertext1 % cur_n
        plaintext2 = (cur_a_inv * (ciphertext2 % cur_n)) % cur_n
        if plaintext1 / cur_n > 0.9:
            plaintext1 -= cur_n
        if plaintext2 / cur_n > 0.9:
            plaintext2 -= cur_n
        return plaintext1, plaintext2

    def mod_inverse(self):
        a_inv = [0 for _ in self.a]
        for i in range(self.key_round):
            a_inv[i] = int(util.invert(self.a[i], self.n[i]))
        return a_inv


class IterativeAffineCiphertext(object):
    def __init__(self, cipher1, cipher2, n_final, multiple=2**50, mult_times=0):
        self.cipher1 = cipher1
        self.cipher2 = cipher2
        self.n_final = n_final
        self.multiple = multiple
        self.mult_times = mult_times

    def __add__(self, other):
        if isinstance(other, IterativeAffineCiphertext):
            if self.multiple != other.multiple or self.n_final != other.n_final:
                raise TypeError("Two addends must have equal multiples and n_finals")
            if self.mult_times > other.mult_times:
                mult_times_diff = self.mult_times - other.mult_times
                multiplier = util.powmod(self.multiple, mult_times_diff, self.n_final)
                cipher1 = util.t_mod(util.add(self.cipher1, util.mul(other.cipher1,
                    multiplier)), self.n_final)
                cipher2 = util.t_mod(util.add(self.cipher2, util.mul(other.cipher2,
                    multiplier)), self.n_final)
                return IterativeAffineCiphertext(
                    cipher1=cipher1,
                    cipher2=cipher2,
                    n_final=self.n_final,
                    multiple=self.multiple,
                    mult_times=self.mult_times
                )
            elif self.mult_times < other.mult_times:
                mult_times_diff = other.mult_times - self.mult_times
                multiplier = util.powmod(self.multiple, mult_times_diff, self.n_final)
                cipher1 = util.t_mod(util.add(util.mul(self.cipher1, multiplier),
                    other.cipher1), self.n_final)
                cipher2 = util.t_mod(util.add(util.mul(self.cipher2, multiplier),
                    other.cipher2), self.n_final)
                return IterativeAffineCiphertext(
                    cipher1=cipher1,
                    cipher2=cipher2,
                    n_final=self.n_final,
                    multiple=self.multiple,
                    mult_times=other.mult_times
                )
            else:
                return IterativeAffineCiphertext(
                    cipher1=(self.cipher1 + other.cipher1) % self.n_final,
                    cipher2=(self.cipher2 + other.cipher2) % self.n_final,
                    n_final=self.n_final,
                    multiple=self.multiple,
                    mult_times=other.mult_times
                )
        elif type(other) is int and other == 0:
            return self
        else:
            raise TypeError("Addition only supports IterativeAffineCiphertext and initialization with int zero")

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self + (other * -1)

    def __rsub__(self, other):
        return other + (self * -1)

    def __mul__(self, other):
        if type(other) is float or type(other) is numpy.float32 or type(other) is numpy.float64:
            other1 = int(other * self.multiple)
            cipher1 = util.t_mod(util.mul(self.cipher1, other1), self.n_final)
            cipher2 = util.t_mod(util.mul(self.cipher2, other1), self.n_final)
            return IterativeAffineCiphertext(
                cipher1=cipher1,
                cipher2=cipher2,
                n_final=self.n_final,
                multiple=self.multiple,
                mult_times=self.mult_times + 1
            )
        elif type(other) is int or type(other) is numpy.int32 or type(other) is numpy.int64:
            cipher1 = util.t_mod(util.mul(self.cipher1, int(other)), self.n_final)
            cipher2 = util.t_mod(util.mul(self.cipher2, int(other)), self.n_final)
            return IterativeAffineCiphertext(
                cipher1=cipher1,
                cipher2=cipher2,
                n_final=self.n_final,
                multiple=self.multiple,
                mult_times=self.mult_times
            )
        else:
            raise TypeError("Multiplication only supports native and numpy int and float")

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self.__mul__(1/other)


def generate_keypair(key_size=1024, key_round=2, encode_precision=2**100):
    """
    Generate public key and private key for iterative affine homomorphic encryption scheme.
    """
    # avoid potential collision
    ns = []
    minimum = encode_precision
    sizes = numpy.linspace(start=key_size//2, stop=key_size, num=key_round, dtype=int)
    for si in sizes:
        ni = randbit_with_minimum(si, minimum)
        minimum = ni * encode_precision
        ns.append(ni)
    a = [0 for _ in ns]
    for i in range(len(ns)):
        ni = ns[i]
        a_ratio = random.SystemRandom().random()
        ai = 0
        while True:
            a_size = int(key_size * a_ratio)
            if a_size == 0:
                continue
            ai = random.SystemRandom().getrandbits(a_size)
            if math.gcd(ni, ai) == 1:
                break
        a[i] = ai

    # pick a generator and a scalar
    g = random.SystemRandom().getrandbits(key_size // 10)
    x = random.SystemRandom().getrandbits(160)
    key = IterativeAffineKey(ns, a, g, x, encode_precision=encode_precision)
    return key


def randbit_with_minimum(key_size, minimum):
    if key_size * math.log(2) < math.log(minimum):
        raise ValueError("2 ** size < minimum, cannot randbit with this minimum")
    while True:
        n = random.SystemRandom().getrandbits(key_size)
        if n > minimum:
            break
    return n