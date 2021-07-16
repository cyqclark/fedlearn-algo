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
EncodedNumber class
"""


_BASE=2
_BASE_EXPONENT=50

class EncodedScaler(object):
    def __init__(self,
                 plaintext,
                 base=None,
                 exponent=None):
        """
        Encoding the plaintext as an large integer:
            enc(x) = int(x * base ** exponent)
        """
        self.plaintext = plaintext
        self.encoding_base = _BASE if base is None else base
        self.exponent = _BASE_EXPONENT if exponent is None else exponent
        self.encoded_number = int(self.plaintext * self.encoding_base ** self.exponent)
        return None

    def get_encoded_number(self):
        return self.encoded_number

    def get_plaintext(self):
        return self.plaintext

    @classmethod
    def convert_from_encoded_number(cls,
                                    encoded_number,
                                    base,
                                    exponent):
        plaintext = encoded_number / (base ** exponent)
        return cls(plaintext, base, exponent)


class EncodedVector(object):
    def __init__(self):
        return None


class EncodedMatrix(object):
    def __init__(self):
        return None
