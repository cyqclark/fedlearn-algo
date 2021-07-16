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
Util functions
"""

# GMP's powmod has greater overhead than Python's pow, but is faster.
# From a quick experiment on our machine, this seems to be the break even:
_USE_MOD_FROM_GMP_SIZE = (1 << (8*2))

import random
import gmpy2


def invert(a, n):
    """
    The multiplicative inverse of a in the integers modulo n.

    Parameters
    ----------
    a: int
    The base integer
    
    n: int
    The modulo

    Returns
    -------
    a_inv: int 
    The integer a_inv where a * a_inv == 1 mod n
    """
    a_inv = int(gmpy2.invert(a, n))
    # according to documentation, gmpy2.invert might return 0 on
    # non-invertible element, although it seems to actually raise an
    # exception; for consistency, we always raise the exception
    if a_inv == 0:
        raise ZeroDivisionError('invert() no inverse exists')
    return a_inv


def powmod(a, x, n):
    """
    Uses GMP, if available, to do a^x mod n where a, x, n are integers.

    Parameters
    ----------
    a: int
    The base integer
    
    x: int
    The exponent
    
    n: int
    The modulo

    Returns
    -------
    m: int
    The integer m where m = (a ** x) % n
    """
    if a == 1:
        return 1
    if max(a, x, n) < _USE_MOD_FROM_GMP_SIZE:
        return pow(a, x, n)
    else:
        return int(gmpy2.powmod(a, x, n))


def isqrt(N):
    """
    returns the integer square root of N
    
    Parameters
    ----------
    N: int
        large integer
    
    Returns
    -------
    n: int
        the integer square root
    """
    return int(gmpy2.isqrt(N))



###############################################################
# some wrapper, these functions will be optimized in the future
###############################################################
def t_mod(x, y):
    # wrapper of gmpy2
    return gmpy2.t_mod(x, y)


def add(x, y):
    # wrapper of gmpy2
    return gmpy2.add(x, y)


def mul(x, y):
    # wrapper of gmpy2
    return gmpy2.mul(x, y)