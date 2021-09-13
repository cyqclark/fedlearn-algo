# import setup
# setup.deal_with_path()
from operator import matmul
import os
import sys
import numpy as np
import scipy.linalg
import copy
from timeit import default_timer as timer
sys.path.append(os.getcwd())

import random
from core.encrypt.he import RandomizedIterativeAffine as RIAC

n = 100000
m = 20
ind = 10
A = np.random.rand(n, m)
v0 = np.sum(A[:, 0:ind], axis=1)

B1 = np.insert(A, [ind], v0[:, np.newaxis], axis=1)
B1 = copy.deepcopy(A)
# print("B1=\n", B1)
t0 = timer()
Q1, R1 = np.linalg.qr(B1)
print("numpy QR: ", timer()-t0, "s")
# print("Q1=\n", Q1)
# print("R1=\n", R1)

B2 = np.insert(A, [ind], v0[:, np.newaxis], axis=1)
# print("B2=\n", B2)
t0 = timer()
Q2, R2, P2 = scipy.linalg.qr(B2, mode='economic', pivoting=True)
print("scipy QR: ", timer()-t0, "s")
# print("Q2=\n", Q2)
# print("R2=\n", R2)

print("end")