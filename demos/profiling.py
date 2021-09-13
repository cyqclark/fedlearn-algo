import setup
setup.deal_with_path()
import numpy as np

import cProfile
import pstats
import os
import linreg_qr

needUpdate = True

if not os.path.isfile('qrstats') or needUpdate:
    nSample = 1000000
    nFeatures = [0, 20, 30, 100, 10]
    clientIdWLabel = 0
    encryLv = 3
    nClient = len(nFeatures)

    # Prepare test data.
    nAllFeature = sum(nFeatures)
    X = np.random.rand(nSample, nAllFeature)
    while np.linalg.matrix_rank(X)<nAllFeature:
        X = np.random.rand(nSample, nAllFeature)
    beta0 = np.random.rand(nAllFeature, 1)
    Y = np.random.rand(nSample, 1)*0.05
    Y = Y + np.matmul(X, beta0)

    cProfile.run('linreg_qr.test_qr_unencrypted_q(X, Y, clientIdWLabel, nFeatures, encryLv)', 'qrstats')

p = pstats.Stats('qrstats')
p.strip_dirs().sort_stats(-1).print_stats()