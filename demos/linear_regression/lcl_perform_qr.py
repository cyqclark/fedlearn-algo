import setup
setup.deal_with_path()

import numpy as np
import scipy.linalg
import math
from timeit import default_timer as timer
import linreg_qr
import linreg_setup

def solve_weights_numpy_least_square(X, Y):
    features = np.append(X, np.ones([X.shape[0], 1]), axis=1)
    t0 = timer()
    weights = np.linalg.lstsq(features, Y, rcond=None)[0]
    print("Numpy least square: ", timer()-t0, "s")
    print("Numpy least square residule: ", np.linalg.norm(np.matmul(features, weights)-Y)/math.sqrt(features.shape[0]))
    print("")
    return weights

def solve_weights_qr(X, Y, colTrunc=False):
    features = np.append(X, np.ones([X.shape[0], 1]), axis=1)
    t = 0
    if not colTrunc:
        t0 = timer()
        q, r = np.linalg.qr(features)
        tDecomp = timer()-t0
        t += tDecomp
    else:
        t0 = timer()
        q, r, p = scipy.linalg.qr(features, mode='economic', pivoting=True)
        tDecomp = timer()-t0
        t += tDecomp
    t0 = timer()
    qtY = np.matmul(q.T, Y)
    tQtY = timer()-t0
    t += tQtY
    if not colTrunc:
        t0 = timer()
        weights = np.linalg.solve(r, qtY)
        tBS = timer()-t0
        t += tBS
    else:
        t0 = timer()
        weights = np.linalg.solve(r, qtY)
        invp = np.empty_like(p)
        invp[p] = np.arange(0, features.shape[1])
        weights[:, 0] = weights[invp, 0]
        tBS = timer()-t0
        t += tBS
    if not colTrunc:
        print("Numpy QR: ", t, "s")
    else:
        print("Scipy QR: ", t, "s")
    print("\tDecomposition: ", tDecomp, "s")
    print("\tApply Q: ", tQtY, "s")
    print("\tBacksolve: ", tBS, "s")
    print("QR residual: ", np.linalg.norm(np.matmul(features, weights)-Y)/math.sqrt(features.shape[0]))
    print("")
    return weights

def solve_weights_federated_qr_householder(X, Y, nFeatures, clientIdWLabel, encryLv=3, weightsExact=np.nan, colTrunc=False):
    features = np.append(X, np.ones([X.shape[0], 1]), axis=1)
    clientMap, coordinator = linreg_setup.setup_problem(X, Y, nFeatures, clientIdWLabel, qrMthd="HouseHolder", colTrunc=colTrunc)
    t = 0
    t0 = timer()
    linreg_qr.preprocessing_wo_constaint(clientMap, coordinator.machine_info_client, encryLv)
    tPreprocess = timer()-t0
    t += tPreprocess
    t0 = timer()
    linreg_qr.compute_qr_householder_unencrypted(clientMap, coordinator.machine_info_client)
    tDecomp = timer()-t0
    t += tDecomp
    t0 = timer()
    linreg_qr.apply_householder_unencrypted(clientMap, coordinator.machine_info_client)
    tQtY = timer()-t0
    t += tQtY
    t0 = timer()
    weights = linreg_qr.apply_back_solve_wo_constraint(clientMap, coordinator.machine_info_client)
    tBS = timer()-t0
    t += tBS
    print("Federated QR using HouseHolder reflection: ", t, "s")
    print("\tPreprocessing: ", tPreprocess, "s")
    print("\tDecomposition: ", tDecomp, "s")
    print("\tApply Q: ", tQtY, "s")
    print("\tBacksolve and postprocessing: ", tBS, "s")
    if np.isscalar(weightsExact) and np.isnan(weightsExact):
        features = np.append(X, np.ones([X.shape[0], 1]), axis=1)
        weightsExact = np.linalg.lstsq(features, Y, rcond=None)[0]
    err = weights - weightsExact
    print("Relative error based on inf-norm: ", np.linalg.norm(err, np.inf)/np.linalg.norm(weightsExact, np.inf))
    print("Relative error based on 2-norm: ", np.linalg.norm(err)/np.linalg.norm(weightsExact))
    print("Federated QR residual: ", np.linalg.norm(np.matmul(features, weights)-Y)/math.sqrt(features.shape[0]))
    print("")
    return weights

def solve_weights_federated_qr_gramschmidt(X, Y, nFeatures, clientIdWLabel, encryLv=3, weightsExact=np.nan, colTrunc=False):
    features = np.append(X, np.ones([X.shape[0], 1]), axis=1)
    clientMap, coordinator = linreg_setup.setup_problem(X, Y, nFeatures, clientIdWLabel, qrMthd="GramSchmidt", colTrunc=colTrunc)
    t = 0
    t0 = timer()
    linreg_qr.preprocessing_wo_constaint(clientMap, coordinator.machine_info_client, encryLv)
    tPreprocess = timer()-t0
    t += tPreprocess
    t0 = timer()
    linreg_qr.compute_qr_gramschmidt_unencrypted(clientMap, coordinator.machine_info_client)
    tDecomp = timer()-t0
    t += tDecomp
    t0 = timer()
    linreg_qr.apply_q_unencrypted(clientMap, coordinator.machine_info_client)
    tQtY = timer()-t0
    t += tQtY
    t0 = timer()
    weights = linreg_qr.apply_back_solve_wo_constraint(clientMap, coordinator.machine_info_client)
    tBS = timer()-t0
    t += tBS
    print("Federated QR using Gram-Schmidt method: ", t, "s")
    print("\tPreprocessing: ", tPreprocess, "s")
    print("\tDecomposition: ", tDecomp, "s")
    print("\tApply Q: ", tQtY, "s")
    print("\tBacksolve and postprocessing: ", tBS, "s")
    if np.isscalar(weightsExact) and np.isnan(weightsExact):
        features = np.append(X, np.ones([X.shape[0], 1]), axis=1)
        weightsExact = np.linalg.lstsq(features, Y, rcond=None)[0]
    err = weights - weightsExact
    print("Relative error based on inf-norm: ", np.linalg.norm(err, np.inf)/np.linalg.norm(weightsExact, np.inf))
    print("Relative error based on 2-norm: ", np.linalg.norm(err)/np.linalg.norm(weightsExact))
    print("Federated QR residual: ", np.linalg.norm(np.matmul(features, weights)-Y)/math.sqrt(features.shape[0]))
    print("")
    return weights

def solve_weights_federated_qr(X, Y, nFeatures, clientIdWLabel, qrMthd, encryLv=3, weightExact=np.nan, colTrunc=False):
    if qrMthd=="HouseHolder":
        weights = solve_weights_federated_qr_householder(X, Y, nFeatures, clientIdWLabel, encryLv, weightExact, colTrunc)
    elif qrMthd=="GramSchmidt":
        weights = solve_weights_federated_qr_gramschmidt(X, Y, nFeatures, clientIdWLabel, encryLv, weightExact, colTrunc)
    else:
        raise RuntimeError("QR by the specific method is not implemented yet.")
    return weights

def demo_performance_qr(qrMthd="HouseHolder"):
    print("*"*120)
    print("Demonstrate the performance of federated QR implemented by ", qrMthd, " method")
    dataMax = 10**4
    print("The features are generated with variate scales. The largest scale is ", dataMax)
    print("The label values are generated by adding the multiplication of the features and some random generated weights with some perturbation")
    clientIdWLabel = 1
    print("The label is kept by beh second client")

    print("="*60)
    print("Problem1: Performance on small scale problem without column truncation")
    nSample = 1000
    nFeatures = [3, 3, 5, 11, 8]
    colTrunc = False
    print("The test data are generated randomly and consist of ", nSample, " samples and ", sum(nFeatures), " features")
    encryLv = 3
    XTrain, YTrain, XInfer = linreg_setup.generate_fullrank_test_data(nSample, 0, nFeatures, dataMax)
    weightsExact = solve_weights_numpy_least_square(XTrain, YTrain)
    solve_weights_qr(XTrain, YTrain, colTrunc)
    solve_weights_federated_qr(XTrain, YTrain, nFeatures, clientIdWLabel, qrMthd, encryLv, weightsExact, colTrunc)

    print("="*60)
    print("Problem2: Performance on large scale problem without column truncation")
    nSample = 1000000
    nFeatures = [40, 20, 30, 100, 10]
    colTrunc = False
    print("The test data are generated randomly and consist of ", nSample, " samples and ", sum(nFeatures), " features")
    encryLv = 3
    XTrain, YTrain, XInfer = linreg_setup.generate_fullrank_test_data(nSample, 0, nFeatures, dataMax)
    weightsExact = solve_weights_numpy_least_square(XTrain, YTrain)
    solve_weights_qr(XTrain, YTrain, colTrunc)
    solve_weights_federated_qr(XTrain, YTrain, nFeatures, clientIdWLabel, qrMthd, encryLv, weightsExact, colTrunc)

    print("="*60)
    print("Problem3: Performance on small scale problem with column truncation")
    nSample = 1000
    nFeatures = [3, 3, 5, 11, 8]
    colTrunc = True
    print("The test data are generated randomly and consist of ", nSample, " samples and ", sum(nFeatures), " features")
    encryLv = 3
    XTrain, YTrain, XInfer = linreg_setup.generate_fullrank_test_data(nSample, 0, nFeatures, dataMax)
    weightsExact = solve_weights_numpy_least_square(XTrain, YTrain)
    solve_weights_qr(XTrain, YTrain, colTrunc)
    solve_weights_federated_qr(XTrain, YTrain, nFeatures, clientIdWLabel, qrMthd, encryLv, weightsExact, colTrunc)

    print("="*60)
    print("Problem4: Performance on large scale problem with column truncation")
    nSample = 1000000
    nFeatures = [40, 20, 30, 100, 10]
    colTrunc = True
    print("The test data are generated randomly and consist of ", nSample, " samples and ", sum(nFeatures), " features")
    encryLv = 3
    XTrain, YTrain, XInfer = linreg_setup.generate_fullrank_test_data(nSample, 0, nFeatures, dataMax)
    weightsExact = solve_weights_numpy_least_square(XTrain, YTrain)
    solve_weights_qr(XTrain, YTrain, colTrunc)
    solve_weights_federated_qr(XTrain, YTrain, nFeatures, clientIdWLabel, qrMthd, encryLv, weightsExact, colTrunc)


# Start the demo.
if __name__ == "__main__":
    demo_performance_qr(qrMthd="HouseHolder")
    demo_performance_qr(qrMthd="GramSchmidt")