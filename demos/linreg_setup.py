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

"""Example code of federated linear regression client based on QR decompositon.


Example
-------
TBA::

    $ TBA


"""

import setup
setup.deal_with_path()

import numpy as np
import qrClient as qrClient
import qrCoordinator as qrCoord
import socket
import pandas
from core.entity.common.machineinfo import MachineInfo


def generate_fullrank_test_data(nTrain, nInfer, nFeatures, dataMax=10**4):
    """
    Generate random full rank test data for training and inference.

    Parameters
    ----------
    nTrain : int
        The number of training samples.
    nInfer : int
        The number of inference samples.
    nFeatures : List
        The number of features in each clients.
    dataMax : float
        The maximal number for scaling the randomly generated features.

    Returns
    -------
    float
        The alligned training data for all the clients.
    float
        The generated label.
    float
        The alligned inference data for all the clients.
    """
    nGlbFeature = sum(nFeatures)
    X = np.random.rand(nTrain+nInfer, nGlbFeature)
    while np.linalg.matrix_rank(X)<nGlbFeature:
        X = np.random.rand(nTrain, nGlbFeature)
    X = X*2-1
    scaling = np.random.rand(nGlbFeature)*dataMax
    X = X*scaling[np.newaxis, :]
    XTrain = X[0:nTrain, :]
    XInfer = X[nTrain:, :]
    beta0 = np.random.rand(nGlbFeature+1, 1)
    YTrain = np.matmul(XTrain, beta0[0:-1, 0].reshape([nGlbFeature, 1]))+beta0[-1, 0]
    perturbation = (np.random.rand(nTrain, 1)*2-1)*0.05
    YTrain *= perturbation
    return XTrain, YTrain, XInfer

def generate_rankdefi_test_data(nTrain, nInfer, nFeatures, dataMax=10**4, rankDefiColIds=[-1]):
    """
    Generate random full rank test data for training and inference.

    Parameters
    ----------
    nTrain : int
        The number of training samples.
    nInfer : int
        The number of inference samples.
    nFeatures : List
        The number of features in each clients.
    dataMax : float
        The maximal number for scaling the randomly generated features.
    rankDefiColIds : List
        The global column indices for those features that are linearly dependent on the their
        previous features in order.

    Returns
    -------
    float
        The alligned training data for all the clients.
    float
        The generated label.
    float
        The alligned inference data for all the clients.
    """
    nGlbFeature = sum(nFeatures)
    X = np.random.rand(nTrain+nInfer, nGlbFeature)
    while np.linalg.matrix_rank(X)<nGlbFeature:
        X = np.random.rand(nTrain, nGlbFeature)
    X = X*2-1
    scaling = np.random.rand(nGlbFeature)*dataMax
    X = X*scaling[np.newaxis, :]
    for colId in rankDefiColIds:
        X[:, colId] = np.sum(X[:, 0:colId], axis=1)
    XTrain = X[0:nTrain, :]
    XInfer = X[nTrain:, :]
    beta0 = np.random.rand(nGlbFeature+1, 1)
    YTrain = np.matmul(XTrain, beta0[0:-1, 0].reshape([nGlbFeature, 1]))+beta0[-1, 0]
    perturbation = (np.random.rand(nTrain, 1)*2-1)*0.05
    YTrain *= perturbation
    return XTrain, YTrain, XInfer

def setup_problem(X0, Y0, nFeatures, clientIdWLabel, encryLv=3, X1=None, qrMthd="GramSchmidt", colTrunc=False):
    """
    Set up the coordinator and all the clients based on the randomly generated training and inference data.

    Parameter
    ---------
    X0 : float
        The alligned training data for all the clients.
    Y0 : float
        The generated label.
    nFeatures : List
        The number of features in each clients.
    clientIdWLabel : int
        The index of the active client.
    encryDeg : int
        The minimal number of columns for preprocessing of the feature matrix.
    X1 : float
        The alligned inference data for all the clients.
    qrMthd : string
        The QR decomposition method, HouseHolder or Gram-Schmidts+builtin.
    colTrunc : bool
        Do the column pivoting and truncation or not.

    Returns
    -------
    List
        List of clients.
    qrCoorinator
        The coordinator object.
    """
    nClient = len(nFeatures)
    clientInfos = []
    clientMap = {}
    portId0 = "899"

    # Generate client information and client objects.
    ip = socket.gethostbyname(socket.gethostname())
    glbFeatureId0 = 0
    for ii in range(0, nClient):
        portId = portId0+str(ii+1)
        tokenFormat = "%s:"+portId
        if X1 is None or X1.shape[0]==0:
            inferFeatures = None
        else:
            inferFeatures = X1[:, glbFeatureId0:glbFeatureId0+nFeatures[ii]]
        parameter = {}
        clientInfos.append(MachineInfo(ip=ip, port=portId, token=tokenFormat%ip))
        if ii!=clientIdWLabel:
            label = None
        else:
            label = Y0
        dataset = {
            "trainFeatures": X0[:, glbFeatureId0:glbFeatureId0+nFeatures[ii]],
            "label": label,
            "inferFeatures": inferFeatures
        }
        clientMap[clientInfos[ii]] = qrClient.LinRegQRClient(clientInfos[ii], parameter, dataset, remote=True)
        glbFeatureId0 += nFeatures[ii]

    # Generate coordinate information and coordinate objects.
    parameter = {
        "qrMthd": qrMthd,
        "colTrunc": colTrunc,
        "nFeatures": np.array(nFeatures),
        "clientIdWLabel": clientIdWLabel,
        "encryLv": encryLv,
        "rankDef": 0
    }
    portId = portId0+str(0)
    tokenFormat = "%s:"+portId
    coordinatorInfo = MachineInfo(ip=ip, port=portId, token=tokenFormat%ip)
    coordinator = qrCoord.LinRegQRCoordinator(coordinatorInfo, clientInfos, parameter, remote=True)
    return clientMap, coordinator

def setup_problem_4_prestored_data(encryLv=3, qrMthd="GramSchmidt", colTrunc=False):
    """
    Set up the coordinator and all the clients based on the prestored classification training and inference data.

    Parameter
    ---------
    encryDeg : int
        The minimal number of columns for preprocessing of the feature matrix.
    qrMthd : string
        The QR decomposition method, HouseHolder or Gram-Schmidts+builtin.
    colTrunc : bool
        Do the column pivoting and truncation or not.

    Returns
    -------
    List
        List of clients.
    qrCoorinator
        The coordinator object.
    """
    # load and align train data
    g1 = pandas.read_csv("data/classificationA/train0.csv")
    g2 = pandas.read_csv("data/classificationA/train1.csv")
    g3 = pandas.read_csv("data/classificationA/train2.csv")
    uid = g1.loc[:, ["uid"]]
    uid = pandas.merge(uid, g2.loc[:, ["uid"]], on="uid", how="inner")
    uid = pandas.merge(uid, g3.loc[:, ["uid"]], on="uid", how="inner")
    g1 = pandas.merge(uid, g1, on="uid", how="inner")
    g2 = pandas.merge(uid, g2, on="uid", how="inner")
    g3 = pandas.merge(uid, g3, on="uid", how="inner")
    label = g1.Outcome.values.astype(float)
    label = label.reshape([label.size, 1])
    dataset1 = {
        "label": label, 
        "trainFeatures": g1.loc[:, g1.columns[1:-1]].values
    }
    dataset2 = {
        "label": None,
        "trainFeatures": g2.loc[:, g2.columns[1:]].values
    }
    dataset3 = {
        "label": None,
        "trainFeatures": g3.loc[:, g3.columns[1:]].values
    }
    XTrain = g1.loc[:, g1.columns[1:-1]].values
    XTrain = np.append(XTrain, g2.loc[:, g2.columns[1:]].values, axis=1)
    XTrain = np.append(XTrain, g3.loc[:, g3.columns[1:]].values, axis=1)
    YTrain = label

    # load and align inference data
    g1 = pandas.read_csv("data/classificationA/inference0.csv")
    g2 = pandas.read_csv("data/classificationA/inference1.csv")
    g3 = pandas.read_csv("data/classificationA/inference2.csv")
    uid = g1.loc[:, ["uid"]]
    uid = pandas.merge(uid, g2.loc[:, ["uid"]], on="uid", how="inner")
    uid = pandas.merge(uid, g3.loc[:, ["uid"]], on="uid", how="inner")
    g1 = pandas.merge(uid, g1, on="uid", how="inner")
    g2 = pandas.merge(uid, g2, on="uid", how="inner")
    g3 = pandas.merge(uid, g3, on="uid", how="inner")
    dataset1["inferFeatures"] = g1.loc[:, g1.columns[1:]].values
    dataset2["inferFeatures"] = g2.loc[:, g2.columns[1:]].values
    dataset3["inferFeatures"] = g3.loc[:, g3.columns[1:]].values
    XInfer = g1.loc[:, g1.columns[1:]].values
    XInfer = np.append(XInfer, g2.loc[:, g2.columns[1:]].values, axis=1)
    XInfer = np.append(XInfer, g3.loc[:, g3.columns[1:]].values, axis=1)

    # set the machine info for coordinator and client
    ip = socket.gethostbyname(socket.gethostname())
    coordinatorInfo = MachineInfo(ip=ip, port="8890", token="%s:8890"%ip)
    clientInfo1 = MachineInfo(ip=ip, port="8891", token="%s:8891"%ip)
    clientInfo2 = MachineInfo(ip=ip, port="8892", token="%s:8892"%ip)
    clientInfo3 = MachineInfo(ip=ip, port="8893", token="%s:8893"%ip)
    
    nFeatures = [dataset1["trainFeatures"].shape[1], dataset2["trainFeatures"].shape[1], dataset3["trainFeatures"].shape[1]]
    clientIdWLabel = 0
    parameter = {
        "qrMthd": qrMthd,
        "colTrunc": colTrunc,
        "nFeatures": np.array(nFeatures),
        "clientIdWLabel": clientIdWLabel,
        "encryLv": encryLv,
        "rankDef": 0
    }

    # create client and coordinator
    client1 = qrClient.LinRegQRClient(clientInfo1, parameter, dataset1, remote=True)
    client2 = qrClient.LinRegQRClient(clientInfo2, parameter, dataset2, remote=True)
    client3 = qrClient.LinRegQRClient(clientInfo3, parameter, dataset3, remote=True)
    clientMap = {
        clientInfo1: client1,
        clientInfo2: client2,
        clientInfo3: client3
    }
    clientInfos = [clientInfo1, clientInfo2, clientInfo3]
    coordinator = qrCoord.LinRegQRCoordinator(coordinatorInfo, clientInfos, parameter, remote=True)
    return clientMap, coordinator, XTrain, YTrain, XInfer

def switch_order_4_dataset(X0, nFeatures, clientIdWLabel, X1=None):
    """
    Reorder the randomly generated training and inference data so that the data of the active client is in the
    last. The output can be used by Numpy builtin least squares method to generate the comparison results.
    
    Parameters
    ----------
    X0 : float
        The alligned training data for all the clients.
    nFeatures : List
        The number of features in each clients.
    clientIdWLabel : int
        The index of the active client.
    X1 : float
        The alligned inference data for all the clients.

    Returns
    -------
    float
        The reordered training data.
    float
        The reordered inference data.
    """
    nClient = len(nFeatures)
    X0Reordered = np.zeros_like(X0)
    nFeaturesReordered = np.empty(0, dtype=int)
    if X1 is None or X1.shape[0]==0:
        X1Reordered = None
    else:
        X1Reordered = np.zeros_like(X1)

    # Generate client information and client objects.
    glbDataId0 = 0
    glbFeatureId0 = 0
    for ii in range(0, nClient):
        if ii==clientIdWLabel:
            trainFeatures4ClientWLabel = X0[:, glbDataId0:glbDataId0+nFeatures[ii]]
            if X1 is not None and X1.shape[0]!=0:
                inferFeatures4ClientWLabel = X1[:, glbDataId0:glbDataId0+nFeatures[ii]]
        else:
            X0Reordered[:, glbFeatureId0:glbFeatureId0+nFeatures[ii]] = X0[:, glbDataId0:glbDataId0+nFeatures[ii]]
            nFeaturesReordered = np.append(nFeaturesReordered, nFeatures[ii])
            if X1 is not None and X1.shape[0]!=0:
                X1Reordered[:, glbFeatureId0:glbFeatureId0+nFeatures[ii]] = X1[:, glbDataId0:glbDataId0+nFeatures[ii]]
            glbFeatureId0 += nFeatures[ii]
        glbDataId0 += nFeatures[ii]
    X0Reordered[:, glbFeatureId0:glbFeatureId0+nFeatures[clientIdWLabel]] = trainFeatures4ClientWLabel
    nFeaturesReordered = np.append(nFeaturesReordered, nFeatures[clientIdWLabel])
    if X1 is not None and X1.shape[0]!=0:
        X1Reordered[:, glbFeatureId0:glbFeatureId0+nFeatures[clientIdWLabel]] = inferFeatures4ClientWLabel
    clientIdWLabel = nClient-1
    return X0Reordered, X1Reordered