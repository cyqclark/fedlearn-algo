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


# Functionals.
def obtain_global_col_id(clientMap, clientInfos):
    """
    Collect the global column ID of all the clients.

    Parameters
    ----------
    clientMap : List
        The list of qrClient objects.
    clientInfos : List
        The list of machine information of the corresponding qrClient objects.

    Returns
    -------
    numpy.array
        The global column indices of all the clients.
    """
    nClient = len(clientInfos)
    glbColId = np.empty(0, dtype=int)
    for ii in range(0, nClient):
        if clientMap[clientInfos[ii]].nFeature>0:
            glbColId = np.append(glbColId, clientMap[clientInfos[ii]].glbColIdInit)
    return glbColId

def obtain_global_col_id_4_features(clientMap, clientInfos):
    """
    Collect the global feature ID of all the clients. The constant term is put at the end.

    Parameters
    ----------
    clientMap : List
        The list of qrClient objects.
    clientInfos : List
        The list of machine information of the corresponding qrClient objects.

    Returns
    -------
    numpy.array
        The global feature ID of all the clients. The constant term is put at the end.
    """
    nClient = len(clientInfos)
    glbColId4Features = np.empty(0, dtype=int)
    for ii in range(0, nClient):
        if clientMap[clientInfos[ii]].nFeature>0:
            if not clientMap[clientInfos[ii]].wConst:
                glbColId4Features = np.append(glbColId4Features, clientMap[clientInfos[ii]].glbColIdInit[clientMap[clientInfos[ii]].lclFeatureId])
            else:
                glbColId4Features = np.append(glbColId4Features, clientMap[clientInfos[ii]].glbColIdInit[clientMap[clientInfos[ii]].lclFeatureId[1:]])
                glbConstId = clientMap[clientInfos[ii]].glbColIdInit[clientMap[clientInfos[ii]].lclFeatureId[0]]
    glbColId4Features = np.append(glbColId4Features, glbConstId)
    return glbColId4Features

def obtain_global_weights(clientMap, clientInfos):
    """
    Collect the computed weights of all the clients.

    Parameters
    ----------
    clientMap : List
        The list of qrClient objects.
    clientInfos : List
        The list of machine information of the corresponding qrClient objects.

    Returns
    -------
    numpy.array
        The computed weights of all the clients. The weights corresponding to the constant term is at the last position.
    """
    nClient = len(clientInfos)
    for ii in range(0, nClient):
        if clientMap[clientInfos[ii]].wLabel:
            clientIdWLabel = ii
            break
    for ii in range(0, nClient):
        if ii!=clientIdWLabel:
            glbColId4Features = clientMap[clientInfos[ii]].glbColIdInit[clientMap[clientInfos[ii]].lclFeatureId]
            clientMap[clientInfos[clientIdWLabel]].fetch_beta_entries(glbColId4Features, clientMap[clientInfos[ii]].get_local_weights())
        else:
            glbColId4Features = clientMap[clientInfos[ii]].glbColIdInit[clientMap[clientInfos[ii]].lclFeatureId]
            if glbColId4Features.size>0:
                clientMap[clientInfos[ii]].beta[glbColId4Features, 0] = clientMap[clientInfos[ii]].lclWts.reshape(-1)
    return clientMap[clientInfos[clientIdWLabel]].get_global_weights(obtain_global_col_id_4_features(clientMap, clientInfos))

# For preprocessing.
def preprocessing_wo_constaint(clientMap, clientInfos, encryLv, colTrunc=False):
    """
    Preprocess the feature data and add the constraints inside the feature matrices.

    Parameters
    ----------
    clientMap : List
        The list of qrClient objects.
    clientInfos : List
        The list of machine information of the corresponding qrClient objects.
    encryLv : int
        The least number of columns the feature matrix of a single client should have to protect its privacy.
    colTrunc : bool
        Do the column pivoting and truncation or not.
    """
    nClient = len(clientInfos)
    for ii in range(0, nClient):
        clientMap[clientInfos[ii]].colTrunc = colTrunc
    mostFeatureClient = 0
    for ii in range(1, nClient):
        if clientMap[clientInfos[ii]].nFeature>clientMap[clientInfos[mostFeatureClient]].nFeature:
            mostFeatureClient = ii
    clientMap[clientInfos[mostFeatureClient]].add_const_feature()  # Add constant feature to the client with most features to protect information security of those with less features.
    nGlbCol = 0
    nGlbConstr = 0
    for ii in range(0, nClient):
        nGlbCol = clientMap[clientInfos[ii]].preprocess_features_4_qr(encryLv, nGlbCol)
        nGlbConstr += clientMap[clientInfos[ii]].nConstr
    rowId0 = clientMap[clientInfos[0]].nSample
    for ii in range(0, nClient):
        rowId0 = clientMap[clientInfos[ii]].add_oblivious_constraint_4_qr(rowId0, nGlbConstr)

# For computing QR decomposition.
def compute_qr_householder_unencrypted(clientMap, clientInfos):
    """
    Compute the QR decomposition using Householder reflection.

    Parameters
    ----------
    clientMap : List
        The list of qrClient objects.
    clientInfos : List
        The list of machine information of the corresponding qrClient objects.
    """
    nClient = len(clientInfos)
    glbRankDefi = 0
    for ii in range(0, nClient):
        clientMap[clientInfos[ii]].update_global_row_indices(glbRankDefi)
        clientMap[clientInfos[ii]].compute_qr_w_householder()
        glbRankDefi += clientMap[clientInfos[ii]].get_local_rank_deficiency()
        for jj in range(ii+1, nClient):
            householders = clientMap[clientInfos[ii]].obtain_householders_unencrypted()
            clientMap[clientInfos[jj]].apply_external_householder_2_features_unencrypted(householders)

def compute_qr_gramschmidt_unencrypted(clientMap, clientInfos):
    """
    Compute the QR decomposition using Numpy/Scipy builtin algorithm and block Gram-Schimdt method.

    Parameters
    ----------
    clientMap : List
        The list of qrClient objects.
    clientInfos : List
        The list of machine information of the corresponding qrClient objects.
    """
    nClient = len(clientInfos)
    glbRankDefi = 0
    for ii in range(0, nClient):
        clientMap[clientInfos[ii]].update_global_row_indices(glbRankDefi)
        clientMap[clientInfos[ii]].compute_qr_w_builtin()
        glbRankDefi += clientMap[clientInfos[ii]].get_local_rank_deficiency()
        for jj in range(ii+1, nClient):
            Q = clientMap[clientInfos[ii]].obtain_q_unencrypted()
            clientMap[clientInfos[jj]].apply_external_q_2_features_by_gs_unencrypted(Q)

# For applying Q matrix.
def apply_householder_unencrypted(clientMap, clientInfos):
    """
    Apply Q matrix by means of a series of householder reflections to the label.

    Parameters
    ----------
    clientMap : List
        The list of qrClient objects.
    clientInfos : List
        The list of machine information of the corresponding qrClient objects.
    """
    nClient = len(clientInfos)
    for ii in range(0, nClient):
        if clientMap[clientInfos[ii]].wLabel:
            clientIdWLabel = ii
            break
    for ii in range(0, nClient):
        if clientMap[clientInfos[ii]].wLabel:
            clientMap[clientInfos[ii]].apply_internal_householder_2_label()
        else:
            householders = clientMap[clientInfos[ii]].obtain_householders_unencrypted()
            clientMap[clientInfos[clientIdWLabel]].apply_external_householder_2_label_unencrypted(householders)
    nGlbCol = obtain_global_col_id(clientMap, clientInfos).size
    clientMap[clientInfos[clientIdWLabel]].initialize_beta(nGlbCol)

def apply_q_unencrypted(clientMap, clientInfos):
    """
    Apply Q matrix to the label by means of the Q matrix computed by Numpy/Scipy builtin algorithm.

    Parameters
    ----------
    clientMap : List
        The list of qrClient objects.
    clientInfos : List
        The list of machine information of the corresponding qrClient objects.
    """
    nClient = len(clientInfos)
    for ii in range(0, nClient):
        if clientMap[clientInfos[ii]].wLabel:
            clientIdWLabel = ii
            break
    nGlbCol = obtain_global_col_id(clientMap, clientInfos).size
    clientMap[clientInfos[clientIdWLabel]].initialize_beta(nGlbCol)
    for ii in range(0, nClient):
        if clientMap[clientInfos[ii]].wLabel:
            clientMap[clientInfos[ii]].apply_internal_q_2_label()
        else:
            Q = clientMap[clientInfos[ii]].obtain_q_unencrypted()
            glbRowId0 = clientMap[clientInfos[ii]].get_glb_row_id0()
            clientMap[clientInfos[clientIdWLabel]].apply_external_q_2_label_unencrypted(Q, glbRowId0)

# For computing back solve.
def apply_back_solve_wo_constraint(clientMap, clientInfos):
    """
    Apply backsolve to the label and compute the weights.

    Parameters
    ----------
    clientMap : List
        The list of qrClient objects.
    clientInfos : List
        The list of machine information of the corresponding qrClient objects.

    Returns
    -------
    numpy.array
        The computed weights of all the clients. The weights corresponding to the constant term is at the last position.
    """
    nClient = len(clientInfos)
    for ii in range(0, nClient):
        if clientMap[clientInfos[ii]].wLabel:
            clientIdWLabel = ii
            break
    beta = clientMap[clientInfos[clientIdWLabel]].obtain_beta_all()
    for ii in range(nClient-1, -1, -1):
        beta = clientMap[clientInfos[ii]].compute_back_solve(beta)
        beta[:, 0] = clientMap[clientInfos[clientIdWLabel]].refresh_encryption(beta.reshape(-1))
    return obtain_global_weights(clientMap, clientInfos)

# For complete QR algorithm.
def linreg_qr_householder_unencrypted(clientMap, coordinator, encryLv=3, colTrunc=False):
    """
    Compute vertical federated linear regression using QR.
    QR decomposition is computed by means of HouseHolder reflection.

    Parameters
    ----------
    clientMap : List
        The list of qrClient objects.
    clientInfos : List
        The list of machine information of the corresponding qrClient objects.
    encryLv : int
        The least number of columns the feature matrix of a single client should have to protect its privacy.
    colTrunc : bool
        Do the column pivoting and truncation or not.

    Returns
    -------
    numpy.array
        The computed weights of all the clients. The weights corresponding to the constant term is at the last position.
    """
    preprocessing_wo_constaint(clientMap, coordinator.machine_info_client, encryLv, colTrunc)
    compute_qr_householder_unencrypted(clientMap, coordinator.machine_info_client)
    apply_householder_unencrypted(clientMap, coordinator.machine_info_client)
    weights = apply_back_solve_wo_constraint(clientMap, coordinator.machine_info_client)
    return weights

def linreg_qr_gramschmidt_unencrypted(clientMap, coordinator, encryLv=3, colTrunc=False):
    """
    Compute vertical federated linear regression using QR.
    QR decomposition is computed by means of Numpy/Scipy builtin algorithm and Gram-Schmidt method.

    Parameters
    ----------
    clientMap : List
        The list of qrClient objects.
    clientInfos : List
        The list of machine information of the corresponding qrClient objects.
    encryLv : int
        The least number of columns the feature matrix of a single client should have to protect its privacy.
    colTrunc : bool
        Do the column pivoting and truncation or not.

    Returns
    -------
    numpy.array
        The computed weights of all the clients. The weights corresponding to the constant term is at the last position.
    """
    preprocessing_wo_constaint(clientMap, coordinator.machine_info_client, encryLv, colTrunc)
    compute_qr_gramschmidt_unencrypted(clientMap, coordinator.machine_info_client)
    apply_q_unencrypted(clientMap, coordinator.machine_info_client)
    weights = apply_back_solve_wo_constraint(clientMap, coordinator.machine_info_client)
    return weights

def solve_weights(clientMap, coordinator, encryLv=3, qrMthd="HouseHolder", colTrunc=False):
    """
    Compute vertical federated linear regression using QR.

    Parameters
    ----------
    clientMap : List
        The list of qrClient objects.
    coordinator : qrCoordinator
        The coordinator object.
    encryLv : int
        The least number of columns the feature matrix of a single client should have to protect its privacy.
    qrMthd : String
        The method name for QR decomposition.
            "HouseHolder": Do QR decomposition using HouseHolder reflection.
            "GramSchmidt": Do QR decomposition using Numpy/Scipy builtin algorithm and block GramSchmidt algorithm.
    colTrunc : bool
        Do the column pivoting and truncation or not.

    Returns
    -------
    numpy.array
        The computed weights of all the clients. The weights corresponding to the constant term is at the last position.
    """
    if qrMthd=="HouseHolder":
        weights = linreg_qr_householder_unencrypted(clientMap, coordinator, encryLv, colTrunc)
    elif qrMthd=="GramSchmidt":
        weights = linreg_qr_gramschmidt_unencrypted(clientMap, coordinator, encryLv, colTrunc)
    else:
        raise RuntimeError("QR by the specific method is not implemented yet.")
    return weights


# Below are the deprecated functions.
def linear_regression_initial_w_constraint(clients, encryLv):
    nClient = len(clients)
    nAllFeature = 0
    for ii in range(0, nClient):
        nAllFeature = clients[ii].mod_feature_4_qr(encryLv, nAllFeature)

def linear_regression_apply_q_encrypted(clients):
    nClient = len(clients)
    for ii in range(0, nClient):
        if clients[ii].wLabel:
            clientIdWLabel = ii
            labelModified = False
            break
    beta = clients[clientIdWLabel].obtain_beta_all_unencrypted()
    # label = clients[ii].obtain_label()

    for ii in range(0, nClient):
        if clients[ii].wLabel and labelModified:
            clients[ii].fetch_beta_all_unencrypted(beta)
            # clients[ii].fetch_label(label)

        clients[ii].internal_apply_householder()
        for jj in range(ii+1, nClient):
            features = clients[jj].obtain_features_unencrypted()
            features = clients[ii].external_apply_householder(features)
            clients[jj].fetch_features_unencrypted(features)
        if not clients[ii].wLabel:
            beta = clients[ii].external_apply_householder(beta)
            labelModified = True
        else:
            beta = clients[ii].obtain_beta_all_unencrypted()
            # label = clients[ii].obtain_label()
            labelModified = False
    
    clients[clientIdWLabel].fetch_beta_all_unencrypted(beta)
    # clients[clientIdWLabel].fetch_label(label)

def linear_regression_back_solve_w_constraint(clients):
    nClient = len(clients)
    for ii in range(0, nClient):
        if clients[ii].wLabel:
            clientIdWLabel = ii
            break
    
    nAllFeature = obtain_global_col_id(clients).size
    beta = clients[clientIdWLabel].obtain_beta_all(nAllFeature)
    for ii in range(nClient-1, -1, -1):
        if clients[ii].usefulFeature.size==clients[ii].nFeature:
            beta = clients[ii].apply_backsolve_all(beta, clients[clientIdWLabel])
            beta = clients[ii].compute_local_weights(beta)
        else:
            beta = clients[ii].apply_backsolve_w_useless_feature(beta, clients[clientIdWLabel])
    
    return clients[clientIdWLabel].obtain_weights(beta, obtain_global_feature_id(clients))