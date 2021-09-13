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
import copy
import argparse
import pandas
import math
import scipy.linalg
from core.encrypt.he import RandomizedIterativeAffine as RIAC
from core.client.client import Client
from core.entity.common.message import RequestMessage, ResponseMessage
from importlib.machinery import SourceFileLoader
from core.entity.common.machineinfo import MachineInfo
from core.grpc_comm.grpc_server import serve


class LinRegQRClient(Client):
    """
    Linear regression client class based on QR algorithm.

    Parameters
    ----------
    machineInfo : MachineInfo
        The machine info class that save the current client information,
        including ip, port and token.
    parameter : Dict
        The parameter of federated random forest model for this training task.
    dataset : Dict
        The binding dataset of this training task, including 'feature' and 'label' key.
    remote : bool
        Whether or not the clients are deployed on remote machines.

    Attributes
    ----------
    machine_info_coordinator : MachineInfo
        The machine_info_coordinator argument passing in.
    machine_info_client : List
        The machine_info_client argument passing in.
    parameter : Dict
        The parameter argument passing in.
    """
    # We need initialization, encrypt, decrypt, offerData, 
    def __init__(self, machineInfo, parameter, dataset, remote=False):
        self.machine_info = machineInfo
        self.parameter = parameter
        self.dataset = dataset
        self.remote = remote
        self.truncTol = 1e-10
        self.invTol = 1e-14
        self.constrWts = 10.0**6

        # Collect information from the inputs.
        if self.dataset["label"] is None:
            self.wLabel = False
        else:
            self.label = copy.deepcopy(self.dataset["label"]).astype(float)
            self.wLabel = True

        if self.dataset["trainFeatures"] is None or self.dataset["trainFeatures"].shape[1]==0:
            if not self.wLabel:
                raise RuntimeError("Passive clients need to offer feature data.")
            self.nSample = self.label.shape[0]
            self.nFeature = 0
        else:
            self.trainFeatures = copy.deepcopy(self.dataset["trainFeatures"]).astype(float)
            self.nSample, self.nFeature = self.trainFeatures.shape

        if self.dataset["inferFeatures"] is not None and self.dataset["trainFeatures"].shape[1]!=0:
            if self.nFeature==0:
                raise RuntimeError("Clients need to offer training feature data when testing feature data is used in inference.")
            self.inferFeatures = copy.deepcopy(self.dataset["inferFeatures"]).astype(float)
        if self.dataset["inferFeatures"] is not None:
            self.nInference = self.dataset["inferFeatures"].shape[0]
        self.wConst = False
        self.nConstr = 0
        self.encrypt_key = RIAC.generate_keypair()
        self.householders = []

        # Build workflow function handles.
        self.preprocessing_func = {}
        self.dict_functions = {
            "0": self.initialize_client,
            "1": self.linear_regression_qr_client_phase1,
            "2": self.linear_regression_qr_client_phase2,
            "3": self.linear_regression_qr_client_phase3,
            "99": self.linear_regression_qr_client_post_training,
            "-1": self.linear_regression_qr_client_inference_phase1,
            "-2": self.linear_regression_qr_client_inference_phase2
        }
        self.postprocessing_func = {}

    # For initialization.
    def add_const_feature(self):
        """
        Add the constant feature into the feature matrix as the first column.
        """
        if self.nFeature==0:
            self.trainFeatures = np.ones([self.nSample, 1])
        else:
            self.trainFeatures = np.append(np.ones([self.nSample, 1]), self.trainFeatures, axis=1)
            self.nFeature += 1
        self.wConst = True

    def normalize_features(self):
        """
        Normalize the feature matrix so that each entry of it is inside the range [-1, 1].
        """
        if self.nFeature>0:
            self.scales = np.amax(np.abs(self.trainFeatures), axis=0)
            self.trainFeatures /= self.scales[np.newaxis, :]

    def add_obliviousness_to_feature(self, oblDeg, colId0):
        """ 
        Add obliviousness into the features. If the number of columns of the features matrix 
        is less than $oblDeg$, add some random features. Finally, add oblivation to the 
        feature matrix by computing a random linear combination of them. Both of them helps to
        protect the privacy of the feature matrix when Q information is shared without encryption.

        Parameters
        ----------
        oblDeg : int
            The least number of columns the final feature matrix should have.
        colId0 : int
            The global index of the first column of the final feature matrix inside the global 
            feature matrix, which is alligned according to the order of the clients decided by
            the coordinator.

        Returns
        -------
        int
            The global index of the first column of the feature matrix of the next client inside 
            the global feature matrix.
        """
        self.ncol = self.nFeature
        if self.nFeature==0:
            return colId0
        # Add some random features, until the number of columns of the features matrix is no less than $oblDeg$, .
        while self.nFeature>0 and self.ncol<oblDeg:
            self.ncol += 1
            self.nConstr += 1
            self.trainFeatures = np.append(self.trainFeatures, np.random.rand(self.nSample, 1)*2-1, axis=1)
            self.trainFeatures[:, -1] /= np.amax(np.abs(self.trainFeatures[:, -1]))
            self.scales = np.append(self.scales, 1.0)
        # Add oblivation to the feature matrix by computing a random linear combination of them.
        while True:
            self.oblMat = np.random.rand(self.ncol, self.ncol)
            if np.linalg.matrix_rank(self.oblMat)<self.ncol:
                continue
            self.oblMatInv = np.linalg.inv(self.oblMat)
            diff = np.matmul(self.oblMat, self.oblMatInv)-np.eye(self.ncol)
            if np.amax(abs(diff))/np.amax(abs(self.oblMat))<self.invTol:
                break
        self.trainFeatures = np.matmul(self.trainFeatures, self.oblMat)
        self.R = np.empty([0, self.ncol])
        return colId0+self.ncol

    def preprocess_features_4_qr(self, oblDeg, colId0):
        """
        Apply preprocessing to the feature matrix. The preprocessing contains the normalization
        of the features and adding obliviousness to the feature matrix.

        Parameters
        ----------
        oblDeg : int
            The least number of columns the final feature matrix should have.
        colId0 : int
            The global index of the first column of the final feature matrix inside the global 
            feature matrix, which is alligned according to the order of the clients decided by
            the coordinator.
        """
        self.normalize_features()
        nAllCol = self.add_obliviousness_to_feature(oblDeg, colId0)
        self.lclFeatureId = np.arange(0, self.nFeature)
        self.glbColId = np.arange(colId0, nAllCol)  # The truncated column will be marked by -1.
        self.glbRowId = np.arange(colId0, nAllCol)
        self.glbColIdInit = np.arange(colId0, nAllCol)
        return nAllCol
    
    def add_oblivious_constraint_4_qr(self, rowId0, nGlbConstr):
        """
        Add extra rows to the feature matrix which are the constraints for the random features.
        We constrain that the computed weights for random features be zeros.

        Parameter
        ---------
        rowId0 : int
            The global index of the first row of the added constraints in the feature matrix.
        nGlbConstr : integre
            The total number of the added constraints of all the clients.
        """
        if self.nFeature>0:
            self.trainFeatures = np.append(self.trainFeatures, np.zeros([rowId0-self.nSample, self.ncol]), axis=0)
            for ii in range(self.ncol-self.nConstr, self.ncol):
                self.trainFeatures = np.append(self.trainFeatures, self.constrWts*self.oblMat[ii, :].reshape([1, self.ncol]), axis=0)
            rowId0 += self.nConstr
            self.trainFeatures = np.append(self.trainFeatures, np.zeros([self.nSample+nGlbConstr-rowId0, self.ncol]), axis=0)
        self.nrow = self.nSample+nGlbConstr
        if self.wLabel:
            self.label = np.append(self.label, np.zeros([nGlbConstr, 1]), axis=0)
        return rowId0

    # For data transfer.
    def obtain_features_unencrypted(self):
        """
        Obtain the feature matrix without encryption.

        Returns
        -------
        float
            The unencrypted feature matrix.
        """
        return copy.deepcopy(self.trainFeatures)

    def obtain_label_unencrypted(self):
        """
        Obtain the label without encryption.

        Returns
        -------
        float
            The unencrypted label.
        """
        return copy.deepcopy(self.label)

    def obtain_beta_all_unencrypted(self):
        """
        Obtain the buffer vector of the right hand side without encryption during
        the back solve process.

        Returns
        -------
        float
            The unencrypted buffer vector of the right hand side.
        """
        return copy.deepcopy(self.beta)

    def obtain_beta_entry_unencrypted(self, ind):
        """
        Obtain one entry of the buffer vector of the right hand side without encryption during
        the back solve process.

        Parameter
        ---------
        ind : int
            The index of the expected entry in the buffer vector.

        Returns
        -------
        float
            The unencrypted entry of the buffer vector of the right hand side.
        """
        return self.beta[ind, 0]

    def obtain_beta_entries_unencrypted(self, inds):
        """
        Obtain a subvector of the buffer vector of the right hand side without encryption during
        the back solve process.

        Parameter
        ---------
        inds : int
            The indices of the entries of the expected subvector in the buffer vector.

        Returns
        -------
        float
            The unencrypted subvector of the buffer vector of the right hand side.
        """
        betaTrunc = np.empty([inds.size,])
        for ii in range(0, inds.size):
            betaTrunc[ii] = self.beta[inds[ii], 0]
        return betaTrunc

    def obtain_householders_unencrypted(self):
        """
        Obtain the computed HouseHolder reflection vectors without encryption.

        Returns
        -------
        float
            The unencrypted HouseHolder reflection vectors.
        """
        return copy.deepcopy(self.householders)

    def obtain_q_unencrypted(self):
        """
        Obtain the computed Q matrix without encryption.

        Returns
        -------
        float
            The unencrypted Q matrix.
        """
        if self.nFeature>0:
            return copy.deepcopy(self.Q)
        else: return np.empty([self.nrow, 0])

    def obtain_features(self):
        """
        Obtain the feature matrix with encryption.

        Returns
        -------
        IterativeAffineCiphertext
            The encrypted feature matrix.
        """
        if self.nFeature==0:
            raise RuntimeError("Active client has no feature data.")
        encryptedFeatures = np.empty_like(self.trainFeatures, dtype=RIAC.IterativeAffineCiphertext)
        for ii in range(0, self.nrow):
            for jj in range(0, self.ncol):
                encryptedFeatures[ii, jj] = self.encrypt_key.encrypt(self.trainFeatures[ii, jj])
        return encryptedFeatures
    
    def obtain_label(self):
        """
        Obtain the label with encryption.

        Returns
        -------
        IterativeAffineCiphertext
            The encrypted label.
        """
        if self.wLabel:
            encryptedLabel = np.empty_like(self.label, dtype=RIAC.IterativeAffineCiphertext)
            for ii in range(0, self.nrow):
                encryptedLabel[ii, 0] = self.encrypt_key.encrypt(self.label[ii, 0])
            return encryptedLabel
        elif not hasattr(self, 'label'):
            raise RuntimeError("Passive client has no label data yet.")
        else:
            return copy.deepcopy(self.label)

    def obtain_beta_all(self):
        """
        Obtain the buffer vector of the right hand side with encryption during
        the back solve process.

        Returns
        -------
        IterativeAffineCiphertext
            The encrypted buffer vector of the right hand side.
        """
        if self.wLabel:
            encryptedBeta = np.empty_like(self.beta, dtype=RIAC.IterativeAffineCiphertext)
            for ii in range(0, self.beta.shape[0]):
                encryptedBeta[ii, 0] = self.encrypt_key.encrypt(self.beta[ii])
            return encryptedBeta
        else:
            return copy.deepcopy(self.beta)
    
    def obtain_beta_entry(self, ind):
        """
        Obtain one entry of the buffer vector of the right hand side with encryption during
        the back solve process.

        Parameter
        ---------
        ind : int
            The index of the expected entry in the buffer vector.

        Returns
        -------
        IterativeAffineCiphertext
            The encrypted entry of the buffer vector of the right hand side.
        """
        if self.wLabel:
            return self.encrypt_key.encrypt(self.beta[ind, 0])
        else:
            return self.beta[ind, 0]

    def obtain_beta_entries(self, inds):
        """
        Obtain a subvector of the buffer vector of the right hand side with encryption during
        the back solve process.

        Parameter
        ---------
        inds : int
            The indices of the entries of the expected subvector in the buffer vector.

        Returns
        -------
        IterativeAffineCiphertext
            The encrypted subvector of the buffer vector of the right hand side.
        """
        if self.wLabel:
            betaTrunc = np.empty([inds.size, 1], dtype=RIAC.IterativeAffineCiphertext)
            for ii in range(0, inds.size):
                betaTrunc[ii] = self.encrypt_key.encrypt(self.beta[inds[ii], 0])
            return betaTrunc
        else:
            return copy.deepcopy(self.beta[inds, 0].reshape(len(inds), 1))

    def fetch_features_unencrypted(self, features):
        """
        Fetch the modified feature matrix from outside the self object which are not encrypted.

        Parameter
        ---------
        features : float
            The modifed feature matrix from outside the self object which are not encrypted.
        """
        self.trainFeatures = copy.deepcopy(features)

    def fetch_label_unencrypted(self, label):
        """
        Fetch the modified label from outside the self object which are not encrypted.

        Parameter
        ---------
        label : float
            The modifed label from outside the self object which are not encrypted.
        """
        assert(self.wLabel)
        self.label = copy.deepcopy(label)

    def fetch_beta_all_unencrypted(self, beta):
        """
        Fetch the modified buffer vector of the right hand side from outside the self 
        object which are not encrypted during the backsolve process.

        Parameter
        ---------
        beta : float
            The modified buffer vector of the right hand side which are not encrypted.
        """
        assert(self.wLabel)
        self.beta = copy.deepcopy(beta)

    def fetch_beta_entry_unencrypted(self, ind, beta0):
        """
        Fetch a modified entry of buffer vector of the right hand side from outside 
        the self object which are not encrypted during the backsolve process.

        Parameter
        ---------
        ind : int
            The index of the modified entry.
        beta0 : float
            The modified entry of buffer vector of the right hand side which are not encrypted.
        """
        assert(self.wLabel)
        self.beta[ind, 0] = beta0
    
    def fetch_beta_entries_unencrypted(self, inds, beta1):
        """
        Fetch a modified subvector of buffer vector of the right hand side from outside 
        the self object which are not encrypted during the backsolve process.

        Parameter
        ---------
        inds : int
            The indices of the modified subvector.
        beta1 : float
            The modified subvector of buffer vector of the right hand side which are not encrypted.
        """
        assert(self.wLabel)
        for ii in range(0, inds.size):
            self.beta[inds[ii], 0] = beta1[ii]
    
    def fetch_features(self, features):
        """
        Fetch the modified feature matrix from outside the self object which are encrypted.

        Parameter
        ---------
        features : IterativeAffineCiphertext
            The modifed feature matrix from outside the self object which are encrypted.
        """
        for ii in range(0, self.nrow):
            for jj in range(0, self.ncol):
                self.trainFeatures[ii, jj] = self.encrypt_key.decrypt(features[ii, jj])
    
    def fetch_label(self, label):
        """
        Fetch the modified label from outside the self object which are encrypted.

        Parameter
        ---------
        label : IterativeAffineCiphertext
            The modifed label from outside the self object which are encrypted.
        """
        assert(self.wLabel)
        for ii in range(0, self.nrow):
            self.label[ii, 0] = self.encrypt_key.decrypt(label[ii, 0])

    def fetch_beta_all(self, beta):
        """
        Fetch the modified buffer vector of the right hand side from outside the self 
        object which are encrypted during the backsolve process.

        Parameter
        ---------
        beta : IterativeAffineCiphertext
            The modified buffer vector of the right hand side which are encrypted.
        """
        assert(self.wLabel)
        for ii in range(0, self.beta.shape[0]):
            self.beta[ii, 0] = self.encrypt_key.decrypt(beta[ii, 0])
    
    def fetch_beta_entry(self, ind, beta0):
        """
        Fetch a modified entry of buffer vector of the right hand side from outside 
        the self object which are encrypted during the backsolve process.

        Parameter
        ---------
        ind : int
            The index of the modified entry.
        beta0 : IterativeAffineCiphertext
            The modified entry of buffer vector of the right hand side which are encrypted.
        """
        assert(self.wLabel)
        self.beta[ind, 0] = self.encrypt_key.decrypt(beta0)

    def fetch_beta_entries(self, inds, beta1):
        """
        Fetch a modified subvector of buffer vector of the right hand side from outside 
        the self object which are encrypted during the backsolve process.

        Parameter
        ---------
        inds : int
            The indices of the modified subvector.
        beta1 : IterativeAffineCiphertext
            The modified subvector of buffer vector of the right hand side which are encrypted.
        """
        assert(self.wLabel)
        for ii in range(0, inds.size):
            self.beta[inds[ii], 0] = self.encrypt_key.decrypt(beta1[ii, 0])
    
    def get_glb_row_id0(self):
        """
        Obtain the global index for the first row of the upper triangular diagonal block
        inside the local R matrix derived by QR decomposition.

        Returns
        -------
        int
            The global index for the first row of the upper triangular diagonal block inside
            R matrix. If R is an empty matrix return -1.
        """
        if self.glbRowId.size>0: return np.amin(self.glbRowId)
        else: return -1

    # For computing QR decomposition.
    def update_global_row_indices(self, glbRankDefi):
        """
        Given the rank deficency of the previous clients, update the global indices for all 
        the rows of the upper triangular diagonal block inside the local R matrix derived 
        by QR decomposition.

        Parameter
        ---------
        glbRankDefi : int
            The rank deficency of the previous clients.
        """
        self.glbRowId -= glbRankDefi

    def compute_householder_1col(self, lclColId):
        """
        Compute HouseHolder reflection for one column.

        Parameter
        ---------
        lclColId : int
            The local index of the column of the feature matrix which is used to derive the
            HouseHolder reflection.
        """
        if self.glbColId[lclColId]<0: return
        rowId0 = self.glbRowId[lclColId]
        alpha = np.linalg.norm(self.trainFeatures[rowId0:, lclColId])
        if self.colTrunc:
            # Select the column with largest norm.
            if self.ncol==self.nFeature: alphaEst = math.sqrt(self.nSample)
            else: alphaEst = math.sqrt(self.nSample)+self.constrWts
            colIdAlphaMax = lclColId
            for ii in range(lclColId+1, self.ncol):
                alpha0 = np.linalg.norm(self.trainFeatures[rowId0:, ii])
                if abs(alpha0)>abs(alpha):
                    alpha = alpha0
                    colIdAlphaMax = ii
            # Do column truncation.
            if (lclColId==0 and abs(alpha/alphaEst)<self.truncTol) or (lclColId>0 and \
                abs(alpha/self.trainFeatures[np.amin(self.glbRowId), 0])<self.truncTol):
                self.glbColId[lclColId:] = -1
                self.glbRowId = self.glbRowId[0:lclColId]
                return
            # Do column pivoting.
            if lclColId!=colIdAlphaMax:
                self.glbColId[[lclColId, colIdAlphaMax]] = self.glbColId[[colIdAlphaMax, lclColId]]
                self.trainFeatures[:, [lclColId, colIdAlphaMax]] = self.trainFeatures[:, [colIdAlphaMax, lclColId]]
                self.oblMat[:, [lclColId, colIdAlphaMax]] = self.oblMat[:, [colIdAlphaMax, lclColId]]
                # self.oblMatInv[[colId, colIdAlphaMax], :] = self.oblMatInv[[colIdAlphaMax, colId], :]
        # Compute HouseHolder reflection.
        if np.sign(self.trainFeatures[rowId0, lclColId])>0:
            alpha = -alpha
        vec = self.trainFeatures[rowId0:, [lclColId]]
        vec[0, 0] -= alpha
        vec = vec/np.linalg.norm(vec)
        self.householders.append(vec)
        self.trainFeatures[rowId0, lclColId] = alpha
        self.trainFeatures[rowId0+1:, lclColId] = 0
    
    def apply_householder_2_next_cols(self, lclColId):
        """
        Apply the computed HouseHolder reflection for one column onto the following columns.

        Parameter
        ---------
        lclColId : int
            The local index of the column of the feature matrix which is used to derive the
            HouseHolder reflection.
        """
        if self.glbColId[lclColId]<0: return
        rowId0 = self.glbRowId[lclColId]
        self.trainFeatures[rowId0:, lclColId+1:] -= self.householders[lclColId]*np.matmul(2*self.householders[lclColId].T, self.trainFeatures[rowId0:, lclColId+1:])

    def assemble_r_after_householder(self):
        """
        Assemble the R matrix for backsolve.
        """
        if self.nFeature>0:
            self.R = self.R[:, self.glbColId>=0]
            if self.glbRowId.size>0:
                self.R = np.append(self.R, self.trainFeatures[0:np.amin(self.glbRowId), self.glbColId>=0], axis=0)
                self.R = np.append(self.R, self.trainFeatures[self.glbRowId, :][:, self.glbColId>=0], axis=0)

    def compute_qr_w_householder(self):
        """
        Compute the local QR decomposition for the local feature matrix using HouseHolder 
        reflection algorithm.

        Returns
        -------
        float
            The computed HouseHolder reflections.
        """
        for ii in range(0, self.ncol):
            self.compute_householder_1col(ii)
            self.apply_householder_2_next_cols(ii)
        self.assemble_r_after_householder()
        return self.householders

    def apply_external_householder_2_features_unencrypted(self, householders):
        """
        Apply the unencrypted HouseHolder reflections from outside the client to the local features matrix.

        Parameter
        ---------
        householders : float
            The unencrypted HouseHolder reflections from outside the client
        """
        if self.nFeature>0 and len(householders)>0:
            for ii in range(0, len(householders)):
                rowId0 = self.nrow-householders[ii].size
                self.trainFeatures[rowId0:, :] -= householders[ii]*np.matmul(2*householders[ii].T, self.trainFeatures[rowId0:, :])

    def compute_qr_w_builtin(self):
        """
        Compute QR decomposition with builtin QR algorithm. If QR decomposition is computed
        without column pivoting and truncation, we can use the QR function offered by Numpy.
        If QR decomposition is computed with column pivoting and truncation, we use the QR 
        function offered by Scipy.

        Returns
        -------
        float
            The computed Q matrix.
        """
        if self.nFeature>0:
            if not self.colTrunc:
                # Compute the QR decomposition without column pivoting and truncation.
                self.Q, R = np.linalg.qr(self.trainFeatures)
                self.R = np.append(self.R, R, axis=0)
            else:
                # Compute the QR decomposition with column pivoting and truncation.
                if self.ncol==self.nFeature: alphaEst = math.sqrt(self.nSample)
                else: alphaEst = math.sqrt(self.nSample)+self.constrWts
                self.Q, R, P = scipy.linalg.qr(self.trainFeatures, mode='economic', pivoting=True)
                self.glbColId = self.glbColId[P]
                self.oblMat = self.oblMat[:, P]
                self.R = self.R[:, P]
                # invP = np.empty([self.ncol])
                # invP[P] = np.arange(0, self.ncol)
                # self.oblMatInv = self.oblMatInv[invP, :]
                for ii in range(0, self.ncol):
                    if (ii==0 and abs(R[ii, ii]/alphaEst)<self.truncTol) or (ii>0 and abs(R[ii, ii]/R[0, 0])<self.truncTol):
                        self.glbColId[ii:] = -1
                        self.glbRowId = self.glbRowId[0:ii]
                        break
                self.Q = self.Q[:, self.glbColId>=0]
                self.R = self.R[:, self.glbColId>=0]
                nrow = sum(self.glbColId>=0)
                self.R = np.append(self.R, R[0:nrow, :][:, self.glbColId>=0], axis=0)
            return self.Q
        return np.empty([self.nrow, 0])

    def apply_external_q_2_features_by_gs_unencrypted(self, Q):
        """
        Apply external unencrypted Q matrix to local features.

        Parameter
        ---------
        Q : float
            The unencrypted Q matrix from outside the client
        """
        if self.nFeature>0 and Q is not None and Q.shape[1]>0:
            rowId0 = self.R.shape[0]
            self.R = np.append(self.R, np.matmul(Q.T, self.trainFeatures), axis=0)
            rowId1 = self.R.shape[0]
            self.trainFeatures -= np.matmul(Q, self.R[rowId0:rowId1, :])

    def get_local_rank_deficiency(self):
        """
        Obtain the local rank deficiency.
        """
        return self.glbColId.size-self.glbRowId.size

    # For applying Q projection.
    def apply_internal_householder_2_label(self):
        """
        Apply the locally computed HouseHolder reflections to the label.
        """
        assert(self.wLabel)
        for ii in range(0, len(self.householders)):
            rowId0 = self.glbRowId[ii]
            self.label[rowId0:, [0]] -= 2*np.matmul(self.householders[ii].T, self.label[rowId0:, [0]])*self.householders[ii]
    
    def apply_external_householder_2_label_unencrypted(self, householders):
        """
        Apply unencrypted HouseHolder reflections from outside the client to the label.

        Parameter
        ---------
        householders : float
            The unencrypted HouseHolder reflections from outside the client.
        """
        assert(self.wLabel)
        nhh = len(householders)
        for ii in range(0, nhh):
            rowId0 = self.nrow-householders[ii].size
            self.label[rowId0:, [0]] -= 2*np.matmul(householders[ii].T, self.label[rowId0:, [0]])*householders[ii]

    def apply_internal_q_2_label(self):
        """
        Apply locally computed Q matrix to the label.
        """
        assert(self.wLabel)
        if self.nFeature>0:
            self.beta[self.glbRowId, 0] = np.matmul(self.Q.T, self.label).reshape(-1)
    
    def apply_external_q_2_label_unencrypted(self, Q, glbRowId0):
        """
        Apply unencrypted Q matrix from outside the client to the label.

        Parameter
        ---------
        Q : float
            The unencrypted Q matrix from outside the client.
        glbRowId0 : int
            The global index of the first row of the computed product of Q and the 
            right-hand-side buffer.
        """
        assert(self.wLabel)
        if glbRowId0>=0:
            self.beta[glbRowId0:glbRowId0+Q.shape[1], 0] = np.matmul(Q.T, self.label).reshape(-1)

    def initialize_beta(self, nGlbCol):
        """
        Initialize buffer vector of the right hand side for backsolve.

        Parameter
        ---------
        nGlbCol : int
            The total number of columns of the global features matrix after QR decomposition
            with/without truncation.
        """
        assert(self.wLabel)
        self.beta = copy.deepcopy(self.label[0:nGlbCol, 0].reshape([nGlbCol, 1]))

    # For applying backsolve.
    def refresh_encryption(self, encryArray):
        """
        Decrypt and re-encrypt an array.

        Parameter
        ---------
        encryArray : IterativeAffineCiphertext
            The array that need to be re-encrypted.

        Returns
        -------
        IterativeAffineCiphertext
            The re-encrypted array.
        """
        assert(self.wLabel)
        for ii in range(0, encryArray.size):
            val = self.encrypt_key.decrypt(encryArray[ii])
            encryArray[ii] = self.encrypt_key.encrypt(val)
        return encryArray

    def apply_blk_inv(self):
        """
        Solve the system for the diagonal upper-triangular block by means of matrix operation.
        """
        # Using matrix operation on encrypted data increase the computational speed by 3-4 times.
        if self.glbRowId.size>0:
            blkInv = np.linalg.inv(self.R[self.glbRowId, :])
            self.beta[self.glbRowId, 0] = np.matmul(blkInv, self.beta[self.glbRowId, 0][:, np.newaxis]).reshape(-1)

    def apply_back_sub(self):
        """
        Substitute the solution to the right hand side by means of matrix operation.
        """
        # Using matrix operation on encrypted data increase the computational speed by 3-4 times.
        if self.glbRowId.size>0:
            rowId0 = np.amin(self.glbRowId)
            self.beta[0:rowId0, 0] -= np.matmul(self.R[0:rowId0, :], self.beta[self.glbRowId, 0][:, np.newaxis]).reshape(-1)
    
    def apply_obliviousness_clarification(self):
        """
        Clarify the obiliviousness in the solution.
        """
        if self.glbRowId.size==0:
            self.beta[self.glbColIdInit, 0] = 0
        else:
            oblMat = self.oblMat[:, self.glbColId>=0]/self.scales[:, np.newaxis]
            self.beta[self.glbColIdInit, 0] = np.matmul(oblMat, self.beta[self.glbRowId, 0][:, np.newaxis]).reshape(-1)
        self.lclWts = self.beta[self.glbColIdInit[self.lclFeatureId], 0][:, np.newaxis]

    def compute_back_solve(self, beta=None):
        """
        Apply the backsolve. The backsolve consists of solving the sub-system corresponding to
        the diagonal upper triangular block, substitue the solution to the right hand side and 
        clarify the obliviousness in the solution.

        Parameter
        ---------
        beta : IterativeAffineCiphertext
            The encrypted buffer vector for the right hand side for the backsolve process.

        Returnes
        --------
        IterativeAffineCiphertext
            The encrypted subvector of the input buffer vector which cooresponds to the unsolved
            global weights.
        """
        if self.nFeature>0:
            if self.wLabel and beta is not None:
                # In active client, the buffer vector is already intialized.
                self.fetch_beta_entries(np.arange(0, beta.shape[0]), beta)
            elif not self.wLabel:
                # In active client, the buffer vector is not intialized.
                self.beta = copy.deepcopy(beta)
            self.apply_blk_inv()
            self.apply_back_sub()
            self.apply_obliviousness_clarification()
            if beta is not None:
                return self.obtain_beta_entries(np.arange(0, beta.shape[0]-self.glbRowId.size))
            else:
                return self.obtain_beta_entries(np.arange(0, self.beta.shape[0]-self.glbRowId.size))
        elif beta is not None:
            return beta
        else:
            return self.obtain_beta_all()

    # For extracting the final solution.
    def get_local_weights(self):
        """
        Obtain the computed local weights. This is used in the global work flow.

        Returns
        -------
        IterativeAffineCiphertext
            The computed local weights.
        """
        return copy.deepcopy(self.lclWts)

    def get_global_weights(self, glbColId4Features):
        """
        Obtain the computed local weights from the buffer vector. This is used in the testing
        script of the QR algorithm.

        Parameters
        ----------
        glbColId4Features : int
            The global indices of the columns of the features matrix which are not added random
            features.
        
        Returns
        -------
        IterativeAffineCiphertext
            The computed local weights.
        """
        assert(self.wLabel)
        return self.beta[glbColId4Features, 0][:, np.newaxis]

    # For remote communication.
    def make_response(self, tarMachineInfo, body, phaseId):
        """
        Making response object. Given the input request and body dictionary,
        this function returns a response object which is ready for sending out.

        Parameters
        ----------
        tarMachineInfo : MachineInfo
            The machine info class that save the target client/coordinator information,
            including ip, port and token.
        body : Dict
            The response body data in dictionary type.
        phaseId : string
            The phase label.

        Returns
        -------
        ResponseMessage
            The response message which is ready for sending out.
        """
        response = ResponseMessage(self.machine_info, tarMachineInfo, body, phase_id=phaseId)
        if self.remote:
            response.serialize_body()
        return response

    def get_request(self, request):
        """
        Deserialize the request body for remote clients.

        Parameters
        ----------
        request : RequestMessage
            The request message sent into the client.

        Returns
        -------
        request : RequestMessage
            The request message after modification.
        """
        if self.remote:
            request.deserialize_body()
        return request

    def control_flow_client(self, phaseId, request):
        """
        The main control of the work flow inside client. This might be able to work in a generic
        environment.

        Parameters
        ----------
        phaseId : string
            The phase label.
        request : RequestMessage
            The request message sent into the client.

        Returns
        -------
        ResponseMessage
            The response message after the processing of the current phase.
        """
        response = request
        # if phase has preprocessing, then call preprocessing func
        if phaseId in self.preprocessing_func:
            response = self.preprocessing_func[phaseId](response)
        if phaseId in self.dict_functions:
            response = self.dict_functions[phaseId](response)
        # if phase has postprocessing, then call postprocessing func
        if phaseId in self.postprocessing_func:
            response = self.postprocessing_func[phaseId](response)
        return response

    def initialize_client(self, request):
        """
        Client initialization function for federated linear regression based on QR. 
        We do all the preprocessings for the feature matrix, including adding 
        constant feature, normalization and adding obliviousness.

        Parameters
        ----------
        request : RequestMessage
            The request message sent into the client.

        Returns
        -------
        ResponseMessage
            The response message which is ready for sending out.
        """
        request = self.get_request(request)
        self.qrMthd = request.body["qrMthd"]
        self.colTrunc = request.body["colTrunc"]
        self.clientId = request.body["clientId"]
        self.glbColIdInit = np.arange(request.body["glbFeatureId0"], request.body["glbFeatureId0"]+self.nFeature)
        if request.body["addConst"]:
            self.add_const_feature()
        self.preprocess_features_4_qr(request.body["encryLv"], request.body["glbColId0"])
        constrRowId0 = self.nSample+request.body["glbConstrId0"]
        self.add_oblivious_constraint_4_qr(constrRowId0, request.body["nGlbConstr"])
        message = {
            "needDecomp": True,
            "clientId": self.clientId
        }
        print("Finished the preprocessing for the local features.")
        return self.make_response(request.server_info, message, request.phase_id)

    def linear_regression_qr_client_phase1(self, request):
        """
        Client phase 1 code of federated linear regression based on QR.
        This process does multiplication of the computed Q with the local feature
        matrix. If needed, compute QR factorization for the local feature matrix. 

        Parameters
        ----------
        request : RequestMessage
            The request message sending into the client.

        Returns
        -------
        ResponseMessage
            The response message which is ready for sending out.
        """
        request = self.get_request(request)
        # Apply Q to features and label.
        if request.body["Q"] is not None:
            if self.qrMthd=="HouseHolder":
                self.apply_external_householder_2_features_unencrypted(request.body["Q"])
                if self.wLabel:
                    self.apply_external_householder_2_label_unencrypted(request.body["Q"])
            elif self.qrMthd=="GramSchmidt":
                self.apply_external_q_2_features_by_gs_unencrypted(request.body["Q"])
                if self.wLabel:
                    if not hasattr(self, "beta"):
                        self.initialize_beta(request.body["nGlbCol"])
                        print("Finished initializing beta for GramSchmidts")
                    self.apply_external_q_2_label_unencrypted(request.body["Q"], request.body["glbRowId0"])
            else:
                raise RuntimeError("The specific QR method is not implemented yet.")
            print("Finished applying external Q to local features and/or labels.")
        # Compute QR decomposition.
        # If this is active client, this is the last block to compute QR. Apply the derived Q onto label directly.
        if request.body["needDecomp"]:
            glbRankDefi = request.body["rankDefi"]
            if self.qrMthd=="HouseHolder":
                self.update_global_row_indices(glbRankDefi)
                Q = self.compute_qr_w_householder()
                glbRankDefi += self.get_local_rank_deficiency()
                if self.wLabel:
                    self.apply_internal_householder_2_label()
                    self.initialize_beta(request.body["nGlbCol"])
            elif self.qrMthd=="GramSchmidt":
                self.update_global_row_indices(glbRankDefi)
                Q = self.compute_qr_w_builtin()
                glbRankDefi += self.get_local_rank_deficiency()
                if self.wLabel:
                    self.apply_internal_q_2_label()
            else:
                raise RuntimeError("The specific QR method is not implemented yet.")
            message = {
                "Q": Q,
                "rankDefi": glbRankDefi
            }
            print("Finished computing Q and applying the obtained Q to label if necessary.")
        else:
            message = {}
        return self.make_response(request.server_info, message, request.phase_id)

    def linear_regression_qr_client_phase2(self, request):
        """
        Client phase 2 code of federated linear regression based on QR.
        This process solves the subsystem correponding to the upper triangular diagonal block and substitue
        the solution to the buffer vector of the right hand side.

        Parameters
        ----------
        request : RequestMessage
            The request message sending into the client.

        Returns
        -------
        ResponseMessage
            The response message which is ready for sending out.
        """
        request = self.get_request(request)
        if self.wLabel:
            beta = self.compute_back_solve()
        else:
            beta = self.compute_back_solve(request.body["beta"])
        message = {
            "beta": beta
        }
        print("Finished the solving, substitution and clearance of obliviation for the weights corresponding to the diagonal block.")
        return self.make_response(request.server_info, message, request.phase_id)

    def linear_regression_qr_client_phase3(self, request):
        """
        Client phase 3 code of federated linear regression based on QR.
        This process re-encrypts the buffer vector of the right hand side.

        Parameters
        ----------
        request : RequestMessage
            The request message sending into the client.

        Returns
        -------
        ResponseMessage
            The response message which is ready for sending out.
        """
        request = self.get_request(request)
        beta = request.body["beta"]
        beta[:, 0] = self.refresh_encryption(beta.reshape(-1))
        message = {
            "beta": beta
        }
        print("Finished re-encryption for remaining right-hand-side vector.")
        return self.make_response(request.server_info, message, request.phase_id)

    def linear_regression_qr_client_post_training(self, request):
        """
        Client phase post processing code of federated linear regression based on QR.
        This process does nothing. Just fit with the implementation pipeline.

        Parameters
        ----------
        request : RequestMessage
            The request message sending into the client.

        Returns
        -------
        ResponseMessage
            The response message which is ready for sending out.
        """
        print("Finished post-processing.")
        return self.make_response(request.server_info, {}, request.phase_id)

    def linear_regression_qr_client_inference_phase1(self, request):
        """
        Inference phase 1 of federated linear regression based on QR.
        This process does the multiplication of the computed local weights and the local inference features.

        Parameters
        ----------
        request : RequestMessage
            The request message sending into the client.

        Returns
        -------
        ResponseMessage
            The response message which is ready for sending out.
        """
        request = self.get_request(request)
        if self.wLabel:
            if self.nFeature>0:
                if not self.wConst:
                    self.inference = np.matmul(self.inferFeatures, self.lclWts)
                else:
                    self.inference = np.matmul(self.inferFeatures, self.lclWts[1:, 0][:, np.newaxis])+self.lclWts[0, 0]
            else:
                self.inference = np.zeros([self.nInference, 1])
            message = {}
        else:
            if not self.wConst:
                Yi = np.matmul(self.inferFeatures, self.lclWts)
            else:
                Yi = np.matmul(self.inferFeatures, self.lclWts[1:, 0][:, np.newaxis])+self.lclWts[0, 0]
            message = {
                "Yi": Yi
            }
        print("Finished computing the sub-values.")
        return self.make_response(request.server_info, message, request.phase_id)

    def linear_regression_qr_client_inference_phase2(self, request):
        """
        Inference phase 2 of federated linear regression based on QR.
        This process sums up the products of all the computed local weights and the local inference features.

        Parameters
        ----------
        request : RequestMessage
            The request message sending into the client.

        Returns
        -------
        ResponseMessage
            The response message which is ready for sending out.
        """
        request = self.get_request(request)
        assert(self.wLabel)
        Yi = request.body["Yi"]
        for ii in range(0, self.nInference):
            self.inference[ii, 0] += self.encrypt_key.decrypt(Yi[ii, 0])
        message = {
            "Y": self.inference
        }
        print("Finished adding up all the sub-values.")
        return self.make_response(request.server_info, message, request.phase_id)








    # The following are the depreciated attribute functions.
    def external_apply_householder(self, features):
        nExtSample, nExtFeature = features.shape
        for ii in range(0, self.ncol):
            rowId0 = self.glbColId[ii]
            for jj in range(0, nExtFeature):
                val = 0
                for kk in range(rowId0, nExtSample):
                    val += self.householders[ii][kk-rowId0]*features[kk, jj]
                val *= -2
                for kk in range(rowId0, nExtSample):
                    tempres = self.householders[ii][kk-rowId0]*val
                    features[kk, jj] += tempres[0]
        return features

    def internal_compute_apply_qr_decomp(self):
        # scipy has scipy.linalg.qr which supports column pivoting.
        if self.ncol==0:
            return
        rowId0 = np.amin(self.glbColId)
        self.Q, self.R = np.linalg.qr(self.trainFeatures[rowId0:, :], mode='reduced')
        self.trainFeatures[rowId0:rowId0+self.ncol, :] = self.R
        if self.wLabel:
            self.beta[rowId0:, [0]] = np.matmul(self.Q.T, self.beta[rowId0:, [0]])

    def apply_backsolve_1col(self, colId, beta, client0=None):
        rowId0 = self.glbColId[colId]
        beta[rowId0, 0] = (1/self.trainFeatures[rowId0, colId])*beta[rowId0, 0]
        client0.fetch_beta_entry(rowId0, beta[rowId0, 0])  # Data transfer.
        beta0 = client0.obtain_beta_entry(rowId0)  # Data transfer.
        for ii in range(0, rowId0):
            beta[ii, 0] += -self.trainFeatures[ii, colId]*beta0
        return beta

    def apply_backsolve_all(self, beta, client0=None):
        for ii in range(self.ncol-1, -1, -1):
            beta = self.apply_backsolve_1col(ii, beta, client0)
        return beta
    
    def compute_local_weights(self, beta):
        buf = np.empty([self.ncol, 1], dtype=RIAC.IterativeAffineCiphertext)
        for ii in range(0, self.ncol):
            for jj in range(0, self.ncol):
                if jj==0:
                    buf[ii, [0]] = (self.oblMat[ii, jj])*beta[self.glbColId[jj], [0]]
                else:
                    buf[ii, [0]] += (self.oblMat[ii, jj])*beta[self.glbColId[jj], [0]]
        for ii in range(0, self.ncol):
            beta[self.glbColId[ii], 0] = buf[ii, 0]
        return beta
    
    def apply_backsolve_w_useless_feature(self, beta, client0):
        rowId0 = np.amin(self.glbColId)
        rowId1 = np.amax(self.glbColId)
        self.trainFeatures[0:rowId1+1, self.lclFeatureId] = np.matmul(self.trainFeatures[0:rowId1+1, :], self.oblMatInv[:, self.lclFeatureId])
        print("test feature is ")
        print(self.trainFeatures[0:rowId1+1, self.lclFeatureId])
        bufSubMatInv = np.linalg.inv(self.trainFeatures[self.glbColId[self.lclFeatureId], :][:, self.lclFeatureId])
        bufVec = np.empty([self.lclFeatureId.size, 1], dtype=RIAC.IterativeAffineCiphertext)
        for ii in range(0, self.lclFeatureId.size):
            for jj in range(0, self.lclFeatureId.size):
                if jj==0:
                    bufVec[ii, 0] = bufSubMatInv[ii, jj]*beta[self.glbColId[self.lclFeatureId[jj]], 0]
                else:
                    bufVec[ii, 0] += bufSubMatInv[ii, jj]*beta[self.glbColId[self.lclFeatureId[jj]], 0]
        client0.fetch_beta_entries(self.glbColId[self.lclFeatureId], bufVec)  # Data transfer.
        print("Computed beta=")
        print(client0.beta[self.glbColId[self.lclFeatureId], 0])
        beta1 = client0.obtain_beta_entries(self.glbColId[self.lclFeatureId])  # Data transfer.
        # print("Obtained beta = ", client0.encrypt_key.decrypt(beta1[0, 0]), ", ", client0.encrypt_key.decrypt(beta1[1, 0]))
        print("Obtained beta = ", client0.encrypt_key.decrypt(beta1[0, 0]))
        for ii in range(0, self.lclFeatureId.size):
            beta[self.glbColId[self.lclFeatureId[ii]], 0] = beta1[ii, 0]
        print("Stored beta = ", client0.encrypt_key.decrypt(beta[0, 0]), ", ", client0.encrypt_key.decrypt(beta[1, 0]))
        for ii in range(0, rowId0):
            for jj in range(0, self.lclFeatureId.size):
                beta[ii, 0] -= -self.trainFeatures[ii, self.lclFeatureId]*beta1[jj, 0]
        return beta


if __name__=="__main__":
    # Setting up a single client
    parser = argparse.ArgumentParser()
    parser.add_argument('-I', '--index', type=int, required=True, help='index of client')
    parser.add_argument('-C', '--python_config_file', type=str, required=True, help='python config file')

    args = parser.parse_args()
    idx = args.index
    config = SourceFileLoader("config", args.python_config_file).load_module()

    g = pandas.read_csv(config.client_train_file_path[idx])
    if idx == config.active_index:
        label = g.pop(config.active_label).values.ravel().astype(float)
        label = label.reshape([label.size, 1])
        dataset = {"label": label,
                   "trainFeatures": g.loc[:, g.columns[1:]].values}
    else:
        dataset = {"label": None,
                   "trainFeatures": g.loc[:, g.columns[1:]].values}
    if "client_inference_file_path" in config.__dict__:
        g = pandas.read_csv(config.client_inference_file_path[idx])
        dataset["inferFeatures"] = g.loc[:, g.columns[1:]].values
    ip, port = config.client_ip_and_port[idx].split(":")
    client_info = MachineInfo(ip=ip, port=port,
                              token=config.client_ip_and_port[idx])

    parameter = config.parameter
    client = LinRegQRClient(client_info, parameter, dataset, remote=True)

    serve(client)

