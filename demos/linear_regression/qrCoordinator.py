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
import socket
from core.server.server import Server
from core.entity.common.message import RequestMessage, ResponseMessage
from importlib.machinery import SourceFileLoader
from core.entity.common.machineinfo import MachineInfo


class LinRegQRCoordinator(Server):
    """
    Linear regression coordinator class based on QR decomposition.

    Note
    ----

    Parameters
    ----------
    machine_info_coordinator : MachineInfo
        The machine info class that saving the current coordinator information,
        including ip, port and token.
    machine_info_client : List
        The list of that saving the machine info of clients, including ip,
        port and token.
    parameter : Dict
        The parameter of federated random forest model for this training task.
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
    def __init__(self, coordinatorInfo, clientInfos, parameter, remote=False):
        super().__init__()  # TODO: Check whether this is necessary. I don't think so.
        self.machine_info_coordinator = coordinatorInfo
        self.machine_info_client = clientInfos
        self.parameter = parameter
        self.remote = remote
        
        # Collect parameters.
        self.colTrunc = self.parameter["colTrunc"]
        self.qrMthd = self.parameter["qrMthd"]
        self.nFeatures = self.parameter["nFeatures"]
        self.nClient = len(self.nFeatures)
        self.clientIdWLabel = self.parameter["clientIdWLabel"]
        self.glbFeatureId = np.arange(0, sum(self.nFeatures))
        self.glbFeaturePartition = np.zeros(self.nClient+1, dtype=int)
        for ii in range(0, self.nClient):
            self.glbFeaturePartition[ii+1] = self.glbFeaturePartition[ii]+self.nFeatures[ii]
        self.encryLv = self.parameter["encryLv"]
        self.rankDefi = self.parameter["rankDef"]
        self.trainFinished = False
        self.inferFinished = False

        # set function mapping
        self.dict_functions = {
            "1": self.linear_regression_qr_coordinator_phase1,
            "2": self.linear_regression_qr_coordinator_phase2,
            "3": self.linear_regression_qr_coordinator_phase3,
            "4": self.linear_regression_qr_coordinator_phase4,
            "-1": self.linear_regression_qr_coordinator_inference_phase1,
            "-2": self.linear_regression_qr_coordinator_inference_phase2,
            "-3": self.linear_regression_qr_coordinator_inference_phase3
        }
    
    def make_request(self, tarMachineInfo, body, phaseId):
        """
        Making request object. Given the input response and body dictionary,
        this function returns a request object which is ready for sending out.

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
        RequestMessage
            The request message which is ready for sending out.
        """
        request = RequestMessage(self.machine_info_coordinator, tarMachineInfo,
                              {str(key): value for key, value in body.items()},
                              phase_id=phaseId)
        if self.remote:
            request.serialize_body()
        return request

    def get_response(self, response):
        """
        Deserialize the response body for remote clients.

        Parameters
        ----------
        response : ResponseMessage
            The response message sent into the client.

        Returns
        -------
        ResponseMessage
            The response message after modification.
        """
        if self.remote:
            response.deserialize_body()
        return response

    def reorder_clients(self):
        """
        Reorder the clients so that the last client is active.
        """
        if self.clientIdWLabel!=self.nClient-1:
            clientInfo = self.machine_info_client.pop(self.clientIdWLabel)
            self.machine_info_client.append(clientInfo)
            nFeature = self.nFeatures[self.clientIdWLabel]
            self.nFeatures[self.clientIdWLabel:-1] = self.nFeatures[self.clientIdWLabel+1:]
            self.nFeatures[-1] = nFeature
            self.clientIdWLabel = self.nClient-1
    
    def get_next_phase(self, currPhase):
        """
        Given current phase, return next phase
        For training, the logic is:
            0 => 1 => ... => 1 => 2 => 3 => ... => 2 => 3 => end
        For inference, the logic is :
            -1 => -2 => end

        Parameter
        ---------
        currPhase : string
            The current phase label.

        Returns
        -------
        string
            The next phase label.

        """
        if int(currPhase) >= 0:
            if currPhase == "0":
                nextPhase = "1"
            elif currPhase=="1" and self.clientIdDecomped<self.nClient-1:
                nextPhase = "1"
            elif currPhase=="1":
                nextPhase = "2"
            elif currPhase=="2" and self.clientIdBackslvd==self.nClient-1:
                nextPhase = "2"
            elif currPhase=="2" and self.clientIdBackslvd>0:
                nextPhase = "3"
            elif currPhase=="2":
                nextPhase = "4"
                self.trainFinished = True
            elif currPhase=="3":
                nextPhase = "2"
            else:
                raise ValueError("Invalid phase number")
        else:
            if currPhase == "-1":
                nextPhase = "-2"
            elif currPhase == "-2":
                nextPhase = "-3"
                self.inferFinished = True
            else:
                raise ValueError("Invalid phase number")
        return nextPhase
    
    def control_flow_coordinator(self, phaseId, responses):
        """
        The main control of the work flow of coordinator. This might be able to work in a generic
        environment.

        Parameters
        ----------
        phaseId : string
            The phase label.
        response : ResponseMessage
            The response message sent into the client.

        Returns
        -------
        RequestMessage
            The request message after the processing of the current phase.
        """
        for _, resi in responses.items():
            resi.phase_id = phaseId
        if phaseId in self.dict_functions:
            requests = self.dict_functions[phaseId](responses)
        return requests

    def init_training_control(self):
        """
        Init training for federated linear regression based on QR.
        This process does the reordering of the clients and add parameter into request for initializing the clients.

        Returns
        -------
        RequestMessage
            The request message after the processing of the current phase.
        """
        self.reorder_clients()
        self.clientIdDecomped = -1
        self.clientIdBackslvd = self.nClient
        requests = {}
        self.nCols = copy.deepcopy(self.nFeatures)
        clientWMostCol = np.argmax(self.nCols)
        self.nCols[clientWMostCol] += 1
        nGlbConstr = 0
        for ii in range(0, self.nClient):
            if ii!=self.clientIdWLabel or self.nCols[ii]!=0:
                self.nCols[ii] = max(self.nCols[ii], self.encryLv)
                if ii!=clientWMostCol:
                    nGlbConstr += self.nCols[ii]-self.nFeatures[ii]
                else:
                    nGlbConstr += self.nCols[ii]-self.nFeatures[ii]-1
        glbConstrId0 = 0
        featureId0 = 0
        colId0 = 0
        for ii in range(0, self.nClient):
            message = {
                "qrMthd": self.qrMthd,
                "colTrunc": self.colTrunc,
                "encryLv": self.encryLv,
                "clientId": ii,
                "glbFeatureId0": featureId0,
                "glbColId0": colId0,
                "glbConstrId0": glbConstrId0,
                "nGlbConstr": nGlbConstr,
                "addConst": ii==clientWMostCol
            }
            requests[self.machine_info_client[ii]] = self.make_request(self.machine_info_client[ii], message, "0")
            featureId0 += self.nFeatures[ii]
            colId0 += self.nCols[ii]
            if ii!=clientWMostCol:
                glbConstrId0 += self.nCols[ii]-self.nFeatures[ii]
            else:
                glbConstrId0 += self.nCols[ii]-self.nFeatures[ii]-1
        return requests

    def is_training_continue(self):
        """
        Check if training is finished. Inherit from super class.
        """
        return not self.trainFinished

    def linear_regression_qr_coordinator_phase1(self, responses):
        """
        Coordinator phase 1 code of federated linear regression based on QR.
        This process decide this client need to compute the QR decomposition in the current phase. It adds 
        parameter into request for initializing the clients.

        Parameters
        ----------
        responses : List
            The list of ResponseMessage sending back from the clients.

        Returns
        ------- 
        List
            The list of RequestMessage sending to the clients.
        """
        if self.clientIdDecomped<0:
            message = {
                "Q": None,
                "rankDefi": self.rankDefi
            }
            phaseId = "1"
        else:
            res0 = self.get_response(responses[self.machine_info_client[self.clientIdDecomped]])
            self.rankDefi = res0.body["rankDefi"]
            message = {
                "Q": res0.body["Q"],
                "glbRowId0": sum(self.nCols[0:self.clientIdDecomped]),
                "nGlbCol": sum(self.nCols),
                "rankDefi": self.rankDefi
            }
            phaseId = "1"
        self.clientIdDecomped += 1
        requests = {}
        for ii in range(self.clientIdDecomped, self.nClient):
            if ii==self.clientIdDecomped:
                message["needDecomp"] = True
            else:
                message["needDecomp"] = False
            requests[self.machine_info_client[ii]] = self.make_request(self.machine_info_client[ii], message, phaseId)
        return requests

    def linear_regression_qr_coordinator_phase2(self, responses):
        """
        Coordinator phase 2 code of federated linear regression based on QR.
        This process decide this client need to solve the subsystem corresponding to the diagonal block in the 
        current phase. It adds parameter into request for initializing the clients.

        Parameters
        ----------
        responses : List
            The list of ResponseMessage sending back from the clients.

        Returns
        ------- 
        List
            The list of RequestMessage sending to the clients.
        """
        self.clientIdBackslvd -= 1
        res0 = self.get_response(responses[self.machine_info_client[self.clientIdWLabel]])
        if self.clientIdBackslvd==self.nClient-1:
            message = {}
        else:
            message = {
                "beta": res0.body["beta"]
            }
        requests = {}
        phaseId = "2"
        requests[self.machine_info_client[self.clientIdBackslvd]] = self.make_request(self.machine_info_client[self.clientIdBackslvd], message, phaseId)
        return requests

    def linear_regression_qr_coordinator_phase3(self, responses):
        """
        Coordinator phase 3 code of federated linear regression based on QR.
        This process ask active client to re-encrypt the buffer vector of the right hand side in the 
        current phase. It adds parameter into request for initializing the clients.

        Parameters
        ----------
        responses : List
            The list of ResponseMessage sending back from the clients.

        Returns
        ------- 
        List
            The list of RequestMessage sending to the clients.
        """
        res0 = self.get_response(responses[self.machine_info_client[self.clientIdBackslvd]])
        message = {
            "beta": res0.body["beta"]
        }
        requests = {}
        phaseId = "3"
        requests[self.machine_info_client[self.clientIdWLabel]] = self.make_request(self.machine_info_client[self.clientIdWLabel], message, phaseId)
        return requests

    def linear_regression_qr_coordinator_phase4(self, responses):
        """
        Coordinator phase 4 code of federated linear regression based on QR.
        This process does nothing. Just fit with the algorithm pipeline.

        Parameters
        ----------
        responses : List
            The list of ResponseMessage sending back from the clients.

        Returns
        ------- 
        List
            The list of RequestMessage sending to the clients.
        """
        return {}
        
    def post_training_session(self):
        """
        Coordinator phase post-processing code of federated linear regression based on QR.
        This process does nothing. Just fit with the algorithm pipeline.

        Parameters
        ----------
        responses : List
            The list of ResponseMessage sending back from the clients.

        Returns
        ------- 
        List
            The list of RequestMessage sending to the clients.
        """
        requests = {clienti: self.make_request(clienti, {}, "99") for clienti in self.machine_info_client}
        return requests

    def init_inference_control(self):
        """
        Init inference for federated linear regression based on QR.
        This process does nothing. Just fit with the algorithm pipeline.

        Returns
        -------
        RequestMessage
            The request message after the processing of the current phase.
        """
        requests = {clienti: self.make_request(clienti, {}, "-1") for clienti in self.machine_info_client}
        return requests

    def is_inference_continue(self):
        """
        Check if inference is finished. Inherit from super class.
        """
        return not self.inferFinished

    def linear_regression_qr_coordinator_inference_phase1(self, responses):
        """
        Coordinator inference phase 1 code of federated linear regression based on QR.
        This process does nothing but let the clients apply the computed weights onto the
        inference local features.

        Parameters
        ----------
        responses : List
            The list of ResponseMessage sending back from the clients.

        Returns
        ------- 
        List
            The list of RequestMessage sending to the clients.
        """
        return {}

    def linear_regression_qr_coordinator_inference_phase2(self, responses):
        """
        Coordinator inference phase 2 code of federated linear regression based on QR.
        This process collects the computed products and deliver them to acitve client to do
        summation and decryption.

        Parameters
        ----------
        responses : List
            The list of ResponseMessage sending back from the clients.

        Returns
        ------- 
        List
            The list of RequestMessage sending to the clients.
        """
        if self.nClient==1:
            Yi = None
        else:
            for ii in range(0, self.nClient-1):
                res0 = self.get_response(responses[self.machine_info_client[ii]])
                phaseId = "-2"
                if ii==0:
                    Yi = res0.body["Yi"]
                else:
                    Yi += res0.body["Yi"]
        message = {
            "Yi": Yi
        }
        requests = {}
        requests[self.machine_info_client[self.clientIdWLabel]] = self.make_request(self.machine_info_client[self.clientIdWLabel], message, phaseId)
        return requests

    def linear_regression_qr_coordinator_inference_phase3(self, responses):
        """
        Coordinator inference phase 3 code of federated linear regression based on QR.
        This process collects the computed inference values.

        Parameters
        ----------
        responses : List
            The list of ResponseMessage sending back from the clients.

        Returns
        ------- 
        List
            The list of RequestMessage sending to the clients.
        """
        res0 = self.get_response(responses[self.machine_info_client[self.clientIdWLabel]])
        self.inference = res0.body["Y"]
        return {}

    def post_inference_session(self):
        """
        Coordinator inference phase post-processing code of federated linear regression based on QR.
        This process does nothing. Just fit with the algorithm pipeline.

        Parameters
        ----------
        responses : List
            The list of ResponseMessage sending back from the clients.

        Returns
        ------- 
        List
            The list of RequestMessage sending to the clients.
        """
        return self.inference


if __name__ == "__main__":
    # Setting up coordinator.
    parser = argparse.ArgumentParser()
    parser.add_argument('-C', '--python_config_file', type=str, required=True, help='python config file')

    args = parser.parse_args()
    config = SourceFileLoader("config", args.python_config_file).load_module()

    ip, port = config.coordinator_ip_and_port.split(":")
    coordinator_info = MachineInfo(ip=ip, port=port, token=config.coordinator_ip_and_port)
    client_infos = []
    for ci in config.client_ip_and_port:
        ip, port = ci.split(":")
        client_infos.append(MachineInfo(ip=ip, port=port, token=ci))
    coordinator = LinRegQRCoordinator(coordinator_info, client_infos, config.parameter, remote=True)
    
    # Training.
    coordinator.training_pipeline("0", client_infos)

    # Inference.
    coordinator.inference_pipeline("-1", client_infos)