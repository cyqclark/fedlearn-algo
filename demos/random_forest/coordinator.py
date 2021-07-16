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

"""Example code of federated random forest coordinator.


Example
-------
TBA::

    $ TBA


Notes
-----
    This is an example of federated random forest coordinator. It assumes the
default folder is './opensource'.


"""
import argparse
import os
import pickle
import sys

from importlib.machinery import SourceFileLoader

import numpy
# assume cwd is ./opensource/
sys.path.append(os.getcwd())
from core.entity.common.machineinfo import MachineInfo
from core.entity.common.message import RequestMessage, ResponseMessage
from core.server.server import Server
import core.server.server

class RandomForestCoordinator(Server):
    """
    Random forest coordinator class.

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

    Attributes
    ----------
    machine_info_coordinator : MachineInfo
        The machine_info_coordinator argument passing in.
    machine_info_client : List
        The machine_info_client argument passing in.
    parameter : Dict
        The parameter argument passing in.
    """
    def __init__(self,
                 machine_info_coordinator,
                 machine_info_client,
                 parameter,
                 remote=False):
        super().__init__()
        # pass arguments
        self.parameter = parameter
        self.machine_info_coordinator = machine_info_coordinator
        self.machine_info_client = machine_info_client
        # set function mapping
        self.dict_functions = {"1": self.random_forest_coordinator_phase1,
                           "2": self.random_forest_coordinator_phase2,
                           "3": self.random_forest_coordinator_phase3,
                           "4": self.random_forest_coordinator_phase4,
                           "5": self.random_forest_coordinator_phase5,
                           "-1": self.random_forest_coordinator_inference_phase1,
                           "-2": self.random_forest_coordinator_inference_phase2,
                           "-3": self.random_forest_coordinator_inference_phase3,
                           }
        # no preprocessing or postprocessing in this demo training code
        self.preprocessing_func = {}
        self.postprocessing_func = {}

        # random forest
        self.forest = {}
        self.current_nodes = {}
        self.train_finish = False
        self.inference_finish = False

        # check if use remote mode 
        self.remote = remote
        return None
    
    def is_training_continue(self):
        """
        Check if training is finished. Inherit from super class.
        """
        return not self.train_finish

    def is_inference_continue(self):
        """
        Check if inference is finished. Inherit from super class.
        """
        return not self.inference_finish
    
    def control_flow_coordinator(self,
                            phase_num,
                            responses):
        """
        The main control flow of coordinator. This might be able to work in a generic
        environment.
        """
        # update phase id
        for _, resi in responses.items():
            resi.phase_id = phase_num
        # if phase has preprocessing, then call preprocessing func
        if phase_num in self.dict_functions:
            requests = self.dict_functions[phase_num](responses)
        return requests

    def get_next_phase(self, old_phase):
        """
        Given old phase, return next phase
        For training, the logic is:
            0 => 1 => 2 => 3 => 4 => 5 => 2 => 3 => 4 => 5 => 2 ...
        For inference, the logic is :
            -1 => -2 => -3
        """
        if int(old_phase) >= 0:
            if old_phase == "0":
                next_phase = "1"
            elif (not self.current_nodes) or (old_phase == "1") or (old_phase == "5"):
                next_phase = "2"
            elif old_phase == "2":
                next_phase = "3"
            elif old_phase == "3":
                next_phase = "4"
            elif old_phase == "4":
                next_phase = "5"
            else:
                raise ValueError("Invalid phase number")
        else:
            if old_phase == "-1":
                next_phase = "-2"
            elif old_phase == "-2":
                next_phase = "-3"
                self.inference_finish = True
            else:
                raise ValueError("Invalid phase number")
        return next_phase

    def create_init_requests(self):
        """
        Create initial requests.
        """
        requests = {clienti: RequestMessage(self.machine_info_coordinator, clienti, {}, "0")
                    for clienti in self.machine_info_client}
        return requests

    def check_ser_deser(self, message):
        if self.remote:
            if isinstance(message, ResponseMessage):
                message.deserialize_body()
            elif isinstance(message, RequestMessage):
                message.serialize_body()
        return None
    
    def make_request(self, response, body, phase_id):
        """
        Making request function. Given the input response and body dictionary,
        this function returns a request object which is ready for sending out.

        Parameters
        ----------
        response : ResponseMessage
            The response message sending into the client.
        body : Dict
            The request body data in dictionary type.

        Returns
        ------- 
        RequestMessage
            The request message which is ready for sending out.
        """
        # parse body to bytes
        # convert key to string type
        request = RequestMessage(self.machine_info_coordinator,
                              response.client_info,
                              {str(key): value for key, value in body.items()},
                              phase_id=phase_id)
        self.check_ser_deser(request)
        return request

    def make_null_requests(self, responses, phase_id):
        """
        Making request with empty body.

        Parameters
        ----------
        responses : ResponseMessage
            The response message sending back to the coordinator.
        body : Dict
            The request body data in dictionary type.

        Returns
        ------- 
        RequestMessage
            The request message which is ready for sending out.
        """
        requests = {}
        for client_info, resi in responses.items():
            requests[client_info] = RequestMessage(self.machine_info_coordinator,
                                                   resi.client_info,
                                                   {},
                                                   phase_id=phase_id)
        return requests

    def make_leaf(self, treeid, nodeid):
        """
        Make a node as leaf node given treeid and nodeid.

        Parameters
        ----------
        treeid : int
            Tree id of the node.
        nodeid : int
            Node id of the node.
        """
        print("make leaf: %i, %i"%(treeid, nodeid))
        self.forest[treeid][nodeid]["is_leaf"] = True
        self.forest[treeid][nodeid]["processed"] = True
        self.current_nodes.pop(treeid, None)
        return None

    def release_current_node(self):
        """
        Release all the active nodes in this round of expansion.
        """
        for treeid, nodeid in self.current_nodes.items():
            self.forest[treeid][nodeid] = {"processed": True}
        self.current_nodes = {}
        return None

    def random_forest_coordinator_phase1(self, responses):
        """
        Coordinator phase 1 code of federated random forest.

        Parameters
        ----------
        responses : List
            The list of ResponseMessage sending back from the clients.

        Returns
        ------- 
        List
            The list of RequestMessage sending to the clients.
        """
        # collect feature dimensions
        features = []
        # here we did not check if samples are aligned
        tmp_response = responses[list(responses.keys())[0]].copy()
        self.check_ser_deser(tmp_response)
        num_samples = numpy.arange(tmp_response.body["num_sample"]).astype(int)
        encrypted_label = None
        for clienti, resi in responses.items():
            self.check_ser_deser(resi)
            featurei = resi.body["feature"]
            features += [str(clienti) + "_" + str(k) for k in range(featurei)]
            if "encrypted_label" in resi.body:
                encrypted_label = resi.body["encrypted_label"]
        # random select features and samples
        trees_info = {}
        for i in range(self.parameter["numTrees"]):
            feature_selected = numpy.random.choice(features, self.parameter["maxSampledFeatures"])
            single_tree_info = {str(clienti):[] for clienti in responses.keys()}
            for fi in feature_selected:
                client_info, feature_id = fi.split("_")
                single_tree_info[client_info].append(int(feature_id))
            sample_id = numpy.random.choice(num_samples, self.parameter["maxTreeSamples"])
            single_tree_info["sample_id"] = sample_id.tolist()
            trees_info[i] = single_tree_info
        # convert selected feature into message
        requests = {}
        for client_info, resi in responses.items():
            body = {}
            body["feature_selected"] = {k: v[str(client_info)] for k, v in trees_info.items()}
            body["sample_id"] = {k: v["sample_id"] for k, v in trees_info.items()}
            body["encrypted_label"] = encrypted_label
            requests[client_info] = self.make_request(resi, body, "1")
        
        for treeid in range(self.parameter["numTrees"]):
            root = {"processed": False}
            self.forest[treeid] = {0: root}
        return requests
    
    def random_forest_coordinator_phase2(self, responses):
        """
        Coordinator phase 2 code of federated random forest.

        Parameters
        ----------
        responses : List
            The list of ResponseMessage sending back from the clients.

        Returns
        ------- 
        List
            The list of RequestMessage sending to the clients.
        """
        # BFS get current node
        for treeid in self.forest.keys():
            for nodeid in self.forest[treeid].keys():
                if not self.forest[treeid][nodeid]["processed"]:
                    self.current_nodes[treeid] = nodeid
                    break
        if self.current_nodes:
            requests = {}
            for client_info, resi in responses.items():
                #requests[client_info] = self.make_request(resi, self.current_nodes)
                requests[client_info] = self.make_request(resi, self.current_nodes, "2")
            return requests
        else:
            self.train_finish = True
            return self.make_null_requests(responses, "2")

    def random_forest_coordinator_phase3(self, responses):
        """
        Coordinator phase 3 code of federated random forest.

        Parameters
        ----------
        responses : List
            The list of ResponseMessage sending back from the clients.

        Returns
        ------- 
        List
            The list of RequestMessage sending to the clients.
        """
        requests = {}
        Ys = {}
        released_treeid = set()
        print(self.current_nodes)
        for client_info, resi in responses.items():
            self.check_ser_deser(resi)
            Y1_info = resi.body["res"]
            for treeid, Y1 in Y1_info.items():
                if "skip" in Y1:
                    if not treeid in released_treeid:
                        node_id = self.current_nodes[treeid]
                        self.make_leaf(treeid, node_id)
                        released_treeid.add(treeid)
                elif treeid in Ys:
                    Ys[treeid][str(client_info)] = Y1
                else:
                    tmp = {str(client_info): Y1}
                    Ys[treeid] = tmp
        if not Ys:
            requests = self.make_null_requests(responses, "3")
        else:
            for client_info, resi in responses.items():
                if resi.body["is_active"]:
                    requests[client_info] = self.make_request(resi, Ys, "3")
                else:
                    requests[client_info] = self.make_request(resi, {}, "3")

        return requests

    def random_forest_coordinator_phase4(self, responses):
        """
        Coordinator phase 4 code of federated random forest.

        Parameters
        ----------
        responses : List
            The list of ResponseMessage sending back from the clients.

        Returns
        ------- 
        List
            The list of RequestMessage sending to the clients.
        """
        requests = {}
        split_info = {}
        for client in self.machine_info_client:
            split_info[str(client)] = {}
        for client_info, resi in responses.items():
            self.check_ser_deser(resi)
            if resi.body["is_active"]:
                message = resi.body["split_info"]
                # parse phase 3
                for treeid in message.keys():
                    tree_info = message[treeid]
                    if tree_info["values"][0] == -1:
                        # make leaf
                        self.make_leaf(treeid, self.current_nodes[treeid])
                    else:
                        # split
                        split_side = tree_info['split_side']
                        split_info[split_side][treeid] = {"split_feature": tree_info["split_feature"],
                                                          "split_percentile": tree_info["split_percentile"]}
        for client_info, resi in responses.items():
            requests[client_info] = self.make_request(resi, split_info[str(resi.client_info)], "4")

        # check finish
        #self.check_if_finish()
        return requests

    def random_forest_coordinator_phase5(self, responses):
        """
        Coordinator phase 5 code of federated random forest.

        Parameters
        ----------
        responses : List
            The list of ResponseMessage sending back from the clients.

        Returns
        ------- 
        List
            The list of RequestMessage sending to the clients.
        """
        split_info = {}
        for client_info, resi in responses.items():
            self.check_ser_deser(resi)
            if "is_left" in resi.body:
                for treeid, value in resi.body["is_left"].items():
                    split_info[treeid] = value
                    nodeid = self.current_nodes[treeid]
                    self.forest[treeid][nodeid * 2 + 1] = {"processed": False}
                    self.forest[treeid][nodeid * 2 + 2] = {"processed": False}
                    
        requests = {}
        for client_info, resi in responses.items():
            requests[client_info] = self.make_request(resi, split_info, "5")
        self.release_current_node()
        return requests

    def random_forest_coordinator_inference_phase1(self, responses):
        """
        Coordinator inference phase 1 code of federated random forest.

        Parameters
        ----------
        responses : List
            The list of ResponseMessage sending back from the clients.

        Returns
        ------- 
        List
            The list of RequestMessage sending to the clients.
        """

        requests = {}
        return requests

    def random_forest_coordinator_inference_phase2(self, responses):
        """
        Coordinator inference phase 2 code of federated random forest.

        Parameters
        ----------
        responses : List
            The list of ResponseMessage sending back from the clients.

        Returns
        ------- 
        List
            The list of RequestMessage sending to the clients.
        """

        requests = {}
        body = {}
        for _, resi in responses.items():
            self.check_ser_deser(resi)
            if not resi.body["is_active"]:
                bodyi = resi.body
                if len(bodyi.keys()) > 1:
                    for sid, sample in bodyi.items():
                        # prediction path of a sample id
                        if not sid == "is_active":
                            for treeid, tree in sample.items():
                                # prediction path of a tree
                                if sid in body:
                                    if treeid in body[sid]:
                                        body[sid][treeid].update(tree)
                                    else:
                                        body[sid][treeid] = tree
                                else:
                                    body[sid] = {treeid: tree}
        for client_info, resi in responses.items():
            if resi.body["is_active"]:
                requests[client_info] = self.make_request(resi, body, "-2")
            else:
                requests[client_info] = self.make_request(resi, {}, "-2")
        return requests

    def random_forest_coordinator_inference_phase3(self, responses):
        """
        Coordinator inference phase 1 code of federated random forest.

        Parameters
        ----------
        responses : List
            The list of ResponseMessage sending back from the clients.

        Returns
        ------- 
        List
            The list of RequestMessage sending to the clients.
        """

        for _, resi in responses.items():
            self.check_ser_deser(resi)
            if resi.body["is_active"]:
                self.prediction_array = resi.body["prediction"]
        return self.make_null_requests(responses, "-3")

    # abs
    def post_inference_session(self):
        return self.prediction_array
        
    def post_training_session(self):
        requests = {clienti: RequestMessage(self.machine_info_coordinator, clienti, {}, "99")
                    for clienti in self.machine_info_client}
        return requests
        
    def init_inference_control(self):
        """
        Init inference
        """
        requests = {clienti: RequestMessage(self.machine_info_coordinator, clienti, {}, "-1")
                    for clienti in self.machine_info_client}
        return requests
        
    def init_training_control(self):
        """
        Init training, add parameter into request
        """
        requests = {clienti: RequestMessage(self.machine_info_coordinator, clienti, {}, "0")
                    for clienti in self.machine_info_client}
        return requests
    
    def metric(self):
        return None


if __name__ == "__main__":
    # for single coordinator
    parser = argparse.ArgumentParser()
    parser.add_argument('-C', '--python_config_file', type=str, required=True, help='python config file')

    args = parser.parse_args()
    config = SourceFileLoader("config", args.python_config_file).load_module()

    ip, port = config.coordinator_ip_and_port.split(":")
    coordinator_info = MachineInfo(ip=ip, port=port,
        token=config.coordinator_ip_and_port)
    client_infos = []
    for ci in config.client_ip_and_port:
        ip, port = ci.split(":")
        client_infos.append(MachineInfo(ip=ip, port=port, token=ci))
    coordinator = RandomForestCoordinator(coordinator_info,
                                          client_infos,
                                          config.parameter,
                                          remote=True)
    
    # train
    coordinator.training_pipeline("0", client_infos)
    # inference
