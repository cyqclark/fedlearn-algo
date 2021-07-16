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

"""Example code of federated random forest client.


Example
-------
TBA::

    $ TBA


Notes
-----
    This is an example of federated random forest client. It assumes the
default folder is './opensource'.


"""

import argparse
import pdb
import orjson
import os
import sys

from importlib.machinery import SourceFileLoader

import numpy
import pandas

# assume cwd is ./opensource/
sys.path.append(os.getcwd())
from core.entity.common.machineinfo import MachineInfo
from core.entity.common.message import RequestMessage, ResponseMessage
from core.client.client import Client
from core.grpc_comm.grpc_server import serve


class RandomForestClient(Client):
    """
    Random forest client class.

    Note
    ----

    Parameters
    ----------
    machine_info : MachineInfo
        The machine info class that save the current client information,
        including ip, port and token.
    parameter : Dict
        The parameter of federated random forest model for this training task.
    dataset : Dict
        The binding dataset of this training task, including 'feature' and 'label' key.

    Attributes
    ----------
    machine_info : MachineInfo
        The machine_info argument passing in.
    parameter : Dict
        The parameter argument passing in.
    dataset : Dict
        The dataset argument passing in.
    """

    def __init__(self,
                 machine_info,
                 parameter,
                 dataset,
                 remote=False):
        # super.__init__(parameter)
        # pass arguments
        self.parameter = parameter
        self.machine_info = machine_info
        self.dataset = dataset
        # set function mapping
        self.dict_functions = {"0": self.initialization_client,
                               "1": self.random_forest_client_phase1,
                               "2": self.random_forest_client_phase2,
                               "3": self.random_forest_client_phase3,
                               "4": self.random_forest_client_phase4,
                               "5": self.random_forest_client_phase5,
                               "99": self.random_forest_client_post_training,
                               "-1": self.random_forest_client_inference_phase1,
                               "-2": self.random_forest_client_inference_phase2,
                               "-3": self.random_forest_client_inference_phase3,
                               }
        # no preprocessing or postprocessing in this demo training code
        self.preprocessing_func = {}
        self.postprocessing_func = {}
        # use current_nodes dictionary as current active node in each tree.
        self.current_nodes = None
        # the underlying random forest
        self.forest = {}

        # check if use remote mode
        self.remote = remote
        return None

    def make_response(self, request, body):
        """
        Making response function. Given the input request and body dictionary,
        this function returns a response object which is ready for sending out.

        Parameters
        ----------
        request : RequestMessage
            The request message sending into the client.
        body : Dict
            The response body data in dictionary type.

        Returns
        -------
        ResponseMessage
            The response message which is ready for sending out.
        """
        # check if the current client is an active client.
        if "is_active" not in body:
            body["is_active"] = self.is_active
        response = ResponseMessage(self.machine_info,
                                   request.server_info,
                                   body,
                                   phase_id=request.phase_id)
        if self.remote:
            response.serialize_body()
        return response

    def make_prediction(self, treeid, nodeid):
        """
        Make prediction function.
        """
        sample_id = self.forest[treeid]["tree"][nodeid]["sample_id"]
        self.forest[treeid]["tree"][nodeid]["is_leaf"] = True
        self.forest[treeid]["tree"][nodeid]["prediction"] = numpy.mean(self.dataset["label"][sample_id])
        return None

    def update_split_info(self, treeid, feature, value):
        """
        Updating split information function.
        """
        nodeid = self.current_nodes[treeid]
        self.forest[treeid]["tree"][nodeid]["feature"] = feature
        self.forest[treeid]["tree"][nodeid]["value"] = value
        self.forest[treeid]["tree"][nodeid]["is_leaf"] = False
        return None

    def control_flow_client(self,
                            phase_num,
                            request):
        """
        The main control flow of client. This might be able to work in a generic
        environment.
        """
        # if phase has preprocessing, then call preprocessing func
        response = request
        if phase_num in self.preprocessing_func:
            response = self.preprocessing_func[phase_num](response)
        if phase_num in self.dict_functions:
            response = self.dict_functions[phase_num](response)
        # if phase has postprocessing, then call postprocessing func
        if phase_num in self.postprocessing_func:
            response = self.postprocessing_func[phase_num](response)
        return response

    def initialization_client(self, request):
        """
        Client initialization function.

        Parameters
        ----------
        request : RequestMessage
            The request message sending into the client.

        Returns
        -------
        """
        self.is_active = self.dataset["label"] is not None
        self.label = self.dataset["label"]
        body = {"feature": self.dataset["feature"].shape[1],
                "num_sample": self.dataset["feature"].shape[0]}
        # if active then send encrypted label
        if self.is_active:
            body["encrypted_label"] = encrypt(self.dataset["label"])
        return self.make_response(request, body)

    def check_ser_deser(self, message):
        if self.remote:
            if isinstance(message, RequestMessage):
                message.deserialize_body()
            elif isinstance(message, ResponseMessage):
                message.serialize_body()
        return None

    def parse_key(self, message):
        message.body = {int(key): value for key, value in message.body.items()}
        return None

    def random_forest_client_phase1(self, request):
        """
        Client phase 1 code of federated random forest.

        Parameters
        ----------
        request : RequestMessage
            The request message sending into the client.

        Returns
        -------
        """
        # receive encrypted_label
        self.check_ser_deser(request)
        if not self.is_active:
            self.encrypted_label = request.body["encrypted_label"]
        feature_selected = request.body["feature_selected"]
        print(str(self.machine_info) + "feature selected: " + str(feature_selected))
        sample_ids = request.body["sample_id"]
        for treeid, features in feature_selected.items():
            self.forest[treeid] = {"feature": features}
        for treeid, sample_id in sample_ids.items():
            self.forest[treeid]["sample_id"] = numpy.array(sample_id)
        for treeid in self.forest.keys():
            # create root
            root = {"sample_id": self.forest[treeid]["sample_id"]}
            self.forest[treeid]["tree"] = {0: root}
        body = {}
        return self.make_response(request, body)

    def random_forest_client_phase2(self, request):
        """
        Client phase 2 code of federated random forest.

        Parameters
        ----------
        request : RequestMessage
            The request message sending into the client.

        Returns
        -------
        """
        self.check_ser_deser(request)
        self.parse_key(request)

        res = {}
        self.current_nodes = request.body
        pop_treeid = []
        for treeid in self.current_nodes.keys():
            tree = self.forest[treeid]
            nodeid = self.current_nodes[treeid]
            node = tree["tree"][nodeid]
            feature = tree["feature"]

            index = node["sample_id"]

            if (len(index) < self.parameter["minSamplesSplit"]) or (nodeid > 2 ** (self.parameter["maxDepth"] - 1)):
                if self.is_active:
                    self.make_prediction(treeid, nodeid)
                pop_treeid.append(treeid)
                res[treeid] = {"skip": True}
            # feature data shape: [num_samples, num_features]
            else:
                X = self.dataset["feature"][index, :][:, feature]
                y = self.dataset["label"][index] if self.is_active else self.encrypted_label[index]

                dim_feature = X.shape[1]

                # get number of bins
                bins = self.parameter["numPercentiles"]

                # create percentiles
                percentiles = numpy.arange(0., 100., 100. / bins)
                if 100. not in percentiles:
                    percentiles = numpy.append(percentiles, 100)

                Y1 = numpy.zeros([dim_feature, bins], dtype=y.dtype)

                target_value = numpy.percentile(X, percentiles, axis=0, interpolation='higher')
                target_value[-1] = target_value[-1] + 1e-8  # include the largest value
                dim_percentile = percentiles.shape[0] - 1
                for i in range(dim_feature):
                    for pi in range(dim_percentile):
                        if target_value[pi, i] == target_value[pi + 1, i]:
                            # left = right, extract categorical feature
                            if (pi > 0) and (target_value[pi, i] == target_value[pi - 1, i]):  # optimize part
                                Y1[i, pi] = Y1[i, pi - 1]
                            else:
                                flag = X[:, i] == target_value[pi, i]
                                if numpy.any(flag):
                                    Y1[i, pi] = numpy.mean(y[flag])
                                else:
                                    print("Exception on Phase2: ")
                                    print("Percentiles: %s" % str(target_value[:, i]))
                                    print("No value for %s" % str(target_value[pi, i]))
                                    numpy.mean(y[flag])
                        else:
                            # left != right, calculate interval mean
                            if (pi > 0) and (target_value[pi, i] == target_value[pi - 1, i]):  # optimize part
                                flag = (X[:, i] > target_value[pi, i]) & (X[:, i] < target_value[pi + 1, i])
                            else:
                                flag = (X[:, i] >= target_value[pi, i]) & (X[:, i] < target_value[pi + 1, i])
                            if numpy.any(flag):
                                Y1[i, pi] = numpy.mean(y[flag])
                            else:
                                print("Warning: No sample in percentile, bleed previous value")
                                Y1[i, pi] = Y1[i, pi - 1]

                res[treeid] = dict(bin_sum_of_Y=Y1)
        for treeid in pop_treeid:
            self.current_nodes.pop(treeid, None)
        body = {"res": res,
                "is_active": self.is_active}
        return self.make_response(request, body)

    def random_forest_client_phase3(self, request):
        """
        Client phase 3 code of federated random forest.

        Parameters
        ----------
        request : RequestMessage
            The request message sending into the client.

        Returns
        -------
        """
        self.check_ser_deser(request)
        self.parse_key(request)
        # if passive client, pass
        if not self.is_active:
            return self.make_response(request, {})

        Ys = request.body
        res = {}
        if not Ys:
            return self.make_response(request, {})

        for treeid in Ys.keys():
            client_infos = []
            bin_sum_of_Y = []

            tmp = Ys[treeid]
            for client in tmp.keys():
                client_infos.append(client)
                bin_sum_of_Y.append(tmp[client]["bin_sum_of_Y"])

            # bin_sum_of_Y = request["bin_sum_of_Y"]
            # mean_Y = numpy.sum(self.dataset["label"])
            mean_Y = numpy.mean(Ys[treeid][str(self.machine_info)]["bin_sum_of_Y"])

            if self.parameter["loss"] == "RMSE":
                res[treeid] = Phase3Variance(bin_sum_of_Y, [mean_Y], treeid)
            else:
                res[treeid] = Phase3Entropy(bin_sum_of_Y, [mean_Y], treeid)
            # check if is_leaf
            if res[treeid]["values"][0] == -1:
                try:
                    self.make_prediction(treeid, self.current_nodes[treeid])
                except:
                    import pdb
                    pdb.set_trace()
            else:
                split_side, feature_id, opt_percentile, opt_values = res[treeid]["values"]
                res[treeid]["split_side"] = client_infos[split_side]
                res[treeid]["split_feature"] = feature_id
                res[treeid]["split_percentile"] = opt_percentile

        return self.make_response(request, {"split_info": res})

    def random_forest_client_phase4(self, request):
        """
        Client phase 4 code of federated random forest.

        Parameters
        ----------
        request : RequestMessage
            The request message sending into the client.

        Returns
        -------
        """
        self.check_ser_deser(request)
        self.parse_key(request)
        if not request.body:
            # no split in this client return null response
            return self.make_response(request, {})

        res = {}

        for treeid, split_info in request.body.items():
            tree_info = self.forest[treeid]
            node_id = self.current_nodes[treeid]
            sample_id = tree_info["tree"][node_id]["sample_id"]
            X = self.dataset["feature"][sample_id, :]
            feature_opt = self.forest[treeid]["feature"][int(split_info["split_feature"])]
            percentile_opt = split_info["split_percentile"]

            X_select = X[:, int(feature_opt)]
            target_number = numpy.percentile(X_select, percentile_opt, interpolation='higher')

            flag = X_select < target_number

            if not numpy.any(flag):  # no left child
                tmp = numpy.min(X_select) + 1e-8
                print("Warning: no left child, try to shift target number from %.6f to %.6f" % (target_number,
                                                                                                tmp))
                target_number = tmp
                flag = X_select < target_number

            X_left = X[flag, :]
            X_right = X[~flag, :]

            # update tree info to forest
            self.update_split_info(treeid, feature_opt, target_number)

            res[treeid] = flag
        return self.make_response(request, {"is_left": res})

    def random_forest_client_phase5(self, request):
        self.check_ser_deser(request)
        self.parse_key(request)
        for treeid, is_left in request.body.items():
            node_id = self.current_nodes.pop(treeid)
            sample_id = self.forest[treeid]["tree"][node_id]["sample_id"]
            self.forest[treeid]["tree"][node_id * 2 + 1] = {"sample_id": sample_id[is_left]}
            self.forest[treeid]["tree"][node_id * 2 + 2] = {"sample_id": sample_id[~is_left]}
            self.forest[treeid]["tree"][node_id].pop("sample_id", None)

        return self.make_response(request, {})

    def random_forest_client_post_training(self, request):
        # TODO: serialization and save
        return self.make_response(request, {})

    def random_forest_client_inference_phase1(self, request):
        """
        Inference phase 1:
        All passive party process data and generate decision path
        """
        body = {}
        X_test = self.dataset["feature_inference"]
        if self.is_active:
            self.prediction_array = numpy.zeros(X_test.shape[0])
        for i in range(X_test.shape[0]):
            for treeid, tree in self.forest.items():
                for nodeid, node in tree["tree"].items():
                    if 'is_leaf' in node:
                        # assign information of ith sample at treeid, nodeid
                        if node["is_leaf"]:
                            if i in body:
                                if treeid in body[i]:
                                    body[i][treeid].update({nodeid: node["prediction"]})
                                else:
                                    body[i][treeid] = {nodeid: node["prediction"]}
                            else:
                                body[i] = {treeid: {nodeid: node["prediction"]}}
                        else:
                            feature = node["feature"]
                            if i in body:
                                if treeid in body[i]:
                                    body[i][treeid].update({nodeid: X_test[i, feature] <= node["value"]})
                                else:
                                    body[i][treeid] = {nodeid: X_test[i, feature] <= node["value"]}
                            else:
                                body[i] = {treeid: {nodeid: X_test[i, feature] <= node["value"]}}
        if self.is_active:
            self.predict_path = body
            return self.make_response(request, {})
        else:
            return self.make_response(request, body)

    def random_forest_client_inference_phase2(self, request):
        """
        Inference phase 2:
        Passive party collect all passive path for each sample
        """
        self.check_ser_deser(request)
        self.parse_key(request)
        body = {}
        if self.is_active:
            # merge dictionaries
            for sid, sample in request.body.items():
                for treeid, treei in sample.items():
                    self.predict_path[sid][treeid].update(treei)
            for sid in range(self.prediction_array.shape[0]):
                sample = self.predict_path[sid]
                predicts = numpy.zeros(len(sample.keys()))
                for treei, tree in sample.items():
                    cur = 0
                    while isinstance(tree[cur], numpy.bool_):
                        cur = cur * 2 + 2 if tree[cur] else cur * 2 + 1
                    try:
                        predicts[treei] = tree[cur]
                    except:
                        import pdb
                        pdb.set_trace()
                    x = 1
                self.prediction_array[sid] = numpy.mean(predicts)
            body["prediction"] = self.prediction_array
        return self.make_response(request, body)

    def random_forest_client_inference_phase3(self, request):
        return self.make_response(request, {})


def Phase3Variance(matrices, values, index):
    """
    Best split selection code. The underlying loss function is mean squared error(MSE).

    Parameters
    ----------

    Returns
    -------
    """
    Ys = matrices
    square_of_sum_label = values[0] ** 2
    if len(values) > 2:
        boosting = 1 + values[2]
    else:
        boosting = 1 + 1e-6

    party_opt = None
    feature_opt = None
    percentile_opt = None
    score_opt = numpy.finfo(numpy.dtype(float)).min

    # here we assume bins have equally number of samples
    confused = 0
    for i in range(len(Ys)):
        # split sum into left sum and right sum by cumsum trick
        if Ys[i].size > 0:
            # unique check
            flag_unique = numpy.all(numpy.abs(Ys[i] - Ys[i][:, :1]) < 1e-8, axis=1)
            cumsum_Y = numpy.cumsum(Ys[i], axis=1)
            max_enumerate = cumsum_Y.shape[1] - 1
            for k in range(max_enumerate):
                E_zl = cumsum_Y[:, k] / (k + 1)
                E_zr = (cumsum_Y[:, -1] - cumsum_Y[:, k]) / (max_enumerate - k)
                score_i = (E_zl ** 2 * (k + 1) + E_zr ** 2 * (max_enumerate - k)) / max_enumerate
                # move out score_i by set them to -1e8
                score_i[flag_unique] = -1e8
                feature_opt_i = numpy.argmax(score_i)
                if score_i[feature_opt_i] > score_opt * (1 + 1e-8):
                    confused = 0
                    party_opt = i
                    feature_opt = feature_opt_i
                    percentile_opt = k
                    score_opt = score_i[feature_opt_i]
                elif abs(score_i[feature_opt_i] - score_opt) < 1e-8:
                    confused += 1

    if score_opt > (square_of_sum_label / cumsum_Y.shape[1]) * boosting:
        node_message = orjson.dumps({"is_leaf": 0})
        return dict(index=index,
                    values=[party_opt, feature_opt, percentile_opt, score_opt],
                    message=node_message)
    else:
        node_message = orjson.dumps({"is_leaf": 1})
        return dict(index=index,
                    values=[-1, 0, 0, 0],
                    message=node_message)


def Phase3Entropy(matrices, values, index):
    """
    Best split selection code. The underlying loss function is cross entropy.

    Parameters
    ----------

    Returns
    -------
    """
    Ys = matrices
    entropy_label = binaryClassEntropy(values[0])
    if len(values) > 2:
        boosting = 1 + values[2]
    else:
        boosting = 1 + 1e-6

    party_opt = None
    feature_opt = None
    percentile_opt = None
    score_opt = numpy.finfo(numpy.dtype(float)).max

    # here we assume bins have equally number of samples
    confused = 0
    for i in range(len(Ys)):
        # split sum into left sum and right sum by cumsum trick
        if Ys[i].size > 0:
            cumsum_Y = numpy.cumsum(Ys[i], axis=1)
            max_enumerate = cumsum_Y.shape[1] - 1
            for k in range(max_enumerate):
                # get entropy on left
                E_zl = cumsum_Y[:, k] / (k + 1)
                entropy_left = binaryClassEntropy(E_zl)
                # get entropy on right
                E_zr = (cumsum_Y[:, -1] - cumsum_Y[:, k]) / (max_enumerate - k)
                entropy_right = binaryClassEntropy(E_zr)
                score_i = (entropy_left * (k + 1) + entropy_right * (max_enumerate - k)) / (max_enumerate + 1)
                feature_opt_i = numpy.argmax(score_i)
                # minimize cross entropy
                if score_i[feature_opt_i] < score_opt * (1 - 1e-6):
                    confused = 0
                    party_opt = i
                    feature_opt = feature_opt_i
                    percentile_opt = k
                    score_opt = score_i[feature_opt_i]
                elif abs(score_i[feature_opt_i] - score_opt) < 1e-6:
                    confused += 1

    if entropy_label > score_opt * boosting:
        node_message = orjson.dumps({"is_leaf": 0})
        res = dict(index=index,
                   values=[party_opt, feature_opt, percentile_opt, score_opt],
                   message=node_message)
        return res
    else:
        node_message = orjson.dumps({"is_leaf": 1})
        return dict(index=index,
                    values=[-1, 0, 0, 0],
                    message=node_message)


def binaryClassEntropy(p):
    '''
    Binary Class Entropy function
    return -p * log(p) - (1-p) * log(1-p)
    '''
    if isinstance(p, float):
        # float version
        if (p < 1e-8) or (1 - p < 1e-8):
            return 2e-8
        else:
            return -p * numpy.log(p) - (1 - p) * numpy.log(1 - p)
    else:
        # numpy array version
        flag = (p < 1e-8) | (1 - p < 1e-8)
        res = numpy.empty(p.shape)
        res[flag] = 2e-8
        p1 = p[~flag]
        res[~flag] = -p1 * numpy.log(p1) - (1 - p1) * numpy.log(1 - p1)
        return res


def encrypt(x):
    return x


if __name__ == "__main__":
    # for single client
    parser = argparse.ArgumentParser()
    parser.add_argument('-I', '--index', type=int, required=True, help='index of client')
    parser.add_argument('-C', '--python_config_file', type=str, required=True, help='python config file')

    args = parser.parse_args()
    idx = args.index
    config = SourceFileLoader("config", args.python_config_file).load_module()

    # load data
    g = pandas.read_csv(config.client_train_file_path[idx])
    if idx == config.active_index:
        label = g.pop(config.active_label).values.ravel().astype(float)
        dataset = {"label": label,
                   "feature": g.loc[:, g.columns[1:]].values}
    else:
        dataset = {"label": None,
                   "feature": g.loc[:, g.columns[1:]].values}
    if "client_inference_file_path" in config.__dict__:
        g = pandas.read_csv(config.client_inference_file_path[idx])
        dataset["feature_inference"] = g.loc[:, g.columns[1:]].values
    ip, port = config.client_ip_and_port[idx].split(":")
    client_info = MachineInfo(ip=ip, port=port,
                              token=config.client_ip_and_port[idx])

    parameter = config.parameter
    client = RandomForestClient(client_info, parameter, dataset, remote=True)

    serve(client)

