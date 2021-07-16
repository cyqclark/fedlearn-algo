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

# assume cwd is ./opensource/
import logging
import os
import socket
import sys
import pandas
sys.path.append(os.getcwd())
print(sys.path)
from core.entity.common.machineinfo import MachineInfo
from demos.random_forest.coordinator import RandomForestCoordinator
from demos.random_forest.client import RandomForestClient

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
dataset1 = {"label": g1.Outcome.values.astype(float),
            "feature": g1.loc[:, g1.columns[1:-1]].values}
dataset2 = {"label": None,
            "feature": g2.loc[:, g2.columns[1:]].values}
dataset3 = {"label": None,
            "feature": g3.loc[:, g3.columns[1:]].values}

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
dataset1["feature_inference"] = g1.loc[:, g1.columns[1:]].values
dataset2["feature_inference"] = g2.loc[:, g2.columns[1:]].values
dataset3["feature_inference"] = g3.loc[:, g3.columns[1:]].values

# set the machine info for coordinator and client
ip = socket.gethostbyname(socket.gethostname())
coordinator_info = MachineInfo(ip=ip, port="8890", token="%s:8890"%ip)
client_info1 = MachineInfo(ip=ip, port="8891", token="%s:8891"%ip)
client_info2 = MachineInfo(ip=ip, port="8892", token="%s:8892"%ip)
client_info3 = MachineInfo(ip=ip, port="8893", token="%s:8893"%ip)

parameter = {
    "numTrees": 3,
    "maxDepth": 5,
    "maxSampledFeatures": 10,
    "maxSampledRatio": 0.6,
    "numPercentiles": 3,
    "minSamplesSplit": 3,
    "eval_metric": ["RMSE"],
    "loss": "RMSE",
    "maxTreeSamples": 2000,
    "encryptionType": "Paillier",
}

# create client and coordinator
client1 = RandomForestClient(client_info1, parameter, dataset1, remote=True)
client2 = RandomForestClient(client_info2, parameter, dataset2, remote=True)
client3 = RandomForestClient(client_info3, parameter, dataset3, remote=True)
client_map = {client_info1: client1,
              client_info2: client2,
              client_info3: client3}

client_infos = [client_info1, client_info2, client_info3]
coordinator = RandomForestCoordinator(coordinator_info, client_infos, parameter, remote=True)

# training
# coordinatorserver.training_pipeline()

phase = "0"
# Initialization
init_requests = coordinator.init_training_control()
responses = {}
for client_info, reqi in init_requests.items():
    client = client_map[client_info]
    responses[client_info] = client.control_flow_client(reqi.phase_id, reqi)

while True:
    phase = coordinator.get_next_phase(phase)
    print("Phase %s start..."%phase)
    requests = coordinator.control_flow_coordinator(phase, responses)
    responses = {}
    if coordinator.is_training_continue():
        for client_info, reqi in requests.items():
            client = client_map[client_info]
            responses[client_info] = client.control_flow_client(reqi.phase_id, reqi.copy())
    else:
        break

# use this code print the client info and the corresponding tree info
#for client_info, client in client_map.items():
#    print(str(client_info))
#    print(str(client.forest))

# inference
phase = "-1"
# Initialization
init_requests = coordinator.init_inference_control()
responses = {}

for client_info, reqi in init_requests.items():
    client = client_map[client_info]
    responses[client_info] = client.control_flow_client(reqi.phase_id, reqi)

while True:
    phase = coordinator.get_next_phase(phase)
    #logging.info("Phase %s start..."%phase)
    print("Phase %s start..."%phase)
    requests = coordinator.control_flow_coordinator(phase, responses)
    responses = {}
    if coordinator.is_inference_continue():
        for client_info, reqi in requests.items():
            client = client_map[client_info]
            responses[client_info] = client.control_flow_client(reqi.phase_id, reqi.copy())
    else:
        break

# post inference
prediction = coordinator.post_inference_session()
print("Prediction: ")
print(prediction)
