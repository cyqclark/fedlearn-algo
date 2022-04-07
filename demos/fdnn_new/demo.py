import logging
import os
import socket
import sys
import pandas

sys.path.append(os.getcwd())
sys.path.append('/Users/bo.liu/Code/FederatedLearning/OpenSource/fedlearn-algo')
from core.entity.common.machineinfo import MachineInfo
from demos.fdnn_new.fdnn_server import FDNNServer
from demos.fdnn_new.fdnn_client import FDNNClient
import core.server.server

data_path1 = '../../data/classificationA/train0.csv'
data_path2 = '../../data/classificationA/train1.csv'

# set the machine info for coordinator and client
ip = '127.0.0.0'  # 'socket.gethostbyname(socket.gethostname())'
server_info = MachineInfo(ip=ip, port="8890", token="server")
client_info1 = MachineInfo(ip=ip, port="8891", token="client1")
client_info2 = MachineInfo(ip=ip, port="8892", token="client2")

parameter = {"batch_size": 32, "num_epoch": 2}

# create client and coordinator
client1 = FDNNClient(client_info1, parameter)
client1.load_training_data(path=data_path1, feature_names=['Pregnancies','Glucose'], label='Outcome')
client2 = FDNNClient(client_info2, parameter)
client2.load_training_data(path=data_path2, feature_names=['SkinThickness','Insulin'])
client_map = {client_info1: client1, client_info2: client2}

clients_info = [client_info1, client_info2]
server = FDNNServer(server_info, clients_info, parameter)

phase = "0"
# Initialization
init_requests = server.init_training_control()
responses = {}

for client_info, req in init_requests.items():
    client = client_map[client_info]
    responses[client_info] = client.control_flow_client(req.phase_id, req.copy())

while True:
    phase = server.get_next_phase(phase)
    print("Phase %s start..." % phase) 
    requests = server.control_flow_coordinator(phase, responses)
    responses = {}
    if server.is_training_continue():
        for client_info, req in requests.items():
            client = client_map[client_info]
            responses[client_info] = client.control_flow_client(req.phase_id, req.copy())
    else:
        requests = server.fdnn_control_flow_finish()
        break

for client_info, req in requests.items():
    client = client_map[client_info]
    responses[client_info] = client.control_flow_client(req.phase_id, req.copy())
for client_info, client in client_map.items():
    print(str(client_info))
