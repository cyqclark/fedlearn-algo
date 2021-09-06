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

import argparse
from configparser import ConfigParser
import logging
# add core's path
import sys
sys.path.append("./")
#
from client_kernelmethod import KernelMethodClient
from server_kernelmethod import KernelMethodsServer
from kernelmethod import KernelMappingParam
#
from core.entity.common.machineinfo import MachineInfo
from core.grpc_comm.grpc_server import serve


def load_coordinator(cfg, args):
    parser = argparse.ArgumentParser()
    config_path = args.config_path
    logging.info(args)

    client_ip = cfg.get('Coordinator', 'client_ip').split(',')
    client_port = cfg.get('Coordinator', 'client_port').split(',')
    client_token = cfg.get('Coordinator', 'client_token').split(',')

    clients_info = []
    for i in range(len(client_ip)):
        clients_info.append(MachineInfo(ip=client_ip[i],
                                        port=client_port[i],
                                        token=client_token[i]))

    server_info = MachineInfo(ip=cfg.get('Coordinator', 'server_ip'),
                              port=cfg.get('Coordinator', 'server_port'),
                              token=cfg.get('Coordinator', 'server_token'))

    server = KernelMethodsServer(server_info)
    server.clients_info = clients_info
    server.clients_token = client_token
    return server

def load_client(cfg, args):

    mode = cfg.get('Machine', 'mode')

    client_info = MachineInfo(ip=cfg.get('Machine', 'ip'),
                              port=cfg.get('Machine', 'port'),
                              token=cfg.get('Machine', 'token'))
    client = KernelMethodClient(client_info)

    data_path = cfg.get('Data', 'data_path')

    if mode == 'train':
        features = cfg.get('Data', 'features').split(',')
        if cfg.has_option('Data', 'label'):
            # active party
            label = cfg.get('Data', 'label')
            client.load_data(data_path=data_path,
                             feature_names=features,
                             label_name=[label])
        else:
            # passive party
            client.load_data(data_path=data_path, feature_names=features)

        client.normalization(norm_type=cfg.get('Data', 'normalization'))
        client.kernel_mapping_param = KernelMappingParam(scale=float(cfg.get('Model', 'scale')),
                                                         feature_dim=len(features),
                                                         map_dim=int(cfg.get('Model', 'map_dim')),
                                                         seed=0)
        # client.model_path = cfg.get('Model', 'model_path')
        client.model_path = cfg.get('Model', 'model')
    elif mode == 'inference':
        client.load_data(data_path=data_path, feature_names=cfg.get('Data', 'features').split(','))
        client.model = client.load_model(model_path=cfg.get('Model', 'model_path'))
    else:
        raise ValueError("Invalid mode!")
    return client


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # load config file
    parser.add_argument('-C', '--config_path', type=str, required=True, help='path of the configuration file')
    parser.add_argument('-F', '--flag_network', type=str, required=False, default="F", help='flag to use new network api')
    
    args = parser.parse_args()

    cfg = ConfigParser()
    cfg.read(args.config_path)

    client = load_client(cfg, args)
    
    if ("flag_network" not in args) or (args.flag_network == "F"):
        # old api framework
        serve(client)
    elif args.flag_network == "T":
        # new api framework
        mode = cfg.get('Machine', 'mode')
        if mode == 'train':
            # serve
            if cfg.has_option('Data', 'label'):
                # create coordinator
                coordinator = load_coordinator(cfg, args)
                client.load_coordinator(coordinator)
                init_phase = "train_init"
                client._exp_training_pipeline(init_phase, client.coordinator.clients_info)
            else:
                serve(client)
        elif mode == 'inference':
            raise NotImplementedError("Not implemented yet!")
            #client.load_data(data_path=data_path, feature_names=cfg.get('Data', 'features').split(','))
            #client.model = client.load_model(model_path=cfg.get('Model', 'model_path'))
    else:
        raise ValueError("Invalid flag network")

