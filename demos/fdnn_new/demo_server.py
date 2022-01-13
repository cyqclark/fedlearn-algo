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
from fdnn_server import FDNNServer
#
from core.entity.common.machineinfo import MachineInfo


def set_args(parser):
    parser.add_argument(
        '--config_path',
        '-C',
        type=str,
        required=False,
        help='path of the configuration file.'
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = set_args(parser)
    config_path = args.config_path
    logging.info(args)

    cfg = ConfigParser()
    cfg.read(args.config_path)

    client_ip = cfg.get('Machine', 'client_ip').split(',')
    client_port = cfg.get('Machine', 'client_port').split(',')
    client_token = cfg.get('Machine', 'client_token').split(',')

    clients_info = []
    for i in range(len(client_ip)):
        clients_info.append(MachineInfo(ip=client_ip[i],
                                        port=client_port[i],
                                        token=client_token[i]))

    server_info = MachineInfo(ip=cfg.get('Machine', 'server_ip'),
                              port=cfg.get('Machine', 'server_port'),
                              token=cfg.get('Machine', 'server_token'))

    server = FDNNServer(server_info)
    server.clients_info = clients_info
    server.clients_token = client_token
    mode = cfg.get('Machine', 'mode')
    if mode == 'train':
        init_phase = "train_init"
        server.training_pipeline(init_phase, server.clients_info)
    elif mode == 'inference':
        init_phase = "inference_init"
        server.inference_pipeline(init_phase, server.clients_info)