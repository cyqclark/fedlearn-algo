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
sys.path.append("./")

import torch
from core.entity.common.machineinfo import MachineInfo
from core.grpc_comm.grpc_server import serve
from fdnn_client import  FDNNClient
import argparse
from configparser import ConfigParser

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # load config file
    parser.add_argument('-C', '--config_path', type=str, required=True, help='path of the configuration file')

    args = parser.parse_args()

    cfg = ConfigParser()
    cfg.read(args.config_path)

    mode = cfg.get('Machine', 'mode')

    client_info = MachineInfo(ip=cfg.get('Machine', 'ip'),
                              port=cfg.get('Machine', 'port'),
                              token=cfg.get('Machine', 'token'))


    client = FDNNClient(client_info)


    data_path = cfg.get('Data', 'data_path')
    if mode == 'train':
        features = cfg.get('Data', 'features').split(',')
        if cfg.has_option('Data', 'label'):
            # active party
            label = cfg.get('Data', 'label')
            client.load_training_data(data_path=data_path,
                                      feature_names=features,
                                      label_name=[label])
        else:
            # passive party
            client.load_training_data(data_path=data_path, feature_names=features)
    elif mode == 'inference':
        client.load_data(data_path=data_path, feature_names=cfg.get('Data', 'features').split(','))
        client.model = client.load_model(model_path=cfg.get('Model', 'model_path'))
        
    serve(client)