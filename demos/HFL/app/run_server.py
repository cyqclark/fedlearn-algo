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

import os,sys
import numpy as np

import argparse
root_path = os.getcwd()

sys.path.append(root_path)
sys.path.append(os.path.join(root_path,'demos/HFL'))

from demos.HFL.server.server import Server


def run_server(args):
    """
    Initiate and start a server
    """
    server = Server(
        port=args.port,
        num_clients_to_train=args.num_client,
        comm_type=args.comm_type,
        algo_arch=args.algo_arch)
    
    server.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()       
    
    parser.add_argument('--num_client',help = 'number of clients to start train', type=int, default=2)
    parser.add_argument('--port',help = 'port', default=8890) 
    parser.add_argument('--comm_type',help='communicator name: Option[grpc,tornado]',type=str, default='grpc')                 
    parser.add_argument('--algo_arch',help='algorithm architecture : Option[sync,async]',type=str, default='sync')                 
    args = parser.parse_args()
    
    run_server(args)
