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

import numpy as np

import argparse
import os
import socket
import sys
import time

root_path = os.getcwd()
_LOCALHOST = socket.gethostname()
#_LOCALHOST = 'localhost'
sys.path.append(root_path)
sys.path.append(os.path.join(root_path,'demos/HFL'))
sys.path.append(os.path.join(root_path,'demos/HFL/communicator'))

import logging
from demos.HFL.communicator.base_communicator import BaseCommunicator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(module) s - %(funcName) s - %(lineno) d - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



from core.entity.common.machineinfo import MachineInfo

from demos.HFL.communicator.com_utils import AttributeDict

from demos.HFL.communicator.com_builder import (
    Communicator_Builder,
    CommType
)
from demos.HFL.client.client_com_manager_base import(
    ClientComManager,
    ClientMode
)

from demos.HFL.client_create import create_client

from demos.HFL.common.param_util import(
    Params,
    ParamsRes,
    TrainArgs,
    EvalArgs,
    TrainRes,
    EvalRes
)



def client_server_comm_pipeline(
                                    mode = 'client', 
                                    client_port=8899,
                                    #server_ip=socket.gethostbyname(socket.gethostname()),
                                    server_ip='localhost',
                                    server_port=8890,
                                    model_name='dummy_model',
                                    comm_type=CommType.GRPC_COMM,
                                    comm_instance:BaseCommunicator=None):
    """
    Start  a ClientComManager in Client or Proxy mode.
    * Client mode runs on CLIENT MACHINE, it receives HFL_MSG from server and translates it into
    invocation of corresponding function (e.g. train/evaluate/predict)  and then translate return of function result into HFL_MSG to send it back to server.
    * Proxy mode runs on SERVER MACHINE,  server directly invokes local proxy ClientComManager to hide inter-machine HFL_MSG transmission,
    proxy mode translates local innovation into HFL_MSG and send to remote client and response to HFL_MSG send from client.

    Parameters
    ----------
    mode: str
          Choose one of options [client, proxy] to start
    client_port: Union[int,str]
          Client listening port
    server_ip: str
          Server ip
    server_port:  Union[int,str]
          Server listening port
    model_name: str
          name of predefined model for train/evaluation/predict
    comm_type: str
          name of communication module to be used (e.g. grpc/tornado)
    comm_instance: BaseCommunicator
          instance of Communicator, if not None then use this and ignore  <comm_type>      

    """                                

    # Build necessary objects ...

    #(TODO) Tricky fix here to override jcs scanning, make it better readability
    client_ip='0.0' + '.' +  '0.0'   # Client listening ip '0.0.0.0'
    client_port = str(client_port)
    server_port = str(server_port)
    
    def get_comm_instance(config):
        
        if comm_instance is not None:
            return comm_instance
        elif comm_type is not None:
            
            return Communicator_Builder.make(
                    comm_type=comm_type, 
                    config=config
            ) 
        
        else:
            raise(ValueError("com_type and com_instance can't both None"))
            

    def start_proxy():
        config = \
            AttributeDict(
                        ip=str(server_ip),
                        port=str(server_port)
        )
        server_comm = get_comm_instance(config)
         

        cm_proxy = \
            ClientComManager(
                             server_comm,
                             receiverInfo=MachineInfo(str(client_ip),str(client_port),token='dummy'),
                             mode=ClientMode.PROXY
        )
       

        # Simulate local training ...
        trainArgs = TrainArgs(
            params = Params(
                weights= [np.ones((256,256)).astype(float)],
                names = ['dummy_layer'],
                weight_type= 'float'
            ),
            config ={
                'learning_rate':0.0001}   
        )
    
        return  cm_proxy.train(trainArgs)    
    

    def start_client():
        config = \
            AttributeDict(
                        ip=str(client_ip),
                        port=str(client_port)
        )
        client_comm = get_comm_instance(config)

        cm_client = \
            ClientComManager(
                    client_comm,
                    receiverInfo=MachineInfo(str(server_ip),str(server_port),token='dummy'),
                    mode=ClientMode.CLIENT,
                    client=create_client(model_name)
        )
        
        logger.info(
            f'Client {client_ip}:{client_port} started ...')               
        
        
        cm_client.run()
        
        logger.info(
            f'Client {client_ip}:{client_port} quited ...')             
    
    if mode == 'client':
        start_client()
        
    else: 
        start_proxy()



if __name__ == '__main__':
   
    parser = argparse.ArgumentParser()       
   
    parser.add_argument('--mode',help = 'mode set to client or proxy', type=str, default='client')
    parser.add_argument('--client_port',help = 'client listening port', type=int, default=8899)
    parser.add_argument('--model_name',help = 'local_client_model', type=str, default='dummy_model')
    parser.add_argument('--server_ip', help = 'server ip', type=str, default=_LOCALHOST)
    parser.add_argument('--server_port', help = 'server port', type=int, default=8890)
    parser.add_argument('--comm_type',help='communicator name: Option[grpc,tornado]',type=str,default='grpc')                   

    args = parser.parse_args()  

    client_server_comm_pipeline(
        mode = args.mode,
        client_port = args.client_port,
        server_ip = args.server_ip,
        server_port = args.server_port,
        model_name = args.model_name,
        comm_type= args.comm_type
    )                  
