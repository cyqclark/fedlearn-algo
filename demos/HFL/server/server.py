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
import concurrent.futures

root_path = os.getcwd()

import numpy as np
from typing import List,Tuple
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(module) s - %(funcName) s - %(lineno) d - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



sys.path.append(root_path)
sys.path.append(os.path.join(root_path,'demos/HFL'))
sys.path.append(os.path.join(root_path,'demos/HFL/communicator'))

from core.entity.common.machineinfo import MachineInfo
from demos.HFL.communicator import com_utils
from com_utils import AttributeDict

from demos.HFL.communicator.com_builder import (
    Communicator_Builder,
    CommType
)

from demos.HFL.communicator.com_utils import GeneralCommunicator

from demos.HFL.common.msg_handler import Msg_Handler
from demos.HFL.common.hfl_message import HFL_MSG
from demos.HFL.basic_control_msg_type import  HFL_Control_Massage_Type as CMT
from demos.HFL.algorithm.fed_avg import FedAvg
from demos.HFL.common.param_util import(
    Params, 
    ParamsRes, 
    TrainArgs, 
    TrainRes, 
    EvalArgs,
)

from demos.HFL.client.client_com_manager_base import (
    ClientComManager,
    ClientMode
)

<<<<<<< HEAD
_NUM_ROUNDS_= 2 #10
=======
_NUM_ROUNDS_= 5 #10
>>>>>>> d90d01507ac2bb03005161ad549f101109c79e99
class Server(Msg_Handler):
    """
    Federal learning Server
    
    There are two modes to start a Server: Passive and Active mode,
    In Passive mode, Server waits for client to be connected, once the minimal numbers of 
    clients are connected to server, it will send  command to clients to start training/evaluation.
    whereas in Active mode (To be implemented), once server started, it connects to clients and immediate
    start training/evaluation/predict.
    Server connected to clients by creating numbers of ClientComManager in Proxy Mode, each corresponds to a remote client,
    and Server call  each ClientComManager to control client as it is in local machine.
    
 
    Parameters
    ----------
    port: int
        Server's listening port.
    num_clients_to_train:int
        numbers of client to start training in Passive mode.
    comm_type: CommType
        Which of communication modules to be used, currently two available,
        CommType.TORNADO_COMM is based on Tornado and CommType.GRPC_COMM on GRPC,
        when comm_instance is not None , <comm_type> will be ignored.
    comm_instance:  GeneralCommunicator
        Instance of Communicator, if not None, Server will uses this instance as Communicator and ignore <comm_type> .

    """
    

    MODE_ACTIVE='active'
    MODE_PASSIVE='passive'
    

    def __init__(self, 
                 port:int=8890, 
                 num_clients_to_train:int=2,
                 comm_type=CommType.TORNADO_COMM,
                 comm_instance:GeneralCommunicator=None,
                 clients_ip_port:List[str]=None):
                
        
        self.port = str(port)
        self.num_clients_to_train = num_clients_to_train

        self.client_proxy_managers = {}
       
        com_config = \
            AttributeDict(
                ip='0.0'+'.'+'0.0',
                port=str(self.port),
                mode='proxy'
        )
        if comm_type is None and comm_instance is not None:
            self.comm = comm_instance
        elif comm_type is not None:
            self.comm = \
                Communicator_Builder.make(
                    comm_type=comm_type,
                    config=com_config
                )
        else: 
            raise(ValueError("comm or com_base can't both be None"))

        
        self.comm.add_msg_handler(self)

        if  clients_ip_port:
            self.client_ip_port_list:List[Tuple[str,str]] = \
                [self._parse_client_ip_port(ip_port) for ip_port in self. clients_ip_port]

        logger.info(f'There are {len(self.comm._msg_handlers)} handlers ...') 
        
        logger.info('Done server init ... ')
    
    def _parse_client_ip_port(self,text:str):
        ip_port = text.strip().split(":")
        try:
            ip = ip_port[0].strip()
            port =ip_port[1].strip() 
        except Exception as e:
            logger.info(e)    
        return ip, port
    
    def run(self, server_mode='passive'):
        if server_mode==Server.MODE_PASSIVE:
            self.comm.run()
        else: 
            raise(NotImplementedError(f'{server_mode} has not been implemented ...'))        
    
    def handle_message(self, msg_type:str, msg:HFL_MSG):
        logger.info(f'Server received  request... {msg_type}')
        if msg.type == CMT.CTRL_CLIENT_JOIN_C2S:
            self.handle_msg_connect(msg)
        elif msg.type == CMT.CTRL_CLIENT_DISJOIN_C2S:
            self.handle_msg_disconnect(msg)
        else: 
            logger.debug(f'Server ignore request ... {msg_type}')
        
    
    def handle_msg_connect(self, msg:HFL_MSG):
        """
        1. when receiving client connection request, 
        create new client proxy and kept in Dict
        
        2. start training if the minimum number of clients is reached.
        """
        logger.info('Server: Client is connecting ...')
        
        # check if sender is already connected
        if msg.sender not in self.client_proxy_managers:

                self.client_proxy_managers[msg.sender] = \
                    self.create_client_proxy_manager(msg.sender)
                
                logger.info(f'Server: New client connected from {msg.sender.ip}:{msg.sender.port}')
                
                if len(self.client_proxy_managers.items()) >= self.num_clients_to_train:
                        # wait a few seconds for the possible delay of establishing listening port on remote client
                        time.sleep(3)
                        self.train(_NUM_ROUNDS_)
                 
    
    def handle_msg_disconnect(self,msg:HFL_MSG):
          
          if msg.sender in self.client_proxy_managers:
              logger.info(f'Server: Client disconnected from {msg.sender.ip}:{msg.sender.port}')

          self.client_proxy_managers.pop(msg.sender, None)
                     

    def create_client_proxy_manager(self, 
                                     receiverInfo : MachineInfo
                                    )->ClientComManager:
            return ClientComManager(
                             self.comm,
                             receiverInfo=receiverInfo,
                             mode=ClientMode.PROXY
                    )                             

    
    def init_globel_model(self):
        
        def _init_dummy_model_weights():
            params = Params(
                weights =[np.ones((256,256))],
                names =['layer_1'],
                weight_type='float'
                )
            return params
        
        def _init_trainConfigs(num_clients):
            config ={
                'learning_rate' :0.0001,
                'optimizer': 'adam',
                'sample_ratio_to_use': 0.8
            }
            return [config for i in range(num_clients)]
        
        
        return  (
            _init_dummy_model_weights(), 
            _init_trainConfigs(len(self.client_proxy_managers))
        )   

            
    
    def aggregate(self, 
                    trainRes_list: List[TrainRes]
                    )->Params:
            """
            Fed Avg algorithm for HFL
            
            Paramters
            ---------
            trainRes_list: List[TrainRes]
                A list of TrainRes, each corresponds to one client's model parameters and training metrics

            Returns
            -------
                Params: Parameters of global model 
             
            """

            empty_config={}
            return FedAvg(empty_config)(trainRes_list)
                    


    
    def train(self, num_rounds=3):
        """
        Start training process on selected clients. At each round, local model's parameter are aggregated by Server to 
        produce global models, which is then send back to and updated in each local model for next round training.
        
        
        Parameters
        ----------
        num_rounds:int
            number of training rounds
        """
        params, configs = self.init_globel_model()

        for rd in range(num_rounds):
            #TODO
            # Adding failure tolerence to allows aggregation on fewer returns of clients instead of every participanted clients.
            with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
                    futures = [
                                executor.submit(
                                    c_proxy.train,
                                    (TrainArgs(params, configs[idx]))
                                ) 
                                for idx,(_,c_proxy) in enumerate(self.client_proxy_managers.items())
                    ]
                    trainRes_All = [ f.result() for  f in futures]


            logger.info(f'--TRAINING {rd} Round Finished --')
            
            params:Params = self.aggregate(trainRes_All)

            #TODO evalution

        
        logger.info('== DONE TRAINING  ==')

        for remote_machine, client_proxy in self.client_proxy_managers.items():
            logger.info(f'Stop remote client {remote_machine.ip}:{remote_machine.port}')
            client_proxy.stop()
            
       
        
