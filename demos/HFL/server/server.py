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
from threading import Condition, Lock, Thread

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

_NUM_ROUNDS_= 10 #10
class AlgoArch():
   SYNC = 'sync'
   ASYNC = 'async'

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
    
    CSTATUS_INIT = 'INIT' 
    CSTATUS_LOCALREADY = 'LOCALREADY'
    CSTATUS_GLOBALREADY = 'GLOBALREADY'

    def __init__(self, 
                 port:int=8890, 
                 num_clients_to_train:int=2,
                 comm_type=CommType.TORNADO_COMM,
                 comm_instance:GeneralCommunicator=None,
                 clients_ip_port:List[str]=None,
                 algo_arch=AlgoArch.SYNC):                
        
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
        
        self.aggregation_type = algo_arch
        logger.info(f'Current algorithm architecture {self.aggregation_type}.') 
        if self.aggregation_type == AlgoArch.ASYNC:
            self._p = 0
            self.lock = Lock()
            self.aggr_event = Condition(self.lock)
            self.n_aggr = 0
            #self.aggr_event = Condition()
            logger.info(f'\nInit aggr_event : {self.aggr_event}') 
            #self.train_events = [Condition(self.lock) for _ in range(num_clients_to_train)]
            self.train_events = [Condition() for _ in range(num_clients_to_train)]
            logger.info(f'\nInit train_events: {self.train_events}') 

            self.modelDB = {}
            self.gModel = None
            self.numSamples = {}
            self.client_status = {}
            for client_id in range(self.num_clients_to_train):
                self.client_status[client_id] = Server.CSTATUS_INIT
                self.numSamples[client_id] = 0
        
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
    
    def async_aggregate(self, client_id):
        with self.aggr_event:
            self.n_aggr += 1 
            logger.info(f'async_aggr client#{client_id} time#{self.n_aggr}')
            if self.gModel is None:
                self.gModel = self.modelDB[client_id]['local'].params
                return
 
            logger.info(f'client#{client_id} time#{self.n_aggr}: {self.modelDB[client_id].keys()}')
            total_samples = sum(self.numSamples.values())
            logger.info(f'client#{client_id} time#{self.n_aggr}: total_sample={total_samples}')
            trainRes_list = [
                    self.modelDB[client_id]['local'],
                    TrainRes(self.gModel, total_samples)]
            #logger.info(f'aggr {trainRes_list}')
            for i, trainRes in enumerate(trainRes_list):
                #logger.info(f'params ###{i}###')
                w_names = trainRes.params.names
                #logger.info(f'params {w_names}')
                num_samples = trainRes.num_samples 
                logger.info(f'client#{client_id} time#{self.n_aggr}: sub_samples={num_samples}')
                weights_nums = len(trainRes.params.weights)
                #logger.info(f'params {weights_nums}')
                metrics = trainRes.metrics
                logger.info(f'client#{client_id} time#{self.n_aggr}: metrics={metrics}')
    
            self.gModel = self.aggregate(trainRes_list)
            self.modelDB[client_id]['global'] = self.gModel
        return

    def aggr_thread(self, num_rounds, client_id):
        logger.info('--Thread: aggregation  --')
        for rd in range(num_rounds):
            #with self.aggr_event:
            with self.train_events[client_id]:
                #logger.info(f'aggr, {self.train_events[client_id]}  wait... when iter#{self._p}')
                #self.aggr_event.wait()
                self.train_events[client_id].wait_for(lambda: self.client_status[client_id] == Server.CSTATUS_LOCALREADY)
                #logger.info(f'aggr, client#{client_id}, in local #{rd} epoch, global {self._p} epoch, activate, {self.train_events}')
                logger.info(f'aggr, client#{client_id}, in local #{rd} epoch, global {self._p} epoch, activate')
                #logger.info(f'aggr, {self.client_status}')
                if self.client_status[client_id] == Server.CSTATUS_LOCALREADY:
                        self.async_aggregate(client_id)
                        #self.modelDB[client_id]['global'] = self.modelDB[client_id]['local']
                        self.client_status[client_id] = Server.CSTATUS_GLOBALREADY
                        #logger.info(f'iter#{self._p} notify+ client#{client_id}, {self.train_events[client_id]}, {self.client_status[client_id]}')
                        self.train_events[client_id].notify()
                        #logger.info(f'iter#{self._p} notify- client#{client_id}, {self.train_events[client_id]}, {self.client_status[client_id]}')

    def train_thread(self, num_rounds, client_id, c_proxy, params, config):
        logger.info(f'--Thread: train for {client_id} --')
        for rd in range(num_rounds):
            #logger.info(f'client#{client_id} prepare train in #{rd} epoch.')
            with self.train_events[client_id]:
                #logger.info(f'client#{client_id} get train lock in local #{rd} epoch, global {self._p} epoch.')
                #if self.client_status[client_id] != Server.CSTATUS_INIT: 
                if self.client_status[client_id] == Server.CSTATUS_LOCALREADY: 
                    #logger.info(f'client#{client_id} wait {self.train_events[client_id]} in local #{rd} epoch, global {self._p} epoch...')
                    self.train_events[client_id].wait_for(lambda: self.client_status[client_id] == Server.CSTATUS_GLOBALREADY)
                    #use global params as local params
                    #params = self.modelDB[client_id]['global'].params
                    params = self.gModel
                logger.info(f'>>>>>>>>>>>>>>>>>')
                logger.info(f'client#{client_id} activate, start to train local model, in local #{rd} epoch, global {self._p} epoch')
                #logger.info(f'client{client_id} iter#{rd}')
                #logger.info(f'client{client_id} {self.client_status[client_id]}')
                #logger.info(f'client{client_id} {args}')
                #trainRes = c_proxy.train(args)
                trainRes = c_proxy.train(TrainArgs(params, config))
                #trainRes = self.dummy_test(client_id, rd)
                logger.info(f'!!!>>>>>>>> client#{client_id} train done, in local #{rd} epoch, global {self._p} epoch')
                self.modelDB[client_id]['local'] = trainRes#.params
                self.numSamples[client_id] = trainRes.num_samples
                self.client_status[client_id] = Server.CSTATUS_LOCALREADY
                #logger.info(f'client{client_id} {trainRes}')
                self._p += 1
                logger.info(f'client#{client_id}, metrics={trainRes.metrics}, in local #{rd} epoch, global {self._p} epoch')
                logger.info(f'<<<<<<<<<<<<<<<<<')
                #logger.info(f'client#{client_id} notify+ aggr {self.aggr_event}, {self.train_events}, in local #{rd} epoch, global {self._p} epoch')
                #self.aggr_event.notify()
                self.train_events[client_id].notify()
                #logger.info(f'client#{client_id} notify- aggr {self.aggr_event}, {self.train_events}, in local #{rd} epoch, global {self._p} epoch')
    
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

        logger.info('\t========== START TRAINING ===========')
        logger.info(f'aggregation_type {self.aggregation_type}')
        logger.info(f'AlgoArch.ASYNC {AlgoArch.ASYNC}')
        if self.aggregation_type == AlgoArch.ASYNC:
            #logger.info(f'{self.client_proxy_managers}')
            logger.info('== START Async TRAINING  ==')
            threads = []
            #threads.append(Thread(target=self.aggr_thread, args=(num_rounds,)))
            #threads[-1].start()

            #for client_id in range(self.num_clients_to_train):
            for idx,(_,c_proxy) in enumerate(self.client_proxy_managers.items()):
                #threads.append(Thread(target=self.train_thread, args=(num_rounds,idx, c_proxy, TrainArgs(params, configs[idx]))))
                threads.append(Thread(target=self.aggr_thread, args=(num_rounds,idx, )))
                threads.append(Thread(target=self.train_thread, args=(num_rounds,idx, c_proxy, params, configs[idx])))
                #threads[-1].start()

                client_id = idx
                self.modelDB[client_id] = {}

            for thread in threads:
                thread.start()
            for thread in threads:
                """Waits for the threads to complete before moving on
                   with the main script.
                """
                thread.join()
        else:
            logger.info('== START Sync TRAINING  ==')
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
            
       
        
