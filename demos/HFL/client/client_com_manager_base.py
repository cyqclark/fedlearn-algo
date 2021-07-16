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
import threading
import numpy as np 
import logging
import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(module) s - %(funcName) s - %(lineno) d - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


root_path = root = os.getcwd()
sys.path.append(os.path.join(root_path,'demos/HFL'))

from  abc import ABC, abstractmethod
from typing import Any, Callable

from core.entity.common.machineinfo import MachineInfo

from demos.HFL.common.msg_handler import Msg_Handler
from demos.HFL.base_client import Client
from demos.HFL.communicator.base_communicator import BaseCommunicator
from demos.HFL.common.hfl_message import HFL_MSG

from demos.HFL.basic_control_msg_type import HFL_Control_Massage_Type as CMT

from demos.HFL.common.param_util import(
    TrainArgs,
    TrainRes,
    NLPInferArgs,
    NLPInferRes
)

from demos.HFL.client.util import(
    trainArgs_to_msg,
    trainRes_to_msg,
    params_to_msg,
    msg_to_trainArgs,
    msg_to_trainRes,
    msg_to_modelParam,
    msg_to_NLPInferArgs,
    NLPInferArgs_to_msg,
    msg_to_NLPInferRes,
    NLPInferRes_to_msg,
)

class  ClientMode():
    CLIENT = 'client'
    PROXY  = 'server_side_proxy'



class ClientComManager(Msg_Handler):
    """
    ClientComManager unifies interface to simplify  management of clients.
    It provides control functions  and hide message communication details. 
    ClientComManager is responsible for message sending/receiving and mapping it to invocation of
    local client's function.
    
    ClientComManager has two modes: Proxy mode and Client mode. 
    Proxy Mode is to be invoke by task controller such as Server, function invocation to it is translated into message and then sends to remote client, 
    at client side, it is then mapped back to function invocation in ClientComManager that runs in Client mode. Pairing with Proxy mode, Client mode receives message 
    from one in Proxy mode and maps message to local function invocation (e.g. train/eval/predict), it returns results as message to one in Proxy mode.
    
    Parameters
    ----------
    comm : BaseCommunicator
           Instance of Communicator
    receiverInfo : MachineInfo
           Receiver's information  e.g. port and ip address
    mode : str
           Running mode, there are two modes: Proxy and Client
    client : Client
           Client isntance that control a local model, it is igored if mode is ClientMode.PROXY        
    """

    def __init__(self,  
                comm: BaseCommunicator,
                receiverInfo: MachineInfo,
                mode:str = ClientMode.PROXY,
                client:Client = None):

        self.Condition  = threading.Condition()
        self.inferCondition = threading.Condition()

        if mode ==   ClientMode.PROXY and client is not None: 
            raise(ValueError(f'{ClientMode.PROXY} must set client to be None'))

        
        self.msg2fuc_map = dict()
        self.client  = client
        self.mode= mode

        self.connected = False
        
        self.comm = comm
        self.comm.add_msg_handler(self)
        
        self.receiverInfo = receiverInfo
        
        self.trainRes  = None
        self.evalRes = None
        
        self.register_message_handlers()
        
        
    def run(self):
        if self.mode == ClientMode.CLIENT:
           try: 
            self.connect()
           except Exception as e: 
             logger.info(e)   
        self.comm.run()    
   

    
    def handle_message(self, msg_type, msg:HFL_MSG):
        """
        Handle HFL_MSG by invoking corresponding registered function
        
        Paramteres
        ----------
        msg_type: str
           message's type to identify corresponding processing funcs
        msg:HFL_MSG
           HFL_MSG received from remote client   
        """
        
        
        
        # Only handle msg from its corresponding remote client 
        # sender info in the HFL_MSG must match target receiver's info that was stored when proxy is created
        if msg_type in  self.msg2fuc_map and  \
           msg.sender.ip == self.receiverInfo.ip and\
           msg.sender.port == self.receiverInfo.port:
              logger.info(f'{self.receiverInfo.ip}:{self.receiverInfo.port} received HFL_MSG Type = {msg_type} : {msg.params[HFL_MSG.KEY_TYPE]}')
              self.msg2fuc_map[msg_type](msg)
        else:
            logger.debug(
                f"Message ignored due to msg_type : <{msg_type}> or sender's address {msg.sender.ip}:{msg.sender.port} \
                does not match receiver's address {self.receiverInfo.ip}:{self.receiverInfo.port}"
            )       
           
    
    
    
    def register_message_handlers(self):
        self.register_message_handler(
            CMT.CTRL_INIT_MODEL_S2C, 
            self.init_model)
        
        self.register_message_handler(
            CMT.CTRL_TRAIN_S2C, 
            self.handle_msg_train)
        
        self.register_message_handler(
           CMT.MSG_TRAIN_RES_C2S, 
           self.handle_msg_trainResponse)
        
        self.register_message_handler(
            CMT.CTRL_NLP_INFER_S2C,
            self.handle_msg_inference
        )

        self.register_message_handler(
            CMT.MSG_NLP_INFER_RES_C2S,
            self.handle_msg_inferenceRespose
        )

        self.register_message_handler(
            CMT.CTRL_CLIENT_STOP_S2C,
            self.handle_msg_stop
        )   
        


        # Do Not  handle Connecting HFL_MSG here, let server handle it so that it can create new ClineComManeger instead.
        
        #self.register_message_handler(
        #    CMT.CTRL_CLIENT_JOIN_C2S, 
        #    self.connect)
        
        # self.register_message_handler(
        #    CMT.MSG_CLIENT_JOIN_RES_S2C, 
        #    self.handle_msg_connected)
        
        # if ClientMode.CLIENT:
            
        #     self.register_message_handler(CMT.CTRL_TRAIN_S2C, self.connected_notice)
        
        # elif  ClientMode.PROXY:   

    def register_message_handler(self, type:str, func:Callable[[HFL_MSG],None]):
        self.msg2fuc_map[type] = func

    def init_model(self, msg:HFL_MSG):
        self.client.set_params(
            self._msg_to_trainParam(msg)
        )
    
    
    def connect(self):
        
        msg = HFL_MSG(
                    type = CMT.CTRL_CLIENT_JOIN_C2S,
                    sender = self.get_MachineInfo(),
                    receiver = self.receiverInfo
        )

        logger.info(f'{type(self).__name__ } connecting to server {self.receiverInfo.ip}:{self.receiverInfo.port}' )
        
       
        self.comm.send_message(msg)

    
    def disconnect(self):
        
        msg = HFL_MSG(
                    type = CMT.CTRL_CLIENT_DISJOIN_C2S,
                    sender = self.get_MachineInfo(),
                    receiver = self.receiverInfo
        )
        
        self.comm.send_message(msg)    


    def get_MachineInfo(self):
        return self.comm.get_MachineInfo()
    

    def handle_msg_train(self, msg:HFL_MSG):
        ''' Client Side: Received train command from server '''
        # Train local model
        logger.info(f' {self.receiverInfo.ip}:{self.receiverInfo.port} : run client training ...')
        trainRes = self.client.train(
            msg_to_trainArgs(msg)
        )

        # Send train response (model params and metrics) back to server
        self.comm.send_message(
            trainRes_to_msg(
                self.get_MachineInfo(), 
                self.receiverInfo, 
                trainRes)
        )
    
    
    def handle_msg_inference(self,msg:HFL_MSG):
        ''' Client Side: Received inference command from server '''
        logger.info(f' {self.receiverInfo.ip}:{self.receiverInfo.port} : run client inference ...')
        nlp_InferRes:NLPInferRes = self.client.inference(
            msg_to_NLPInferArgs(msg)
        )
        
        self.comm.send_message(
            NLPInferRes_to_msg(
                self.get_MachineInfo(), 
                self.receiverInfo, 
                nlp_InferRes)
        )

    def handle_msg_inferenceRespose(self,msg:HFL_MSG):
        '''Server side client proxy: Received Inference Response from Client'''
        nlp_InferRes = msg_to_NLPInferRes(msg)
        logger.info(f' {self.receiverInfo.ip}:{self.receiverInfo.port}  : received inference response ...')
        self._set_inferRes(
            nlp_InferRes
        )    



    def handle_msg_trainResponse(self, msg:HFL_MSG):
        '''Server side client proxy: 
        Received Train Response from Client'''
        
        trainRes = msg_to_trainRes(msg)
        logger.info(f' {self.receiverInfo.ip}:{self.receiverInfo.port}  : received train response ...')
        self._set_trainRes(
            trainRes
        )        

    def _set_trainRes(self, trainRes:TrainRes):
        '''Server side client proxy: 
        Reset the Condition to let the blocked thread to continue retrieving TrainRes'''
        with self.Condition:
          logger.debug(f' {self.receiverInfo.ip}:{self.receiverInfo.port} : notifying condition Obj...')     
          self.__trainRes = trainRes
          self.Condition.notify()

    
    def _get_trainRes(self)->TrainRes:
        '''Server Side Client proxy: 
        Block thread for trainRes to be ready after trainArgs is send to Client'''
        with self.Condition: 
            logger.info(f' {self.receiverInfo.ip}:{self.receiverInfo.port} : Wait for train to be done...')   
            self.Condition.wait()
            logger.info(f' {self.receiverInfo.ip}:{self.receiverInfo.port} : Train Done condition signal received ---getting result ...')  
            return self.__trainRes
    

    def _set_inferRes(self, inferRes:NLPInferRes):
        '''Server side client proxy: Reset the Condition to let blocked thread to continue retrieving InferRes'''
        with self.inferCondition:
          logger.info(f' {self.receiverInfo.ip}:{self.receiverInfo.port} : notifying condition Obj...')     
          self.__inferRes = inferRes
          self.inferCondition.notify()

    def _get_inferRes(self)->NLPInferRes:
        '''Server Side Client proxy: 
        After inferArgs is send to Client, block thread for inferRes to be ready '''
        with self.inferCondition: 
            logger.info(f' {self.receiverInfo.ip}:{self.receiverInfo.port} : Wait for inference to be done...')   
            self.inferCondition.wait()
            logger.info(f' {self.receiverInfo.ip}:{self.receiverInfo.port} : Condition signal received ---unbloccked... Inference done ...')  
            return self.__inferRes        

    def handle_msg_stop(self, msg:HFL_MSG):
        if self.mode == ClientMode.CLIENT:
            self.stop()


    def train(self, trainArgs:TrainArgs)->TrainRes:
        
        if self.mode == ClientMode.CLIENT:
            return self.client.train(trainArgs)
        
        elif self.mode == ClientMode.PROXY:
            # Send command to remote client and wait for result to return
            msg = trainArgs_to_msg(
                            sender =self.get_MachineInfo(),
                            receiver = self.receiverInfo,
                            trainArgs = trainArgs
                        )
            
           
            #msg.set_type(CMT.CTRL_TRAIN_S2C)         
            rec_msg = self.comm.send_message(msg)

            logger.info(f'Start train on remote <{self.receiverInfo.ip}:{self.receiverInfo.port}>: type = {msg.type}')

            logger.info(f' : Init from {msg.sender.ip}:{msg.sender.port}  Proxy')          
            
            trainRes =self._get_trainRes()
            
            disp_metrics = {k:f"{v:.3f}" if isinstance(v,float) else v for k,v in trainRes.metrics.items()}
            logger.info(f'Client {self.receiverInfo.ip}:{self.receiverInfo.port} train matrics : {disp_metrics}')
            return trainRes

    def stop(self)->None:
        if self.mode == ClientMode.CLIENT:
            self.comm.stop()
        elif self.mode == ClientMode.PROXY:
            msg = HFL_MSG(type=CMT.CTRL_CLIENT_STOP_S2C,
                        sender=self.get_MachineInfo(),
                        receiver=self.receiverInfo
                        ) 
            self.comm.send_message(msg)
            logger.info(f'Stop remote <{self.receiverInfo.ip}:{self.receiverInfo.port}>: type = {msg.type}')                      

    
    def inference(self, inputArgs:NLPInferArgs)->Any:
        if self.mode == ClientMode.CLIENT:
            return self.client.inference(inputArgs)
        
        elif self.mdoe == ClientMode.PROXY:
            # send command to remote client and wait for result to return
            msg = NLPInferArgs_to_msg(
                            sender = self.get_MachineInfo(),
                            receiver = self.receiverInfo,
                            nlp_inferAgrs = inputArgs
                        )
            
           
            #msg.set_type(CMT.CTRL_TRAIN_S2C)          
            rec_msg = self.comm.send_message(msg)

            logger.info(f' : {self.receiverInfo.ip}:{self.receiverInfo.port}  start predicting {self.receiverInfo.ip}:{self.receiverInfo.port}: type = {msg.type}')       
            
            inferRes =self._get_inferRes()
           
            return inferRes



    def handle_msg_evaluate(self, msg:HFL_MSG):
        if self.client and self.mode == ClientMode.CLIENT:
            self.client.eval(
                 self._msg_to_evalArgs(msg)
            )    
