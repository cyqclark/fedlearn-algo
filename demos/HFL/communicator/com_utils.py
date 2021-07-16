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
from typing import Any, Optional, Union, Dict, List
from abc import abstractmethod,ABC


root_path = os.getcwd()

sys.path.append(root_path)
sys.path.append(os.path.join(root_path,'demos/HFL'))

from demos.HFL.common.hfl_message import HFL_MSG

from demos.HFL.common.msg_handler import (Msg_Handler, Raw_Msg_Observer)
from demos.HFL.communicator.base_communicator import BaseCommunicator

from core.entity.common.machineinfo import MachineInfo

import queue
import threading
lock = threading.Lock()

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(module) s - %(funcName) s - %(lineno) d - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AttributeDict(dict):
    __slots__ = () 
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


class Com_Machine_Info(ABC):
    def get_id(self)->str:
        '''return machine identification '''


class HFL_Message_Raw_Converter():
    """
    Convert between HFL_MSG  and low-level communication specific data/message format,
    The HFL framework allow users to build their own communication methods such as grpc, http, socket or MPI etc.
    To replace framework provided communication methods with customized ones, user need to implement <raw2HFLMsg> and <HFLMsg2raw>
    functions to convert between HFL_MSG and their own data/message used in lower-level user build communication methods.
    """
    @abstractmethod
    def raw2HFLMsg(self,rawMsg:Any)->HFL_MSG:
        '''
        Convert raw message of base communication to HFL message
        
        Parameters:
        ----------
            rawMsg: raw message of base communication
        
        Return:
        _______
            HFL_MSG: converted HFL message
               
        '''
        pass 

    @abstractmethod
    def HFLMsg2raw(self,msg:HFL_MSG)->Any:
        '''
        Convert HFL message to message of base communication 
        
        Parameters:
        ----------
            msg: HFL message
        
        Return:
        ----------
            message of base communication
               
        '''
        pass



class Message_Receiver(object):
    def __init__(self, 
                config,
                msg_observer:Raw_Msg_Observer=None):
        ''' 
        Message_Receiver are suppose to work as dispatcher, which runs on thread or new process to receive message send from remote machine(s), 
        and then forward to observer if it is given.       
        
        Parameters:
        ----------
          config :Dict[Union[str,str,float,int]], receiver's configuration
          msg_observer: Observer that gets notified when raw message is received
           '''
        self.config  = config
        self.msg_observer: Raw_Msg_Observer  = msg_observer

    def set_msg_observer(self,
                         observer:Raw_Msg_Observer,
                        )->None:        
        '''
        Set observer that will receive raw message forwarded from <Message_Receiver>
        
        Parameters:
        ----------      
        observer: Observer that gets notified when raw message is received
       
        Return:
        ----------
           None  
        '''
        self.msg_observer = observer        
   

    def receiving_msg(self, data:Any)->Any:
        '''

        Parameters:
        ---------- 
        data: raw message received 

        Return:
        ----------
        raw message
        '''
        if self.msg_observer:
           return self.msg_observer.receive_message(data)
               
    
    @abstractmethod
    def start(self):
        pass
    
    @abstractmethod
    def stop(self):
        pass        


# Response = Union[str,bytes]

class Message_Sender():
    """ Message_Sender send message/data to remote machine via <send> func """
    def __init__(self,
                receiver_info:Com_Machine_Info):
        self.receiver_info =  receiver_info
    
    def get_receiver_info(self)->Com_Machine_Info:
        return self.receiver_info
    
    @abstractmethod    
    def send(self, data:Any)->Any:
        pass


class GeneralCommunicator(BaseCommunicator):
    
    def __init__(self,
                sender:Message_Sender,
                receiver:Message_Receiver,
                msg_converter:HFL_Message_Raw_Converter,
                mode ='client'):
        self.mode = mode
        self.sender = sender
        self.receiver = receiver
        self.msg_converter = msg_converter
        self.receiver.set_msg_observer(self)
        
        self._msg_handlers = []
        self.msg_queue = queue.Queue()
        self.is_running = False
        

    def receive_message(self, data: Any) -> Any:
        ''' Convert receiver's Raw message into HFL_MSG and put to processing queque '''
        msg:HFL_MSG = self.msg_converter.raw2HFLMsg(data)

        logger.info(f'{type(self).__name__} receiverd msg :Type= {msg.type}')
        
        lock.acquire()
        self.msg_queue.put(msg)
        lock.release()

        resp_msg =  HFL_MSG(HFL_MSG.CONTROL_RECEIVE,
                            msg.sender, 
                            msg.receiver)

        raw_resp_data = self.msg_converter.HFLMsg2raw(resp_msg)                    
        logger.info(f'{type(self).__name__} port:{msg.receiver.port}   put msg into queque:  Type= {msg.type}')                 
        return raw_resp_data

    #@abstractmethod
    def run(self):
        # # 1. start gprc service that reive msg and stor in quque
        # thread = threading.Thread(target=grpc_server.serve, args=(self.receiver,))
        # thread.start()
        self.start_message_receiving_routine()
        
        
        # #2. start main message routine that retreive HFL_MSG and routine to corresponding processing fuction
        self.is_running = True
        self.msg_handling_routine()

    @abstractmethod    
    def start_message_receiving_routine(self):
        pass    


    def stop(self):
        self.is_running = False

    def msg_handling_routine(self):
        ''' 
        Start Message routine, once msg received forward to upstream handler,
        Note that it is critical NOT to start a new thread at client side to avoid "slow CUDA GPU training"
        '''
        while self.is_running:
            if self.msg_queue.qsize() > 0:

                lock.acquire()
                msg : HFL_MSG = self.msg_queue.get()
                lock.release()
                msg_type = msg.get_type()
                
                logger.info(f'{type(self).__name__} in Mode:{self.mode} msg routine handling ==========>>: Type = {msg_type}')
                
                for handler in self._msg_handlers:
                    logger.info(f'{type(self).__name__} : Forward msg to {type(handler)}----->>:type={msg_type}')
                    # Critical: set mode to "client" to avoid starting thread for GPU training, which cause very slow training!
                    if self.mode =='client':
                        handler.handle_message(msg_type, msg)
                    elif self.mode=='proxy':    
                        threading.Thread(target=handler.handle_message, args=(msg_type, msg,)).start()
                    else: 
                        raise(ValueError(f'Mode {self.mode} is not a valid mode'))            
        return 

    def add_msg_handler(self, handler:Msg_Handler)->None:
        self._msg_handlers.append(handler)

    def send_message(self, msg:HFL_MSG)->HFL_MSG:
        res_msg = self.msg_converter.HFLMsg2raw(msg)
        raw_msg:Any = \
            self.sender.send(res_msg)
           
        return self.msg_converter.raw2HFLMsg(raw_msg)

    def remove_msg_handler(self, handler: Msg_Handler):
        try:
            self._msg_handlers.remove(handler)
        except Exception as e:
            logger.info(f'{type(self)} error {e}')
    
    def get_MachineInfo(self)->MachineInfo:
        return self.receiver.machine_info


class MachineInfo_Wrapper(Com_Machine_Info):
    def __init__(self, ip, port, token='dummy_w'):
        self.core_MachineInfo = MachineInfo(ip,port,token)
    
    def get_id(self)->str:
        rep_str = f'{self.core_MachineInfo.ip}:{self.core_MachineInfo.port}'
        return rep_str
    
    def __repr__(self) -> str:
        return self.get_id()

    def get_CoreMachineInfo(self):
        return self.core_MachineInfo        

