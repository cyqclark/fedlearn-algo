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

"""
Create  GRPC_Communicator
"""
import sys,os
import numpy as np
import logging
from typing import Any,Dict
import threading

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(module) s - %(funcName) s - %(lineno) d - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

root_path = os.getcwd()

sys.path.append(root_path)
sys.path.append(os.path.join(root_path,'demos/HFL'))

from demos.HFL.common.hfl_message import (HFL_MSG, MsgConverter)
from demos.HFL.common.msg_handler import (Msg_Handler, Raw_Msg_Observer)
from core.entity.common.message import (RequestMessage, ResponseMessage)
#from demos.HFL.common.base_communicator import BaseCommunicator
from core.client.client import Client
from core.proto.transmission_pb2_grpc import TransmissionServicer, add_TransmissionServicer_to_server
from core.entity.common.message import (RequestMessage, ResponseMessage)
from core.proto.transmission_pb2 import ReqResMessage
from core.grpc_comm.grpc_converter import grpc_msg_to_common_msg, common_msg_to_grpc_msg

from core.grpc_comm.grpc_node import (GRPCNode, send_request)

from demos.HFL.communicator.com_utils import (
    MachineInfo_Wrapper,
    Message_Sender,
    HFL_Message_Raw_Converter,
    Message_Receiver,
    GeneralCommunicator
)


# from core.proto.transmission_pb2_grpc import TransmissionServicer, 
from concurrent import futures
import grpc




# global variables
_MAX_MESSAGE_LENGTH = 1 << 30
# class grpc_server():
    # def  serve(self, grpc_servicer: TransmissionServicer) -> None:
        # options = [
        # ('grpc.max_send_message_length', _MAX_MESSAGE_LENGTH),
        # ('grpc.max_receive_message_length', _MAX_MESSAGE_LENGTH),
        # ]
        # self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=50), options=options)
        # add_TransmissionServicer_to_server(grpc_servicer, self.server)
        # self.server.add_insecure_port("%s:%s" % (grpc_servicer.machine_info.ip, grpc_servicer.machine_info.port))
        # logger.info("starting %s:%s" % (grpc_servicer.machine_info.ip, grpc_servicer.machine_info.port))
        # self.server.start()
        # self.server.wait_for_termination()
    # def stop(self):
        # self.server.stop(1)    

class GRPC_Sender(Message_Sender):
    def send(self, data:RequestMessage):
        return send_request(data)


class HFLMsg_CoreMsg_Converter(HFL_Message_Raw_Converter):
    """ Wrapper for the specific converter of existing code """
    def raw2HFLMsg(self,rawMsg:Any)->HFL_MSG:
        return MsgConverter.coreMsg2HFLMsg(rawMsg)

    
    def HFLMsg2raw(self,msg:HFL_MSG)->Any:
        return MsgConverter.HFLMsg2CoreMsg(msg)
        


class GRPC_Receiver(Message_Receiver, TransmissionServicer):    
    def __init__(self,
                config,
                machine_info:MachineInfo_Wrapper,
                msg_observer:Raw_Msg_Observer=None):
            
            Message_Receiver.__init__(self,config,msg_observer)
            TransmissionServicer.__init__(self)
            self.machine_info = machine_info.get_CoreMachineInfo()
    
    def process_request(self, 
                        request: RequestMessage
                        ) -> ResponseMessage:
        '''Overide this func of Parent <TransmissionServicer> class.
           this function will receive information routed from TransmissionServicer (grpc)
           and then forward to <receiving_msg> which is to be processed by 
           Observer
        '''
        return self.receiving_msg(request)
     
    
    def comm(self, grpc_request: ReqResMessage, context) -> ReqResMessage:
        '''Override comm func to obtain sender's ip address'''
        
        common_req_msg = grpc_msg_to_common_msg(grpc_request)
        common_req_msg.server_info.ip=self._parse_sender_ip(context.peer())
        logger.info(f'REMOTE IP address from Comm = {common_req_msg.server_info.ip}')
        
        common_res_msg = self.process_request(common_req_msg)
        return common_msg_to_grpc_msg(common_res_msg) 

    def _parse_sender_ip(self, context):
        try:
            _,ip_addr,_ = context.peer().split(':')    # 'ipv4:49.123.106.100:44420'
        except:  
             _,ip_addr,_ = context.split(':')     
        ip_addr =ip_addr.strip()
        return ip_addr          

class GRPC_Communicator(GeneralCommunicator):
    def start_message_receiving_routine(self):
         self.grpc_server = GRPCNode(self.receiver.machine_info)
         self.grpc_server.start_serve(self.receiver)
        
        
    
    def stop(self):
        super(GRPC_Communicator, self).stop()
        self.grpc_server.stop_serve()
       

