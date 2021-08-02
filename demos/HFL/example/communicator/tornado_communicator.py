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
import pickle
import tornado.web
import tornado.ioloop
import tornado.httpserver
#from tornado.options import define, options

from typing import List, Any, Dict
import argparse
import json
import logging
import os
import requests
import socket
import sys
import threading

root_path = os.getcwd()
#_LOCALHOST = socket.gethostbyname(socket.gethostname())
_LOCALHOST = socket.gethostname()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(module) s - %(funcName) s - %(lineno) d - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sys.path.append(root_path)
sys.path.append(os.path.join(root_path,'demos/HFL'))

from demos.HFL.communicator.com_utils import(
    MachineInfo_Wrapper,
    Message_Receiver,
    Message_Sender,
    HFL_Message_Raw_Converter,
    GeneralCommunicator
)

from demos.HFL.common.msg_handler import (Msg_Handler, Raw_Msg_Observer)

from demos.HFL.common.hfl_message import HFL_MSG
from core.entity.common.machineinfo import MachineInfo


__RECEIVER__ = [None]

class TnMessage_Receiver(Message_Receiver):
    def __init__(self,
                config,
                machine_info:MachineInfo_Wrapper,
                msg_observer:Raw_Msg_Observer=None):
            
            super().__init__(config,msg_observer)
            self.machine_info = machine_info.get_CoreMachineInfo()

    def start(self):
        app = tornado.web.Application(
        [
            (r"/", Tonado_Handler)
        ],
        )
        server = tornado.httpserver.HTTPServer(app, max_buffer_size=2485760000)  # 2G
        server.listen(self.machine_info.port)

        #app.listen(self.machine_info.port)
        thread = threading.Thread(target=tornado.ioloop.IOLoop.current().start)
        thread.start()



def get_active_receiver(port=8810):
    if __RECEIVER__[0] is None:
        __RECEIVER__[0] = TnMessage_Receiver(
            config={'port', port},
            machine_info=MachineInfo_Wrapper(ip=_LOCALHOST,port=port)
        )
    return __RECEIVER__[0]


class Tonado_Handler(tornado.web.RequestHandler):
    def post(self):
        remote_ip = self.request.remote_ip
        raw_data = pickle.loads(self.request.body)
        raw_data[HFL_MSG.KEY_SENDER_IP] = remote_ip
        logger.info(type(raw_data))
        __RECEIVER__[0].receiving_msg(raw_data)
        self.write({'response':'msg received'})


class TestObserver(Raw_Msg_Observer):
    def receive_message(self, msg_data:Any) -> Any:
        logger.info(f'{type(self)} received data ... ')


class TnMessage_Sender(Message_Sender):
    
    def send(self, data:Dict[str,Any])->Any:
        
        if self.receiver_info is not None: 
            receiver_ip = self.receiver_info.ip
            receiver_port = self.receiver_info.port
        else: 
            receiver_ip = data[HFL_MSG.KEY_RECEIVER_IP]
            receiver_port = data[HFL_MSG.KEY_RECEIVER_PORT]

        api_endpoint = f"http://{receiver_ip}:{receiver_port}"
        res = requests.post(
            api_endpoint,
            data= pickle.dumps(data)
        )
        return data


class TnHFL_Message_Raw_Converter(HFL_Message_Raw_Converter):

    def raw2HFLMsg(self,rawMsg:Any)->HFL_MSG:
       
        msg_dict = rawMsg

        server_info = MachineInfo(
            ip=msg_dict[HFL_MSG.KEY_SENDER_IP],
            port=msg_dict[HFL_MSG.KEY_SENDER_PORT],
            token='dummy_token')
        
        # receiver_info suppose to be local machine, but since Converter 
        # doesn't have local machine_info, we set it to be the same as server_info temperally 
        receiver_info = server_info  
       
        hfl_msg = HFL_MSG(
                            msg_dict[HFL_MSG.KEY_TYPE], 
                            server_info, 
                            receiver_info)
        
        for k,v in msg_dict.items():
            hfl_msg.params[k]=v
               
        return hfl_msg
        
   

    def HFLMsg2raw(self,msg:HFL_MSG)->Any:
        
        body ={}
        body[HFL_MSG.KEY_TYPE] = msg.type
        for k,v in msg.params.items():
            body[k] = v
        
        
        # Add senderInfo in case it is not processed by lower level API:
        body[HFL_MSG.KEY_SENDER_IP] = msg.sender.ip
        body[HFL_MSG.KEY_SENDER_PORT] = msg.sender.port

        return body
      


class Tornado_Communicator(GeneralCommunicator):
    def start_message_receiving_routine(self):
        '''already started '''
        self.receiver.start()
       

def test(args):
    if args.mode == 0:
        sender = TnMessage_Sender(receiver_info=None)
        data = {'layer_{i}':np.zeros((1000,1000)).tolist() for i in range(24)}
        data[HFL_MSG.KEY_RECEIVER_IP] = _LOCALHOST
        data[HFL_MSG.KEY_RECEIVER_PORT] = str(8890)
        sender.send(data)
    else:
        __RECEIVER__[0].set_msg_observer(TestObserver())
        __RECEIVER__[0].start()




if __name__ == "__main__":
    parser = argparse.ArgumentParser()       
    parser.add_argument('--mode', help = 'mode set to client proxy', type=str, default='client')
    args = parser.parse_args()
    test(args)
    
    
