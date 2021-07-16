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

import sys, os
import socket
import threading
from dataclasses import dataclass
root_path = os.getcwd()

sys.path.append(root_path)
sys.path.append(os.path.join(root_path,'demos/HFL'))
import sys 

from demos.HFL.communicator.base_communicator import BaseCommunicator
from core.entity.common.machineinfo import MachineInfo
from demos.HFL.common.hfl_message import HFL_MSG
from demos.HFL.basic_control_msg_type import HFL_Control_Massage_Type as HCT
from demos.HFL.common.msg_handler import Msg_Handler
import time

from multiprocessing  import Process


from demos.HFL.communicator.com_builder import (
    Communicator_Builder,
    CommType
)

import unittest   # The test framework


Condition = threading.Condition()
@dataclass
class Config:
    ip:str
    port:str
    mode:str

class TestCommunicator(unittest.TestCase):
    def test_grpc_communicator(self):
        class Receiver(Msg_Handler):
            def __init__(self,comm:BaseCommunicator):
                self.comm = comm

            def handle_message(self,msg_type, msg_data:HFL_MSG):
                msg_data.sender,msg_data.receiver = msg_data.receiver,msg_data.sender
                self.comm.send_message(msg_data)
                '''DO nothing'''


        class Sender(Msg_Handler):
            def __init__(self,comm:BaseCommunicator):
                self.comm = comm
                self.return_type = ''

            def handle_message(self, msg_type, msg_data:HFL_MSG):
                self.return_type = msg_data.get_type()
                with Condition:
                    Condition.notify()
            
            def get_return(self)->str:
                with Condition:
                   Condition.wait()
                return self.return_type
            
            def send(self,msg_type_str):
                sender_info = MachineInfo(ip=local_ip, port=sender_port,token='dummy1')
                receiver_info = MachineInfo(ip=local_ip, port=receiver_port,token='dummy2')
                msg_1 = HFL_MSG(
                                type=msg_type_str,
                                sender=sender_info,
                                receiver=receiver_info)

                               
                self.comm.send_message(msg_1)
                return self.get_return()                

        local_ip = socket.gethostbyname(socket.gethostname())
        sender_port = '9991'
        receiver_port = '9909'

        
        config_1 =Config(
            ip=local_ip,
            port=sender_port,
            mode='proxy')

        comm_1 =  Communicator_Builder.make(
                    comm_type=CommType.GRPC_COMM, 
                    config=config_1)
       
        config_2 =Config(
            ip=local_ip,
            port=receiver_port,
            mode='client')

        comm_2 = Communicator_Builder.make(
                    comm_type= CommType.GRPC_COMM,
                    config = config_2)


        sender = Sender(comm_1)
        comm_1.add_msg_handler(sender)
        
        receiver = Receiver(comm_2)
        comm_2.add_msg_handler(receiver)
        
        thread = threading.Thread(target=comm_2.run)
        thread.start()
        

        thread = threading.Thread(target=comm_1.run)
        thread.start()
        time.sleep(1)
        
        
        send_msg_type_str = 'dummy'
        ret_msg_type_str = sender.send(send_msg_type_str)
        
        self.assertEqual(send_msg_type_str, ret_msg_type_str)
        comm_1.stop()
        comm_2.stop()


if __name__ == '__main__':
    unittest.main()
        



