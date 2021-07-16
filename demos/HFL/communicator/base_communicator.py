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

import sys
import os

root_path = os.getcwd()
sys.path.append(root_path)

sys.path.append(os.path.join(root_path,'demos/HFL'))

from abc import abstractmethod

from demos.HFL.common.hfl_message import HFL_MSG
from demos.HFL.common.msg_handler import (Raw_Msg_Observer, Msg_Handler)


class BaseCommunicator(Raw_Msg_Observer):
    '''
    Abstract class is designed for bi-directional communication between client-to-client or client-to-server,
    Communicator is capable of  both receiving message and send message to/from
    other client/server.  Communicator will forward message to registered handler that process related
    message.
    '''

    @abstractmethod
    def send_message(self, msg: HFL_MSG):
        pass

    @abstractmethod
    def add_msg_handler(self, handler: Msg_Handler):
        pass

    @abstractmethod
    def remove_msg_handler(self, handler: Msg_Handler):
        pass
    
    @abstractmethod
    def receive_message(self, msg:HFL_MSG):
        ''' override parent Msg_Observer's func to receive HFL_MSG '''
        pass 

    @abstractmethod
    def run(self):
        '''start message  handling routine'''
        pass 

    def stop(self):
        '''stop message  handling routine'''