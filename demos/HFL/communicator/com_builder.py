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

import sys,os
import numpy as np
import logging
from typing import Any,Dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(module) s - %(funcName) s - %(lineno) d - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


root_path = os.getcwd()

sys.path.append(root_path)
sys.path.append(os.path.join(root_path,'demos/HFL'))

from demos.HFL.communicator.com_utils import (
    GeneralCommunicator,
    MachineInfo_Wrapper
)

from demos.HFL.communicator.grpc_communicator import (
    GRPC_Sender,
    GRPC_Receiver,
    HFLMsg_CoreMsg_Converter,
    GRPC_Communicator
)

from demos.HFL.example.communicator.tornado_communicator import (
    TnMessage_Sender,
    TnMessage_Receiver,
    get_active_receiver,
    TnHFL_Message_Raw_Converter,
    Tornado_Communicator
)


class CommType():
   GRPC_COMM= 'grpc'
   MPI_COMM = 'mpi'
   TORNADO_COMM ='tornado'


class Communicator_Builder():
    """
    Factory to create Communicator instance by given its name string,
    To build user's own Communicator, implementation of following three classes are required :
        1. Message_Receiver:
            Responsible for receiving user-defined  message and forward it to its observer.
        2. Message_Sender:
            Allow user to send user-defined message to remote Communicator 
        3. HFL_Message_Raw_Converter:
            Convert between user-defined message and framework's <HFL_MSG>

    Example
    -------
    from HFL.communicator import (Communicator_Builder, CommType)
    # create A grpc Communicator instance
    comm = Communicator_Builder.make(CommType.GRPC_COMM)

    # Refer to <make_tornado> function as an example of how to create Three Classes and use them to assemble customized  Communicator. 

    """
    
    @staticmethod
    def make(
        comm_type:str = CommType.GRPC_COMM, 
        config:Dict[str,Any]=None):
        
        if comm_type  == CommType.GRPC_COMM:
            return Communicator_Builder.make_grpc(config)
        if comm_type  == CommType.TORNADO_COMM:
            return Communicator_Builder.make_tornado(config)    
        else: 
            raise(NotImplementedError(f'{comm_type} base Communicator is not implemented ...'))    


    @staticmethod
    def make_grpc(config)->GeneralCommunicator:
        sender = GRPC_Sender(
                            receiver_info=None
        )
        
        receiver = GRPC_Receiver(
                                config=config,
                                machine_info = \
                                    MachineInfo_Wrapper(
                                        config.ip,
                                        config.port
                                    )
        )                            
        msg_converter = HFLMsg_CoreMsg_Converter()    
        
       
        try:
             mode = config.mode if 'mode' in config else 'client'
        except:
             mode = config.mode if hasattr(config,'mode') else 'client'     
            
        return GRPC_Communicator(
                                    sender,
                                    receiver,
                                    msg_converter,
                                    mode=mode
        )

    
    @staticmethod
    def make_tornado(config)->GeneralCommunicator:
        sender = TnMessage_Sender(receiver_info=None)
        receiver:TnMessage_Receiver = get_active_receiver(config.port)
        msg_converter = TnHFL_Message_Raw_Converter()

        #mode = config.mode if 'mode' in config else 'client'
        try:
             mode = config.mode if 'mode' in config else 'client'
        except:
             mode = config.mode if hasattr(config,'mode') else 'client'     
            
        
        return  Tornado_Communicator(
                                    sender,
                                    receiver,
                                    msg_converter,
                                    mode=mode
        )