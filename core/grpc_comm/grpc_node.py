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

from core.entity.common.message import RequestMessage, ResponseMessage
from core.grpc_comm.grpc_converter import common_msg_to_grpc_msg, grpc_msg_to_common_msg
from core.proto.transmission_pb2_grpc import TransmissionServicer, add_TransmissionServicer_to_server
from core.proto.transmission_pb2_grpc import TransmissionStub
from concurrent import futures
import grpc

# global variables
_MAX_MESSAGE_LENGTH = 1 << 30
# max size is: 2147483648-1 bytes (i.e., 1024*1024*1024-1 bytes, or 2GB)
_MAX_INT = 2147483647
_GRPC_OPTIONS = [('grpc.max_message_length', _MAX_INT),
                 ('grpc.max_receive_message_length', _MAX_INT)]


class GRPCNode(object):
    """gRPC node.

    Based on the unary communication in gRPC framework, this class packs the message sending and receiving.

    Attributes
    ----------

    Methods
    -------
    send_request(message: RequestMessage)
        Static method. Active sending RequestMessage and receiving ResponseMessage.
    start_serve_termination_block(grpc_servicer: TransmissionServicer)
        Open a gRPC service by blocking termination, and monitor gRPC request and return gRPC response.
    start_serve(grpc_servicer: TransmissionServicer)
        Open gRPC service as a server, and monitor gRPC request and return gRPC response.
    stop_serve()
        Stop a gRPC service.
    """
    def __init__(self, machine_info=None):
        self.machine_info = machine_info
        self.__server = None
        self.is_serve_running = False

    def set_machine_info(self, machine_info):
        self.machine_info = machine_info

    # input is RequestMessage
    @staticmethod
    def send_request(common_req_msg: RequestMessage) -> ResponseMessage:
        """ Send RequestMessage and receive ResponseMessage.

        Active sending RequestMessage and receiving ResponseMessage.

        Parameters
        ----------
        common_req_msg : RequestMessage
            request message created by alg in the sending terminal.

        Returns
        -------
        ResponseMessage
            the feedback of response results in the ResponseMessage from gRPC server.
            If it is asynchronous, the response only needs to confirm that the message has been received.
        """
        # unary connection: send a request and receive a response
        with grpc.insecure_channel("%s:%s" % (common_req_msg.client_info.ip, common_req_msg.client_info.port),
                                   options=_GRPC_OPTIONS) as channel:
            stub = TransmissionStub(channel)
            grpc_res_msg = stub.comm(common_msg_to_grpc_msg(common_req_msg))

        return grpc_msg_to_common_msg(grpc_res_msg, comm_req_res=1)

    def start_serve_termination_block(self, grpc_servicer: TransmissionServicer):
        """ gRPC server.

        Open a gRPC service by blocking termination, and monitor gRPC request and return gRPC response.
        This function should be called by the federated terminals which are receiving messages.

        Parameters
        ----------
        grpc_servicer : class derived from transmission_pb2_grpc.TransmissionServicer
            request message created by alg in the sending terminal.

        Returns
        -------
        None
            This function has no return variables.
        """
        options = [
            ('grpc.max_send_message_length', _MAX_MESSAGE_LENGTH),
            ('grpc.max_receive_message_length', _MAX_MESSAGE_LENGTH),
        ]
        self.__server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=options)
        add_TransmissionServicer_to_server(grpc_servicer, self.__server)
        self.__server.add_insecure_port("%s:%s" % (self.machine_info.ip, self.machine_info.port))
        # TODO: using debug level (DebugOutput), not print, to display information
        print("---------------")
        print("starting %s:%s" % (self.machine_info.ip, self.machine_info.port))
        # 
        self.__server.start()
        self.is_serve_running = True
        self.__server.wait_for_termination()

    def start_serve(self, grpc_servicer: TransmissionServicer):
        """ gRPC server.

        Open gRPC service as a server, and monitor gRPC request and return gRPC response.

        Parameters
        ----------
        grpc_servicer : class derived from transmission_pb2_grpc.TransmissionServicer
            request message created by alg in master terminal.

        Returns
        -------
        None
            This function has no return variables.
        """
        options = [
            ('grpc.max_send_message_length', _MAX_MESSAGE_LENGTH),
            ('grpc.max_receive_message_length', _MAX_MESSAGE_LENGTH),
        ]
        self.__server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=options)
        add_TransmissionServicer_to_server(grpc_servicer, self.__server)
        self.__server.add_insecure_port("%s:%s" % (self.machine_info.ip, self.machine_info.port))
        # TODO: using debug level (DebugOutput), not print, to display information
        print("---------------")
        print("starting %s:%s" % (self.machine_info.ip, self.machine_info.port))
        # 
        self.__server.start()
        self.is_serve_running = True

    def stop_serve(self):
        """ stop gRPC server.

        Stop a gRPC service.

        Parameters
        ----------
        None

        Returns
        -------
        None
            This function has no return variables.
        """
        if self.__server != None and self.is_serve_running == True:    
            self.__server.stop(None)
            self.is_serve_running = False


send_request = GRPCNode.send_request
