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

from core.grpc_comm.grpc_converter import common_msg_to_grpc_msg, grpc_msg_to_common_msg
from core.proto.transmission_pb2_grpc import TransmissionStub
import grpc

# global variables
# max size is: 2147483648-1 bytes (i.e., 1024*1024*1024-1 bytes, or 2GB)
_MAX_INT = 2147483647
_GRPC_OPTIONS = [('grpc.max_message_length', _MAX_INT),
                 ('grpc.max_receive_message_length', _MAX_INT)]


class GRPCClient(object):
    """gRPC Client.

    This class packs the gRPC unary communication, and it also converts the input and output messages.

    Attributes
    ----------

    Methods
    -------
    send_request(message: RequestMessage)
        Static method. Active sending RequestMessage and receiving ResponseMessage.
    """

    # input is RequestMessage
    @staticmethod
    def send_request(common_req_msg):
        """ Send RequestMessage and receive ResponseMessage.

        This function should be called by federated master terminal.
        Active sending RequestMessage and receiving ResponseMessage.

        Parameters
        ----------
        common_req_msg : RequestMessage
            request message created by alg in master terminal.

        Returns
        -------
        ResponseMessage
            the feedback of response results in the ResponseMessage from gRPC server.
        """
        # unary connection: send a request and receive a response
        with grpc.insecure_channel("%s:%s" % (common_req_msg.client_info.ip, common_req_msg.client_info.port),
                                   options=_GRPC_OPTIONS) as channel:
            stub = TransmissionStub(channel)
            grpc_res_msg = stub.comm(common_msg_to_grpc_msg(common_req_msg))

        return grpc_msg_to_common_msg(grpc_res_msg, comm_req_res=1)


send_request = GRPCClient.send_request
