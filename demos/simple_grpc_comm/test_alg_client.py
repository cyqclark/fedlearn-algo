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

# add core's path
import os
import sys
sys.path.append(os.getcwd())
#
from core.entity.common.message import ResponseMessage
from core.entity.common.machineinfo import MachineInfo
from core.grpc_comm.grpc_converter import grpc_msg_to_common_msg, common_msg_to_grpc_msg
from core.grpc_comm.grpc_server import serve
from core.proto.transmission_pb2 import ReqResMessage
from core.proto.transmission_pb2_grpc import TransmissionServicer
import argparse
import socket

try:
    _LOCALHOST = socket.gethostbyname(socket.gethostname())
except socket.gaierror:
    # if failed to fetch 127.0.0.1, try 0.0.0.0
    _LOCALHOST = socket.gethostbyname("")


class ClientTest(TransmissionServicer):
    def __init__(self, machine_info: MachineInfo):
        super().__init__()
        self.machine_info = machine_info

    def comm(self, grpc_req_msg: ReqResMessage, context) -> ReqResMessage:
        common_req_msg = grpc_msg_to_common_msg(grpc_req_msg)

        # message processing simulation
        # this demo has no process_request

        common_res_msg = ResponseMessage(sender=common_req_msg.client_info, 
                                         receiver=common_req_msg.server_info,
                                         body=common_req_msg.body,
                                         phase_id=common_req_msg.phase_id+"_"+self.machine_info.token)

        return common_msg_to_grpc_msg(common_res_msg)


if __name__ == '__main__':
    # sending arguments
    parser = argparse.ArgumentParser()
    # machine info
    parser.add_argument('-I', '--ip', type=str, help="string of server ip, exp:%s" % _LOCALHOST)
    parser.add_argument('-P', '--port', type=str, help="string of server port, exp: 8890")
    parser.add_argument('-T', '--token', type=str, help="string of server name, exp: client_1")
    #
    args = parser.parse_args()
    # TODO
    print("ip: %s" % args.ip)
    print("port: %s" % args.port)
    print("token: %s" % args.token)
    #
    client_info = MachineInfo(ip=args.ip, port=args.port, token=args.token)
    client = ClientTest(client_info)
    #
    serve(client)
