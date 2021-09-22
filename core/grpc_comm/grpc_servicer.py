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

from core.grpc_comm.grpc_converter import grpc_msg_to_common_msg, common_msg_to_grpc_msg
from core.proto.transmission_pb2 import ReqResMessage
from core.proto.transmission_pb2_grpc import TransmissionServicer
from typing import Any


class GRPCServicer(TransmissionServicer):
    def __init__(self, msg_processor: Any = None):
        # set observer for msg processing.
        print("init grpc servicer")
        print("type of observer: %s" % str(type(msg_processor)))
        self.msg_processor = msg_processor

    def set_msg_processor(self, msg_processor):
        # reset observer for msg processing.
        print("reset grpc servicer")
        print("type of observer: %s" % str(type(msg_processor)))
        self.msg_processor = msg_processor

    def comm(self, grpc_request: ReqResMessage, context) -> ReqResMessage:
        # print("running com", grpc_request, context)
        common_req_msg = grpc_msg_to_common_msg(grpc_request)
        common_res_msg = self.msg_processor.process_request(common_req_msg)
        return common_msg_to_grpc_msg(common_res_msg)

    # sender_ip = self._parse_sender_ip(context.peer())
    # common_res_msg = self.msg_processor.process_request(common_req_msg, sender_ip=sender_ip)
    # def _parse_sender_ip(self, context):
    #     try:
    #         _, ip_addr, _ = context.peer().split(':')    # 'ipv4:49.123.106.100:44420'
    #     except:  
    #         _, ip_addr, _ = context.split(':')     
    #     ip_addr =ip_addr.strip()
    #     return ip_addr
