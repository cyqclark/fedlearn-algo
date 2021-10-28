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
from core.grpc_comm.grpc_node import GRPCNode
from core.grpc_comm.grpc_servicer import GRPCServicer
import copy
import queue
import threading
import json
import time
lock = threading.Lock()
import heapq
from abc import ABC, abstractmethod


class Cache(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def get(self):
        pass

    @abstractmethod
    def put(self, msg):
        pass

    @abstractmethod
    def size(self):
        pass


class MQ(Cache):

    def __init__(self):
        super().__init__()
        self.cache = queue.Queue()

    def put(self, msg):
        self.cache.put(msg)

    def get(self):
        return self.cache.get()

    def size(self):
        return self.cache.qsize()


class PQ(Cache):

    def __init__(self):
        super().__init__()
        self.cache = []

    def put(self, msg):
        phase_id = msg.phase_id
        if phase_id.startswith('layer'):
            idx = int(phase_id.split('_')[1])
        elif phase_id == 'finish':
            idx = 1000
        elif phase_id == 'init_comput_graph':
            idx = -1000
        else:
            print(f"phase {phase_id} is not supported yet!")
        heapq.heappush(self.cache, (-idx, time.time(), msg) ) # python uses min queue by default

    def get(self):
        # print(self.cache)
        _, _, msg = heapq.heappop(self.cache)
        return msg

    def size(self):
        return len(self.cache)


class AsyncGRPCCommunicator(object):
    def __init__(self, common_req_msg_processor):
        # set observer
        self.common_req_msg_processor = common_req_msg_processor

        # node
        self.grpc_servicer = GRPCServicer(self)
        self.grpc_node = GRPCNode(common_req_msg_processor.machine_info)
        self.grpc_node.start_serve(self.grpc_servicer)

        # init queue
        self.common_req_msg_q = PQ()

    def process_request(self, common_req_msg):
        sender_info = common_req_msg.server_info
        receiver_info = common_req_msg.client_info

        if common_req_msg.phase_id == "init_comput_graph":
            resp_msg = ResponseMessage(
                sender = sender_info,
                receiver = receiver_info,
                body =  {'graph': json.dumps(self.common_req_msg_processor.sync_server.compute_graph)},
                phase_id = common_req_msg.phase_id + "_req_received")
            return resp_msg

        lock.acquire()
        self.common_req_msg_q.put(common_req_msg)
        lock.release()
        common_res_msg = ResponseMessage(sender=receiver_info,
                                         receiver=sender_info,
                                         body={}, # TODO
                                         phase_id=common_req_msg.phase_id+"_req_received")
        return common_res_msg

    def stop_grpc_node_receive_routine(self):
        self.grpc_node.stop_serve()

    def send_message(self, common_req_msg):
        return GRPCNode.send_request(common_req_msg)

    def start_grpc_message_processing(self):
        thread = threading.Thread(target=self.grpc_message_processing)
        thread.start()

    def grpc_message_processing(self):
        while self.grpc_node.is_serve_running:
            time.sleep(0.0001) # check every 0.1 ms
            if self.common_req_msg_q.size() > 0:
                lock.acquire()

                common_req_msg = self.common_req_msg_q.get()

                lock.release()

                common_res_msg = self.common_req_msg_processor.process_queue(common_req_msg)

                if common_res_msg.phase_id == "finish":
                    print("image id: ", common_res_msg.body['_id'], "finished secure inference", " similarity distance: ", common_res_msg.body['dist'], " prediction: ", common_res_msg.body['pred'], " time stamp: ", time.time())
                    # break
                else:
                    common_req_msg_2 = RequestMessage(sender=common_res_msg.server_info,
                                                      receiver=common_res_msg.client_info,
                                                      body=common_res_msg.body,
                                                      phase_id=common_res_msg.phase_id)
                    GRPCNode.send_request(common_req_msg_2)
        return
