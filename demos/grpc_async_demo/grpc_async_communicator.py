from core.entity.common.message import RequestMessage, ResponseMessage
from core.grpc_comm.grpc_node import GRPCNode
from core.grpc_comm.grpc_servicer import GRPCServicer
import copy
import queue
import threading

lock = threading.Lock()


class AsyncGRPCCommunicator(object):
    def __init__(self, common_req_msg_processor, machine_info):
        # set observer
        self.common_req_msg_processor = common_req_msg_processor

        # node
        self.grpc_servicer = GRPCServicer(self)
        self.grpc_node = GRPCNode(machine_info)
        self.grpc_node.start_serve(self.grpc_servicer)

        # init queue
        self.common_req_msg_q = queue.Queue()

    def process_request(self, common_req_msg, sender_ip=""):
        lock.acquire()
        self.common_req_msg_q.put([common_req_msg, sender_ip])
        lock.release()

        sender_info = common_req_msg.server_info
        receiver_info = common_req_msg.client_info
        common_res_msg = ResponseMessage(sender=sender_info,
                                         receiver=receiver_info,
                                         body={},
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
            if self.common_req_msg_q.qsize() > 0:
                lock.acquire()

                common_req_msg, sender_ip = self.common_req_msg_q.get()

                lock.release()

                print("")
                print(self.__class__.__name__)
                print("receiving msg, %s, %s, %s" % (str(common_req_msg.server_info), 
                                                     str(common_req_msg.client_info),
                                                     str(common_req_msg.phase_id)))
                print("")
                common_res_msg = self.common_req_msg_processor.process_queue(common_req_msg)
                
                if common_res_msg.phase_id == "finish":
                    print("get a finish phase, break the message receiving loop")
                    break
                else:
                    common_req_msg_2 = RequestMessage(sender=common_res_msg.server_info,
                                                      receiver=common_res_msg.client_info,
                                                      body=common_res_msg.body,
                                                      phase_id=common_res_msg.phase_id)
                    GRPCNode.send_request(common_req_msg_2)
        return
