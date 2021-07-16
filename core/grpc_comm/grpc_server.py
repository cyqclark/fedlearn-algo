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

from core.proto.transmission_pb2_grpc import TransmissionServicer, add_TransmissionServicer_to_server
from concurrent import futures
import grpc

# global variables
_MAX_MESSAGE_LENGTH = 1 << 30


def serve(grpc_servicer: TransmissionServicer) -> None:
    """ gRPC server.

    Open gRPC service as a server, and monitor gRPC request and return gRPC response.
    This function should be called by the federated clients.

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
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=options)
    add_TransmissionServicer_to_server(grpc_servicer, server)
    server.add_insecure_port("%s:%s" % (grpc_servicer.machine_info.ip, grpc_servicer.machine_info.port))
    # TODO: using debug level (DebugOutput), not print, to display information
    print("---------------")
    print("starting %s:%s" % (grpc_servicer.machine_info.ip, grpc_servicer.machine_info.port))
    server.start()
    server.wait_for_termination()
