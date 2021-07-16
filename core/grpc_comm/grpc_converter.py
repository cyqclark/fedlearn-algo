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

from core.entity.common.machineinfo import MachineInfo
from core.entity.common.message import RequestMessage, ResponseMessage
from core.proto.transmission_pb2 import Vector, Matrix, ReqResMessage
from enum import Enum, unique
import numpy as np
from typing import Dict


@unique
class MsgListElmType(Enum):
    empty_t = -1
    strings_t = 0
    ints_t = 1
    matrices_t = 2
    vectors_t = 3
    values_t = 4
    bytes_t = 5


def determine_elm_type(first_elm) -> int:
    """ Find the type of list elm.

    Support string, int, transmission_pb2.Matrix, transmission_pb2.Vector, float, bytes.

    Parameters
    ----------
    first_elm : str, int, one- or two-dimensional numpy array, float, bytes
        the first element of list (list is not empty).

    Returns
    -------
    int
        an enum value (its value, an integrate) about the element type.
    """
    # determine the element type of one dict value
    if isinstance(first_elm, str):
        return MsgListElmType.strings_t.value
    elif isinstance(first_elm, (int, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
        return MsgListElmType.ints_t.value
    elif isinstance(first_elm, np.ndarray) and len(first_elm.shape) == 2:
        return MsgListElmType.matrices_t.value
    elif isinstance(first_elm, np.ndarray) and len(first_elm.shape) == 1:
        return MsgListElmType.vectors_t.value
    elif isinstance(first_elm, (float, np.float16, np.float32, np.float64)):
        return MsgListElmType.values_t.value
    elif isinstance(first_elm, bytes):
        return MsgListElmType.bytes_t.value
    else:
        raise ValueError("Currently, grpc message does not support this elm type: %s" % type(first_elm))


# convert proto types to numpy n-dimensional array
# vector to n-dimensional array
def vector_to_numpy_array(vector: Vector) -> np.array:
    """ transmission_pb2.Vector to numpy.array.

    A function to convert a transmission_pb2.Vector to numpy.array.

    Parameters
    ----------
    vector : transmission_pb2.Vector
        a gRPC vector object.

    Returns
    -------
    numpy.array
        a vector of numpy.array.
    """
    if isinstance(vector, Vector):
        return np.array(vector.values)
    else:
        raise ValueError("Not a vector!")


# matrix to n-dimensional array
def matrix_to_numpy_array(matrix: Matrix) -> np.array:
    """ transmission_pb2.Matrix to numpy.array.

    A function to convert a transmission_pb2.Matrix to numpy.array.

    Parameters
    ----------
    matrix : transmission_pb2.Matrix
        a gRPC matrix object.

    Returns
    -------
    numpy.array
        a matrix of numpy.array.
    """
    if isinstance(matrix, Matrix):
        return np.array([i_row.values for i_row in matrix.rows])
    else:
        raise ValueError("Not a matrix!")


def common_dict_msg_to_arrays(msg_body_dict: Dict):
    """ convert msg body to a dict buffer and a dict description.

    A function to reorganize the buffer used in the input msg body, 
    and record each element's location (buffer indices).
    A common msg dict to arrays for proto building, e.g., matrices, vectors and so on.

    Parameters
    ----------
    msg_body_dict : dictionary
        alg defined msg body.

    Returns
    -------
    temp_dict_buffs : dictionary
        dict buffer.
    temp_dict_notes : dictionary
        dict description.
    """
    # buffs
    temp_dict_buffs = dict()
    for v in MsgListElmType:
        temp_dict_buffs[v.value] = []
    # notes
    temp_dict_notes = dict()
    # indices
    temp_record_indices = dict()
    for v in MsgListElmType:
        temp_record_indices[v.value] = 0

    # check None
    if msg_body_dict is None:
        return temp_dict_buffs, temp_dict_notes

    # parse common msg body (dict)
    for key, value in msg_body_dict.items():
        # single or list element
        if isinstance(value, list):
            is_list_value = True
            temp_value_tolist = value
        else:
            is_list_value = False
            temp_value_tolist = [value]
        #
        val_elm_amount = len(temp_value_tolist)
        if val_elm_amount == 0:
            buff_type = MsgListElmType.empty_t.value
        else:
            buff_type = determine_elm_type(temp_value_tolist[0])
        # according to the buff_type, assign one dict value to its matched buff 
        for i in range(val_elm_amount):
            temp_dict_buffs[buff_type].append(temp_value_tolist[i])
        # record temporary indices
        buff_start_ind = temp_record_indices[buff_type]
        temp_record_indices[buff_type] = temp_record_indices[buff_type] + val_elm_amount
        # record all
        temp_dict_notes[key] = [is_list_value, buff_type, buff_start_ind, val_elm_amount]

    return temp_dict_buffs, temp_dict_notes


def create_grpc_message(sender: MachineInfo,
                        receiver: MachineInfo,
                        dict_buffs: Dict,
                        dict_notes: Dict,
                        phase_num: str) -> ReqResMessage:
    """ create grpc message.

    Create grpc messages for both input and output.

    Parameters
    ----------
    sender : MachineInfo
        sender's ip, port and token information.
    receiver : MachineInfo
        receiver's ip, port and token information.
    dict_buffs : dictionary
        dict buffer.
    dict_notes : dictionary
        dict description.
    phase_num : str
        algorithm's phase.

    Returns
    -------
    temp_dict_buffs : transmission_pb2.ReqResMessage
        grpc message.
    """
    # init grpc msg
    grpc_message = ReqResMessage()

    # assign sender and receiver's machine info
    grpc_message.source_machine_info.ip = sender.ip
    grpc_message.source_machine_info.port = sender.port
    grpc_message.source_machine_info.token = sender.token
    grpc_message.target_machine_info.ip = receiver.ip
    grpc_message.target_machine_info.port = receiver.port
    grpc_message.target_machine_info.token = receiver.token
    # assign phase num
    grpc_message.phase_num = phase_num

    # assign multi_bytes
    multi_bytes = dict_buffs[MsgListElmType.bytes_t.value]
    grpc_message.multi_bytes[:] = multi_bytes
    # assign matrices
    matrices = dict_buffs[MsgListElmType.matrices_t.value]
    for mxi in matrices:
        matrix = grpc_message.matrices.add()
        for i_row in mxi:
            row = matrix.rows.add()
            row.values[:] = i_row.tolist()
    # assign vectors
    vectors = dict_buffs[MsgListElmType.vectors_t.value]
    for i_vec in vectors:
        vector = grpc_message.vectors.add()
        vector.values[:] = i_vec.tolist()
    # assign values
    values = dict_buffs[MsgListElmType.values_t.value]
    if isinstance(values, np.ndarray):
        grpc_message.values[:] = values.tolist()
    else:
        grpc_message.values[:] = values
    # assign strings and ints
    grpc_message.strings[:] = dict_buffs[MsgListElmType.strings_t.value]
    grpc_message.ints[:] = dict_buffs[MsgListElmType.ints_t.value]

    # TODO: note still needs 0, ..., 3 to index the content. Need to further modify the grpc proto format.
    for key, note in dict_notes.items():
        grpc_message.dict_notes[key].is_list_value = note[0]
        grpc_message.dict_notes[key].buff_type = note[1]
        grpc_message.dict_notes[key].buff_start_ind = note[2]
        grpc_message.dict_notes[key].val_elm_amount = note[3]

    return grpc_message


def parse_grpc_message(grpc_message: ReqResMessage):
    """ parse grpc message.

    A function to parse grpc message to msg notes, bytes, phase_num and other variables.

    Parameters
    ----------
    grpc_message : transmission_pb2.ReqResMessage, 
        grpc message.

    Returns
    -------
    sender : MachineInfo
        sender's ip, port and token information.
    receiver : MachineInfo
        receiver's ip, port and token information.
    dict_buffs : dictionary
        dict buffer.
    dict_notes : dictionary
        dict description.
    phase_num : str
        algorithm's phase.    
    """
    # unpack sender and receiver's machine info
    sender = MachineInfo(ip=grpc_message.source_machine_info.ip, 
                         port=grpc_message.source_machine_info.port, 
                         token=grpc_message.source_machine_info.token)
    receiver = MachineInfo(ip=grpc_message.target_machine_info.ip, 
                           port=grpc_message.target_machine_info.port, 
                           token=grpc_message.target_machine_info.token)
    # unpack phase_num
    phase_num = grpc_message.phase_num

    # buffs
    dict_buffs = dict()
    for v in MsgListElmType:
        dict_buffs[v.value] = []
    # unpack strings
    dict_buffs[MsgListElmType.strings_t.value] = list(grpc_message.strings)
    # unpack ints
    dict_buffs[MsgListElmType.ints_t.value] = list(grpc_message.ints)
    # unpack multi_bytes
    for i_bt in grpc_message.multi_bytes:
        dict_buffs[MsgListElmType.bytes_t.value].append(i_bt)
    # unpack matrices
    for i_mx in grpc_message.matrices:
        dict_buffs[MsgListElmType.matrices_t.value].append(matrix_to_numpy_array(i_mx))
    # unpack vectors
    for i_vec in grpc_message.vectors:
        dict_buffs[MsgListElmType.vectors_t.value].append(vector_to_numpy_array(i_vec))
    # unpack values
    dict_buffs[MsgListElmType.values_t.value] = list(grpc_message.values)

    # unpack dict_notes
    dict_notes = dict()
    for key in grpc_message.dict_notes.keys():
        dict_notes[key] = [grpc_message.dict_notes[key].is_list_value,
                           grpc_message.dict_notes[key].buff_type,
                           grpc_message.dict_notes[key].buff_start_ind,
                           grpc_message.dict_notes[key].val_elm_amount]

    return sender, receiver, phase_num, dict_buffs, dict_notes


def arrays_to_common_dict_msg(dict_buffs: Dict, dict_notes: Dict) -> Dict:
    """ convert ResponseMessage to grpc response message.

    A function to convert buffer and records to a common msg dict.

    Parameters
    ----------
    dict_buffs : dictionary
        dict buffers.
    dict_notes : dictionary
        dict description.

    Returns
    -------
    msg_body_dict : dictionary
        common msg body, same as the algorithm's input.
    """
    msg_body_dict = dict()
    for key, note in dict_notes.items():
        # TODO: note still needs 0, ..., 3 to index the content. Need to further modify the grpc proto format.
        is_list_value = note[0]
        buff_type = note[1]
        buff_start_ind = note[2]
        val_elm_amount = note[3]
        if MsgListElmType(buff_type) == MsgListElmType.empty_t:
            msg_body_dict[key] = []
        else:
            if is_list_value:
                msg_body_dict[key] = \
                    dict_buffs[buff_type][buff_start_ind:buff_start_ind+val_elm_amount]
            else:
                msg_body_dict[key] = dict_buffs[buff_type][buff_start_ind]
    return msg_body_dict


def common_msg_to_grpc_msg(common_msg) -> ReqResMessage:
    """ convert ResponseMessage to grpc response message.

    A function to convert algorithm's ResponseMessage to grpc message which could be sent.

    Parameters
    ----------
    common_msg : RequestMessage or ResponseMessage
        algorithm's common message from algorithms.

    Returns
    -------
    transmission_pb2.ReqResMessage
        grpc message.
    """
    dict_buffs, dict_notes = common_dict_msg_to_arrays(common_msg.body)
    if isinstance(common_msg, RequestMessage):
        sender = common_msg.server_info
        receiver = common_msg.client_info
    else:
        sender = common_msg.client_info
        receiver = common_msg.server_info
    return create_grpc_message(sender, receiver, dict_buffs, dict_notes, common_msg.phase_id)


def grpc_msg_to_common_msg(grpc_msg: ReqResMessage, comm_req_res: int = 0):
    """ convert grpc request message to RequestMessage or ResponseMessage.

    A function to convert grpc's message to RequestMessage or ResponseMessage.

    Parameters
    ----------
    grpc_msg : grpc message, ReqResMessage
        grpc message sent from grpc massage.
    comm_req_res : int
        identify the type of output common message, is RequestMessage or ResponseMessage

    Returns
    -------
    RequestMessage or ResponseMessage
        common request message for alg.
    """
    sender, receiver, phase_num, dict_buffs, dict_notes = parse_grpc_message(grpc_msg)
    msg_body = arrays_to_common_dict_msg(dict_buffs, dict_notes)
    if comm_req_res == 0:
        return RequestMessage(sender, receiver, msg_body, phase_num)
    else:
        return ResponseMessage(sender, receiver, msg_body, phase_num)
