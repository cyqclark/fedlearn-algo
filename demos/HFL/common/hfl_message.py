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
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(module) s - %(funcName) s - %(lineno) d - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

root_path = os.getcwd()
logger.info(f'PATH = {root_path}')

sys.path.append(root_path)
sys.path.append(os.path.join(root_path, 'demos/HFL'))

from abc import abstractmethod
import json

from core.entity.common.message import (
    RequestMessage,
    ResponseMessage,
    MachineInfo
)

import json

from io import BytesIO
import numpy as np
from typing import cast, Tuple, List, Dict


def numpy_to_bytes(ndarray) -> bytes:
    bytes_io = BytesIO()
    np.save(bytes_io, ndarray, allow_pickle=False)
    return bytes_io.getvalue()


def bytes_to_numpy(bdata: bytes) -> np.ndarray:
    bytes_io = BytesIO(bdata)
    ndarray_deserialized = np.load(bytes_io, allow_pickle=False)

    return cast(np.ndarray, ndarray_deserialized)


class HFL_MSG():
    """
    Basic Message used by HFL framework to communicate between clients and server.

    Attribute
    ---------
    type: str
        message type define in HFL_Control_Massage_Type, e.g. train/eval/init
    """

    KEY_TYPE = "type_of_message"
    KEY_SENDER_IP = "sender_ip"
    KEY_SENDER_PORT = "sender_port"
    KEY_SENDER_TOKEN = "sender_token"

    KEY_RECEIVER_IP = "receiver_ip"
    KEY_RECEIVER_PORT = "receiver_port"
    KEY_RECEIVER_TOKEN = "receiver_token"

    CONTROL_SEND = "send"
    CONTROL_RECEIVE = "receive"

    KEY_PARAMS_WEIGHTS = 'model_weights'
    KEY_PARAMS_NAMES = 'model_names'
    KEY_CONFIG_TRAIN = 'train_config'

    KEY_METRICS = 'metrics'

    KEY_MODEL_PARAMS = "model_params"
    KEY_NUM_TRAIN_SAMPLES = "num_train_samples"

    KEY_INFER_SENTENCES = 'nlp_inference_sentences'
    KEY_INFER_SENTENCES_REP = 'nlp_inference_sentences_response'

    def __init__(self, type: str, sender: MachineInfo, receiver: MachineInfo):
        self.type = type
        self.sender = sender
        self.receiver = receiver

        # To be compatible with core/commen/message
        if sender.token is None:
            sender.token = 'dummy_1'
        if self.receiver is None:
            receiver.token = 'dummy_2'

        self.params = {}
        self.params[HFL_MSG.KEY_TYPE] = type
        self.params[HFL_MSG.KEY_SENDER_PORT] = sender.port
        self.params[HFL_MSG.KEY_SENDER_IP] = sender.ip

        self.params[HFL_MSG.KEY_RECEIVER_PORT] = receiver.port
        self.params[HFL_MSG.KEY_RECEIVER_IP] = receiver.ip

    def get_sender(self) -> MachineInfo:
        return self.sender

    def get_receiver(self) -> MachineInfo:
        return self.receiver

    def add(self, p_name, p_value):
        self.params[p_name] = p_value

    def get(self, p_name):
        return self.params[p_name]

    def get_type(self):
        return self.params[HFL_MSG.KEY_TYPE]

    def set_type(self, msg_type: str):
        self.type = msg_type
        self.params[HFL_MSG.KEY_TYPE] = msg_type

    def numpy_weight_to_list(self) -> Dict[str, List[List[float]]]:
        """
        Convert model weights of numpy array to list
        """
        names = []
        params = []
        if HFL_MSG.KEY_PARAMS_WEIGHTS not in self.params:
            return None, None

        for key, value in self.params[HFL_MSG.KEY_PARAMS_WEIGHTS].items():
            params.append(value.tolist())
            names.append(key)
        return dict(zip(names, params))

    def list_to_numpy_weight(self, data: Dict[str, List[List[float]]]) -> Dict[str, np.ndarray]:
        """
        Convert model weights of list to numpy array
        """
        return {name: np.array(lvalue) for name, lvalue in data.items()}

    def numpy_weight_to_bytes(self) -> Dict[str, bytes]:
        names = []
        params = []
        if HFL_MSG.KEY_PARAMS_WEIGHTS not in self.params:
            return None, None

        for key, value in self.params[HFL_MSG.KEY_PARAMS_WEIGHTS].items():
            params.append(numpy_to_bytes(value))
            names.append(key)

        return dict(zip(names, params))

    def bytes_to_numpy_weight(self,
                              data: Dict[str, bytes]
                              ) -> Dict[str, np.ndarray]:
        return {name: bytes_to_numpy(dvalue) for name, dvalue in data.items()}


class MsgConverter():
    """
    Convert between core's RequestMessage and HFL_MSG
    """

    @staticmethod
    def coreMsg2HFLMsg(rmsg: RequestMessage) -> HFL_MSG:

        if not isinstance(rmsg.server_info, MachineInfo):
            rmsg.server_info = MachineInfo(
                ip=rmsg.body[HFL_MSG.KEY_SENDER_IP],
                port=rmsg.body[HFL_MSG.KEY_SENDER_PORT])

        msg_dict = rmsg.body
        hfl_msg = HFL_MSG(
            type=msg_dict[HFL_MSG.KEY_TYPE],
            sender=rmsg.server_info,  # server is remote machine
            receiver=rmsg.client_info)  # client is local machine

        if HFL_MSG.KEY_PARAMS_NAMES in msg_dict \
                and HFL_MSG.KEY_PARAMS_WEIGHTS in msg_dict:
            weight_bytes = \
                dict(
                    zip(msg_dict[HFL_MSG.KEY_PARAMS_NAMES],
                        msg_dict[HFL_MSG.KEY_PARAMS_WEIGHTS])
                )

            hfl_msg.add(
                HFL_MSG.KEY_PARAMS_WEIGHTS,
                hfl_msg.bytes_to_numpy_weight(weight_bytes)
            )

        if HFL_MSG.KEY_CONFIG_TRAIN in msg_dict:
            hfl_msg.add(HFL_MSG.KEY_CONFIG_TRAIN,
                        {
                            k: v
                            for k, v in json.loads(
                            msg_dict[HFL_MSG.KEY_CONFIG_TRAIN]
                        ).items()
                        }
                        )

        if HFL_MSG.KEY_METRICS in msg_dict:
            hfl_msg.add(HFL_MSG.KEY_METRICS,
                        {
                            k: v
                            for k, v in json.loads(
                            msg_dict[HFL_MSG.KEY_METRICS]
                        ).items()
                        }
                        )

        for k, v in msg_dict.items():
            if k not in [HFL_MSG.KEY_PARAMS_WEIGHTS,
                         HFL_MSG.KEY_CONFIG_TRAIN,
                         HFL_MSG.KEY_METRICS]:
                hfl_msg.add(k, v)

        return hfl_msg

    @staticmethod
    def HFLMsg2CoreMsg(msg: HFL_MSG,
                       symbol: str = 'HFL') -> RequestMessage:

        def fill_token(ms: HFL_MSG):
            if msg.sender.token is None:
                msg.sender.token = 'dummy_s'
            if msg.receiver.token is None:
                msg.receiver.token = 'dummy_r'
            return msg

        body = {}
        body[HFL_MSG.KEY_TYPE] = msg.type

        if HFL_MSG.KEY_PARAMS_WEIGHTS in msg.params:
            weights_dict = msg.numpy_weight_to_bytes()
            body[HFL_MSG.KEY_PARAMS_WEIGHTS]:List[bytes] = list(weights_dict.values())
            body[HFL_MSG.KEY_PARAMS_NAMES]:List[str] = list(weights_dict.keys())

        if HFL_MSG.KEY_CONFIG_TRAIN in msg.params:
            for k, v in msg.params[HFL_MSG.KEY_CONFIG_TRAIN].items():
                msg.params[HFL_MSG.KEY_CONFIG_TRAIN][k] = v

            body[HFL_MSG.KEY_CONFIG_TRAIN] = \
                json.dumps(msg.params[HFL_MSG.KEY_CONFIG_TRAIN])

        if HFL_MSG.KEY_METRICS in msg.params:
            for k, v in msg.params[HFL_MSG.KEY_METRICS].items():
                msg.params[HFL_MSG.KEY_METRICS][k] = v

            body[HFL_MSG.KEY_METRICS] = \
                json.dumps(msg.params[HFL_MSG.KEY_METRICS])

        for k, v in msg.params.items():
            if k not in [
                HFL_MSG.KEY_PARAMS_WEIGHTS,
                HFL_MSG.KEY_CONFIG_TRAIN,
                HFL_MSG.KEY_METRICS]:
                body[k] = v

        # Add senderInfo in case it is not processed by lower level API:
        # add local machine info (sender)
        body[HFL_MSG.KEY_SENDER_IP] = msg.sender.ip
        body[HFL_MSG.KEY_SENDER_PORT] = msg.sender.port

        logger.info(
            f'Prepared MSG <{body[HFL_MSG.KEY_TYPE]}> to send to  {msg.receiver.ip}:{msg.receiver.port} from {msg.sender.ip}:{msg.sender.port}')

        msg = fill_token(msg)  # ensuring  token is not empty to avoid crashing problem while converting

        req_msg = RequestMessage(
            sender=msg.sender,  # ->client_info # grpc sender channel
            receiver=msg.receiver,  # ->server_info
            body=body,
            phase_id=symbol
        )

        return req_msg

