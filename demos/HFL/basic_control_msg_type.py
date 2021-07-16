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

class HFL_Control_Massage_Type():
    """
    Define the command string  for control and communication between server and client
    """

    CTRL_INIT_MODEL_S2C = 'init_model'

    CTRL_TRAIN_S2C = 'train'
    CTRL_EVALUTE_S2C = 'evaluate_local'

    MSG_TRAIN_RES_C2S = 'train_response'
    MSG_RECEIVED_C2S = 'msg_received'

    CTRL_CLIENT_JOIN_C2S = 'client_connected'
    CTRL_CLIENT_DISJOIN_C2S = 'client_disconnected'
    MSG_CLIENT_JOIN_RES_S2C = 'new_client_connecting'

    CTRL_NLP_INFER_S2C = 'nlp_inference_representation'
    MSG_NLP_INFER_RES_C2S = 'nlp_inference_representation_response'

    CTRL_CLIENT_STOP_S2C = 'client_stop'


