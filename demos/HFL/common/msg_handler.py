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

import os,sys
import numpy as np
from typing import Any

root_path = os.getcwd()
sys.path.append(root_path)

sys.path.append(os.path.join(root_path,'demos/HFL'))

from abc import ABC, abstractmethod

from demos.HFL.common.hfl_message import HFL_MSG

class Raw_Msg_Observer(ABC):
    @abstractmethod
    def receive_message(self, msg_data:Any) -> Any:
        pass

class Msg_Handler(ABC):
    @abstractmethod
    def handle_message(self, msg_type, msg_data:HFL_MSG) -> None:
        pass