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

"""
Test script
"""
import logging
import os
import sys
import socket
sys.path.append(os.getcwd()) # this might cause some issues
import socket
from core.entity.common import machineinfo

import unittest   # The test framework

ip = socket.gethostbyname(socket.gethostname())

class Test_TestMachineinfo(unittest.TestCase):
    def test_instance_create(self):
        client = machineinfo.MachineInfo(ip, "8890", "123")
        self.assertEqual(str(client), "%s:8890"%ip)


if __name__ == '__main__':
    logging.info("Test machineinfo ...")
    unittest.main()
