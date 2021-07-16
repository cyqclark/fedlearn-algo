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
Basic machine info class
"""
class MachineInfo:
    def __init__(self, ip: str = None, port: str = None, token: str = None):
        assert isinstance(ip, str), "ip must be in string type, but got %s type"%type(ip)
        assert isinstance(port, str), "port must be in string type, but got %s type"%type(port)
        assert isinstance(token, str), "token must be in string type, but got %s type"%type(token)
        self.ip = ip
        self.port = port
        self.token = token
        return

    def __str__(self):
        return self.ip+':'+self.port
