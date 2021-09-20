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
The config py file for random forest remote demo.
We pack the three config files into one py file for simplicity
"""
import socket
ip = socket.gethostbyname(socket.gethostname())

parameter = {
    "batch_size": 32,
    "num_epoch": 5,
}

client_train_file_path = ["data/classificationA/train0.csv",
                          "data/classificationA/train1.csv",
                        ]

client_inference_file_path = ["data/classificationA/inference0.csv",
                              "data/classificationA/inference1.csv",
                            ]

client_ip_and_port = ["%s:8891"%ip,
                      "%s:8892"%ip,
                    ]
coordinator_ip_and_port = "%s:8890"%ip

active_index = 0
active_label = "Outcome"

