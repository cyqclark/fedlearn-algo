#!/bin/bash
# This script runs all the valid local test.
# The aim of this scipt is to ensure the code will
# not be broken by any changes.

# run a local random forest test
python demos/random_forest/demo_local.py

# run a local fdnn test
python demos/fdnn/demo_local.py

# run a local machineinfo test
python3 tests/entity/common/test_machineinfo.py

# run two local grpc communication tests
python tests/grpc_comm/test_grpc_converter.py
python tests/grpc_comm/test_grpc_client.py

# run the unit tests (optional, package coverate is required)
coverage run -m unittest discover tests/; coverage report -m
