# Federated feature distributed neural network

A demo of Federated feature distributed neural network algorithm. The demo includes local training-inference and remote training (with ip and port) parts.

The following code runs in root path.


## Usage

To run the local demo, simply run `demo_local.py`. 

For the remote version:

1) create a config file (for example `config.py`). A demo config
file is included as `config.py`

2) run the following command to create 3 clients:
`python demos/fdnn/tensorflow/client.py -I 0 -C demos/fdnn/tensorflow/config.py`
`python demos/fdnn/tensorflow/client.py -I 1 -C demos/fdnn/tensorflow/config.py`

3) run the following command to create coordinator:
`python demos/fdnn/tensorflow/coordinator.py -C demos/fdnn/tensorflow/config.py`


## Reference

TBA