# Federated feature distributed neural network

A demo of Federated feature distributed neural network algorithm. The demo includes local training-inference and remote training (with ip and port) parts.

The following code runs in root path.


## Usage

To run the local demo, simply run `demo_local.py`. 

For the remote version:

### Star network topology
With start network topology, coordinator is deployed on a server which connect all the clients. In this case, one could run a training process demo as the following:

1) create a config file (for example `config.py`). A demo config
file is included as `config.py`

2) run the following command to create 2 clients:
`python demos/fdnn/tensorflow/client.py -I 1 -C demos/fdnn/tensorflow/config.py`
`python demos/fdnn/tensorflow/client.py -I 0 -C demos/fdnn/tensorflow/config.py`

3) run the following command to create coordinator:
`python demos/fdnn/tensorflow/coordinator.py -C demos/fdnn/tensorflow/config.py`

### Non-star network topology (experimental)
In this experimental code, coordinator is combined with active client so that the framework can support potentially any network topology. In this case, one could run a training process demo as the following:

1) create a config file (for example `config.py`). A demo config
file is included as `config.py`

2) run the following command to create 2 clients:
`python demos/fdnn/tensorflow/client.py -I 0 -C demos/fdnn/tensorflow/config.py`
`python demos/fdnn/tensorflow/client.py -I 1 -C demos/fdnn/tensorflow/config.py -F T`

Notice that client0 should be the last one to start.


## Reference

TBA