# Federated RandomForest

A demo of federated random forest algorithm. The demo includes local 
training-inference and remote training (with ip and port) parts.

The following code runs in root path.


## Usage

To run the local demo, simply run `demo_local.py`. 

For the remote version:

### Star network topology
With start network topology, coordinator is deployed on a server which connect all the clients. In this case, one could run a training process demo as the following:

1) create a config file (for example `config.py`). A demo config
file is included as `config.py`

2) create 3 terminals (either local or remote), run the following command to create 3 clients (one for each terminal):
`python demos/random_forest/client.py -I 0 -C demos/random_forest/config.py`
`python demos/random_forest/client.py -I 1 -C demos/random_forest/config.py`
`python demos/random_forest/client.py -I 2 -C demos/random_forest/config.py`

3) create a new terminal, run the following command to create coordinator:
`python demos/random_forest/coordinator.py -C demos/random_forest/config.py`

### Non-star network topology (experimental)
In this experimental code, coordinator is combined with active client so that the framework can support potentially any network topology. In this case, one could run a training process demo as the following:

1) create a config file (for example `config.py`). A demo config
file is included as `config.py`

2) run the following command to create 3 clients:
`python demos/random_forest/client.py -I 1 -C demos/random_forest/config.py`
`python demos/random_forest/client.py -I 2 -C demos/random_forest/config.py`
`python demos/random_forest/client.py -I 0 -C demos/random_forest/config.py -F T`

Notice that client0 should be the last one to start.


## Reference
Yao, Houpu, Jiazhou Wang, Peng Dai, Liefeng Bo, and Yanqing Chen. "An Efficient and Robust System for Vertically Federated Random Forest." arXiv preprint arXiv:2201.10761 (2022).
(priprint is available at https://arxiv.org/pdf/2201.10761.pdf)
