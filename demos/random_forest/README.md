# Federated RandomForest

A demo of federated random forest algorithm. The demo includes local 
training-inference and remote training (with ip and port) parts.

The following code runs in root path.


## Usage

To run the local demo, simply run `demo_local.py`. 

For the remote version:

1) create a config file (for example `config.py`). A demo config
file is included as `config.py`

2) create 3 terminals (either local or remote), run the following command to create 3 clients (one for each terminal):
`python demos/random_forest/client.py -I 0 -C demos/random_forest/config.py`
`python demos/random_forest/client.py -I 1 -C demos/random_forest/config.py`
`python demos/random_forest/client.py -I 2 -C demos/random_forest/config.py`

3) create a new terminal, run the following command to create coordinator:
`python demos/random_forest/coordinator.py -C demos/random_forest/config.py`


## Reference

TBA