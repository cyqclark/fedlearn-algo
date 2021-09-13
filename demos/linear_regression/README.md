# Federated LinearRegression based on QR decomposition

A demo of federated linear regression algorithm based on QR decomposition. The demo includes local 
training-inference and remote training (with ip and port) parts.

The following code runs in root path.


## Usage

To run the local demo, simply run `demo_local_qr.py`. 

For the remote version:

1) create a config file (for example `config.py`). A demo config
file is included as `config.py`

2) run the following command to create 3 clients:
`python demos/linear_regression/qrClient.py -I 0 -C demos/linear_regression/config.py`
`python demos/linear_regression/qrClient.py -I 1 -C demos/linear_regression/config.py`
`python demos/linear_regression/qrClient.py -I 2 -C demos/linear_regression/config.py`

3) run the following command to create coordinator:
`python demos/linear_regression/qrCoordinator.py -C demos/linear_regression/config.py`


## Reference

TBA