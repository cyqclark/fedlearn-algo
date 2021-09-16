# Federated Kernel Learning
This folder implements the kernel binary classification algorithm. The algorithm uses approximate kernel mapping to encrypt
the sample features. Then a model is learned by minimizing a squared training loss.

The python code files are introduced as follows:
```
1. demo_server.py is the interface code running on server side.
2. demo_client.py is the interface code running on client side.
3. server_kernelmethod.py is the algorithm code on server side.
4. client_kernelmethod.py is the algorithm code on client side.
5. util.py contains several utility functions.
```

## Run kernel method training
Prepare the config files on server and clients. The configure files contain necessary parameter settings.

For server, the configure parameters include server ip, port and token, clients’ ip, port and token, mode.

For client, the configure parameters include mode (training or inference), ip address, port number,
model token, data path, feature names and model_path, kernel mapping parameters scale and dimension, model save path.

### Network topology demo 1: 3 clients with 1 coordinator 
Run **demo_server.py** and **demo_client.py** and set the right configure file, e.g.
```python
python ./demos/kernelmethod/demo_client.py -C ./demos/kernelmethod/config/config_train_client1.config
python ./demos/kernelmethod/demo_client.py -C ./demos/kernelmethod/config/config_train_client2.config
python ./demos/kernelmethod/demo_client.py -C ./demos/kernelmethod/config/config_train_client3.config
python demos/kernelmethod/demo_server.py -C ./demos/kernelmethod/config/config_train_server.config
```

### Network topology demo 2: 3 clients
In this case one only needs to start 3 clients. For example, one can run the following code to start a training process:
```python
python ./demos/kernelmethod/demo_client.py -C ./demos/kernelmethod/config/config_train_client2.config
python ./demos/kernelmethod/demo_client.py -C ./demos/kernelmethod/config/config_train_client3.config
python ./demos/kernelmethod/demo_client.py -C ./demos/kernelmethod/config/config_train_client1.config -F T
```
Notice that client1 should be the last one to start.

## Run kernel method inference
To run the kernel method inference example, the training process has to be conducted first.

Prepare the config files on server and clients.

For server, the configure parameters include server ip, port and token, clients’ ip, port and token, mode.

For client, the configure parameters include ip, port, token, data path, feature name and model path.

Run **demo_server.py** and **demo_client.py** and set the right configure file, e.g.
```python
python ./demos/kernelmethod/demo_client.py -C ./demos/kernelmethod/config/config_inference_client1.config
python ./demos/kernelmethod/demo_client.py -C ./demos/kernelmethod/config/config_inference_client2.config
python ./demos/kernelmethod/demo_client.py -C ./demos/kernelmethod/config/config_inference_client3.config
python demos/kernelmethod/demo_server.py -C ./demos/kernelmethod/config/config_inference_server.config
```

## Reference
[Federated Doubly Stochastic Kernel Learning for Vertically Partitioned Data](https://arxiv.org/pdf/2008.06197.pdf)

[Privacy-Preserving Asynchronous Vertical Federated Learning Algorithms for Multiparty Collaborative Learning](https://ieeexplore.ieee.org/abstract/document/9463409)
