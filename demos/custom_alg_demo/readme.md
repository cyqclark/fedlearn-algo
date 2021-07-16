# Algorithm Flow Demonstration
The algorithm flow demonstrates a customized algorithm development (one server terminal with three client terminals) for training. Server should communicate with each client. The server and three clients could be sited on different machines or started by command line terminal in one machine.

In this demonstration, one server and three clients should be setup. IP, port and token should be given by user.

First, in client terminals, run the following commands, respectively.
```
python demos/custom_alg_demo/custom_client.py -I 127.0.0.1 -P 8891 -T client_1
python demos/custom_alg_demo/custom_client.py -I 127.0.0.1 -P 8892 -T client_2
python demos/custom_alg_demo/custom_client.py -I 127.0.0.1 -P 8893 -T client_3
```

Second, in the server terminal, run the following commands to start the server and complete a simulated training pipeline.
```
python demos/custom_alg_demo/custom_server.py
```