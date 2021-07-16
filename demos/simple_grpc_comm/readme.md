# Simple gRPC Communication Demonstration
In this demonstration, one server and three clients are built to show how to send/receive messages between these terminals. 

First, in client terminals, run the following commands, respectively.
```
python demos/simple_grpc_comm/test_alg_client.py -I 127.0.0.1 -P 8891 -T client_1
python demos/simple_grpc_comm/test_alg_client.py -I 127.0.0.1 -P 8892 -T client_2
python demos/simple_grpc_comm/test_alg_client.py -I 127.0.0.1 -P 8893 -T client_3
```

Second, start the server and complete the training.
```
python demos/simple_grpc_comm/test_alg_master.py
```

If try to use the server with serialized message, please try the following command.
```
python demos/simple_grpc_comm/test_alg_master_serialization.py
```