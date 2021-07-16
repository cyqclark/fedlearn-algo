# gRPC Module Unit Tests

## gRPC message and algorithm message conversion
Three pairs of kernel functions in grpc_converter.py are executed in this unit test.
- **common_dict_msg_to_arrays** and **arrays_to_common_dict_msg**
- **create_grpc_message** and **parse_grpc_message**
- **common_msg_to_grpc_msg** and **grpc_msg_to_common_msg**

```
python tests/grpc_comm/test_grpc_converter.py
```

## One-to-one gRPC message communication
An one-to-one gRPC communication (one sender and one receiver) is built in this unit test.
```
python tests/grpc_comm/test_grpc_client.py
```
