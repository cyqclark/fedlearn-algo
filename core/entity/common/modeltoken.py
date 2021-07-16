from datetime import datetime


def generate_token(algorithm_type: str, machine_token: str) -> str:
    now = datetime.now()
    time = now.strftime("%m_%d_%H_%M")
    return algorithm_type+"-"+machine_token+"-"+time


