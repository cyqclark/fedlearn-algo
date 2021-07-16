# demo in remote version
python demos/random_forest/client.py -I 0 -C demos/random_forest/config.py
python demos/random_forest/client.py -I 1 -C demos/random_forest/config.py
python demos/random_forest/client.py -I 2 -C demos/random_forest/config.py


python demos/random_forest/coordinator.py -C demos/random_forest/config.py