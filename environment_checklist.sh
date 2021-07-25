# environment checklist
# usage 1: read and check the packages
# usage 2: run the script
echo "Run the checklist..."

yum -y update

echo "1. check development env..."

# install development environment
yum -y install gcc openssl-devel bzip2-devel libffi libffi-devel
yum -y install @development

echo "Development env checking finished."

echo "2. check python 3.6 env..."

# install python 3.6

yum -y install python3 python3-devel

echo "Python 3.6 checking finished."

## paillier related
### gmpy2, paillier
echo "3. check paillier packages..."

yum install -y gmp-devel mpfr-devel mpc-devel libmpc-devel
python3 -m pip install gmpy2 phe orjson

echo "Paillier packages checking finished."

echo "4. check numpy/scipy related..."
## numpy scipy related

python3 -m pip install intel-numpy intel-scipy pandas sklearn

## grpc related

python3 -m pip install grpcio grpcio-tools
