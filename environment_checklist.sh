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

## Note: intel no longer support single distribution of numpy/scipy with intel MKL.
## Here we use normal numpy scipy in pip.
## For users that would like to get the best performance, we suggest using numpy/scipy built with intel MKL.
## There are several ways to get numpy/scipy with MKL. For example, one can get it from intel distribution for python:
## https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/distribution-for-python.html#gs.av4cnp
## or one can use anaconda which has a built-in runtime of MKL.
python3 -m pip install numpy scipy pandas sklearn tensorboard datasets tornado


## grpc related

python3 -m pip install grpcio grpcio-tools
