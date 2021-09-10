# fedlearn-algo dockerfile with tensorflow
# version: 0.1.5

#########################
# Note:
#
# Base image: centos 7.4
#########################
FROM centos:7.4.1708

# install java environment (optional)
RUN yum -y update \
    && yum install -y java-1.8.0-openjdk-devel

RUN echo "export JAVA_HOME=/usr/lib/jvm/java-1.8.0-openjdk-1.8.0.262.b10-0.el7_8.x86_64/jre/bin/java" > ~/.bash_profile

######################### 
# The fedlearn-algo image
#########################
# Install environment
RUN yum -y install gcc openssl-devel bzip2-devel libffi libffi-devel \
    && yum -y install python3 python3-devel \
    && yum install -y gmp-devel mpfr-devel mpc-devel libmpc-devel

RUN python3 -m pip install pip --upgrade

RUN python3 -m pip install gmpy2 phe \
    && python3 -m pip install intel-numpy intel-scipy \
    && python3 -m pip install grpcio grpcio-tools \
    && python3 -m pip install orjson pandas sklearn

######################### 
# The fedlearn-algo + tensorflow image
#########################

# Tensorflow environment
RUN python3 -m pip install tensorflow

# Copy files
Copy . /app
