FROM elasticsearch:7.15.2
RUN yum install -y https://centos7.iuscommunity.org/ius-release.rpm
RUN yum update && \
    yum install -y \
        python36u \
        python36u-libs \
        python36u-devel \
        python36u-pip \
        git && \
    yum clean all
WORKDIR /usr/share/elasticsearch/
RUN git clone https://github.com/vincnunes/productsearch-2.0.git && \
    python3.6 -m pip install -r /usr/share/elasticsearch/productsearch-2.0/requirements.txt
