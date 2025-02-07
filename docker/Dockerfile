FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update --allow-releaseinfo-change && \
    apt-get install -y locales && \
    localedef -i en_US -f UTF-8 en_US.UTF-8

# 필수 패키지 설치
RUN apt-get install -y \
        python3 \
        python3-dev \
        libmariadb-dev \
        libpq-dev \
        net-tools \
        wget \
        nano \
        vim \
	git \
        lsof \
	libgl1 \
        iputils-ping \
        python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


# Anaconda 설치
WORKDIR /tmp
COPY ./Anaconda3-2024.10-1-Linux-x86_64.sh /tmp
RUN bash Anaconda3-2024.10-1-Linux-x86_64.sh -b -p /root/anaconda3 && \
    rm Anaconda3-2024.10-1-Linux-x86_64.sh

# PATH 환경 변수 설정
ENV PATH="/root/anaconda3/bin:${PATH}"

RUN pip install --upgrade pip

# Conda 초기화 및 Jupyter 설치
RUN /root/anaconda3/bin/conda init bash && \
    /root/anaconda3/bin/conda install -y jupyterlab jupyter

# Python 심볼릭 링크 설정
RUN rm /usr/bin/python3 && ln -s /root/anaconda3/bin/python /usr/bin/python3 && \
    ln -s /usr/bin/python3 /usr/local/bin/python

# 작업 디렉토리 설정
WORKDIR /smwik

CMD ["/bin/bash"]
