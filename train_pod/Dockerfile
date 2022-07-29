FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-devel

RUN rm /etc/apt/sources.list.d/cuda.list
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
RUN apt-get update && apt-get install -y openslide-tools ca-certificates


COPY . /app

RUN apt-get update

RUN pip3 install -r /app/requirements.txt
RUN pip3 install protobuf==3.14.0 grpcio==1.34.0 grpcio-tools==1.33.1

# RUN apt-get update && apt-get install tk-dev && rm -r /var/lib/apt/lists/*
# RUN apt-get install python-tk python3-tk tk-dev


WORKDIR /app

CMD [ "python3", "fl_edge.py", "-o" , "/app/train_config.yml" ]