from concurrent import futures
from grpcpb import service_pb2, service_pb2_grpc
from multiprocessing import Process
import argparse
import fl_train as train
import grpc
import logging
import multiprocessing
import os
import shutil
import time
# import fl_train_t3_1_3t as train  #2


event = multiprocessing.Event()

namespace = None

loop = True
cofig_path = 'store_file3.yaml'  #2  #1

trainingProcess = None

logging.info("TrainInit, reset the current epoch, increase the version")

def subprocess():
    event.wait(30)
    event.clear()


def run():

    trainingProcess = Process(
    target=train.train, args=(cofig_path, event, namespace)
)

    trainingProcess.start()
    resp = service_pb2.Empty()
    logging.info(f"Sending response: {resp}")
    logging.info("finish training")
    event.clear()


if __name__ == '__main__':
    run()
