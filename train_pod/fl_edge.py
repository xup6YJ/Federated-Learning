from concurrent import futures
from grpcpb import service_pb2, service_pb2_grpc
from multiprocessing import Process
import argparse
import fl_train
import grpc
import logging
import multiprocessing
import os
import shutil
import time


OPERATOR_URI = os.getenv("OPERATOR_URI") or "127.0.0.1:8787"
APPLICATION_URI = "0.0.0.0:7878"
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
event = multiprocessing.Event()
mgr = multiprocessing.Manager()
namespace = mgr.Namespace()
loop = True
cofig_path = 'train_config.yml'

trainingProcess = None


def train_model(event, namespace, yml_path, base_model, local_model_dir, epochCount):

    logging.info(f"base model path: [{base_model.path}]")
    logging.info(f"local model path: [{local_model_dir}]")
    logging.info(f"epoch count: [{epochCount}]")

    local_model_path = os.path.join("/repos", local_model_dir, "weights.ckpt")
    base_model_path = os.path.join("/repos", base_model.path, "merged.ckpt")

    namespace.pretrained_path = base_model_path
    event.set()

    logging.info("wait until the event is clear")
    while event.is_set():
        time.sleep(5)

    logging.info(f"model last epoch path: [{namespace.epoch_path}]")
    shutil.copyfile(namespace.epoch_path, local_model_path)

    logging.info(f"model datasetSize: {namespace.dataset_size}")
    logging.info(f"model metrics: {namespace.metrics}")
    logging.info(f"config.GRPC_CLIENT_URI: {OPERATOR_URI}")
    try:
        channel = grpc.insecure_channel(OPERATOR_URI)
        logging.info(f"grpc.insecure_channel: {OPERATOR_URI} Done.")
        stub = service_pb2_grpc.EdgeOperatorStub(channel)
        logging.info("service_pb2_grpc.EdgeOperatorStub Done.")
        result = service_pb2.LocalTrainResult(
            error=0, datasetSize=namespace.dataset_size, metrics=namespace.metrics
        )
        logging.info("service_pb2.LocalTrainResult Done.")
        response = stub.LocalTrainFinish(result, timeout=30)
        logging.info("stub.LocalTrainFinish Done.")
        logging.info(f"namespace: {namespace}")
        namespace = {}
        logging.info(f"clean namespace: {namespace}")
        logging.debug(f"sending grpc message succeeds, response: {response}")
        channel.close()
        logging.info("channel.close() Done.")
    except grpc.RpcError as rpc_error:
        logging.error(f"grpc error: {rpc_error}")
    except Exception as err:
        logging.error(f"got error: {err}")


class EdgeAppServicer(service_pb2_grpc.EdgeAppServicer):
    def TrainInit(self, request, context):
        logging.info("TrainInit, reset the current epoch, increase the version")
        global trainingProcess
        trainingProcess = Process(
            target=fl_train.train, args=(cofig_path, event, namespace)
        )
        trainingProcess.start()
        resp = service_pb2.Empty()
        logging.info(f"Sending response: {resp}")
        return resp

    def LocalTrain(self, request, context):
        global event
        global namespace
        global cofig_path
        logging.info("LocalTrain")
        p = Process(
            target=train_model,
            args=(
                event,
                namespace,
                cofig_path,
                request.baseModel,
                request.localModel.path,
                request.EpR,
            ),
        )
        p.start()
        resp = service_pb2.Empty()
        logging.info(f"Sending response: {resp}")
        return resp

    def TrainInterrupt(self, request, context):
        # Not Implemented
        return service_pb2.Empty()

    def TrainFinish(self, _request, _context):
        logging.info("TrainFinish")
        global loop
        loop = False
        return service_pb2.Empty()


def serve():
    logging.basicConfig(level=logging.DEBUG)
    logging.info(f"Start server... {APPLICATION_URI}")

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    service_pb2_grpc.add_EdgeAppServicer_to_server(EdgeAppServicer(), server)
    server.add_insecure_port(APPLICATION_URI)
    server.start()
    while loop:
        time.sleep(10)
    server.stop(None)
    time.sleep(200)
    os._exit(os.EX_OK)


if __name__ == "__main__":
    # Parse Yaml
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--out", type=str, help="Yaml name")
    args, unparsed = parser.parse_known_args()

    cofig_path = args.out

    serve()
