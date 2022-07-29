from concurrent import futures
from grpcpb import service_pb2, service_pb2_grpc
from multiprocessing import Process
import grpc
import logging
import os
import time
import torch
import sys  #c

# sys.path.insert(0, '/repos/gitea/aspect_edge_repo1/')
# sys.path.insert(0, '/repos/gitea/aspect_edge_repo2/')

OPERATOR_URI = os.getenv("OPERATOR_URI", "127.0.0.1:8787")
APPLICATION_URI = os.getenv("APPLICATION_URI", "0.0.0.0:7878")
LOG_LEVEL = os.environ.get("LOG_LEVEL", "DEBUG")
REPO_ROOT = os.environ.get("REPO_ROOT", "/repos")
MODEL_FILENAME = os.environ.get("MODEL_FILENAME", "weights.tar")

AGGREGATE_SUCCESS = 0
AGGREGATE_CONDITION = 1
AGGREGATE_FAIL = 2

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=LOG_LEVEL,
    datefmt="%Y-%m-%d %H:%M:%S",
)

loop = True


def send_result(err):
    logging.info("config.GRPC_CLIENT_URI: [%s]", OPERATOR_URI)
    try:
        channel = grpc.insecure_channel(OPERATOR_URI)
        stub = service_pb2_grpc.AggregateServerOperatorStub(channel)
        res = service_pb2.AggregateResult(
            error=err,
        )
        response = stub.AggregateFinish(res)
    except grpc.RpcError as rpc_error:
        logging.error("grpc error: [%s]", rpc_error)
    except Exception as err:
        logging.error("got error: [%s]", err)

    logging.debug("sending grpc message succeeds, response: [%s]", response)


def merge(models, merged_output_path):
    logging.info(f"models: {models}")
    logging.info(f"merged_output_path: {merged_output_path}")

    merged = {}
    total = sum(model["size"] for model in models)
    logging.info(f"total images: {total}")
    for model in models:
        try:
            data = torch.load(model["path"], map_location="cpu")["state_dict"]
            # data = data.state_dict()  #c
        except:
            logging.info("model in models loaded failed")
        
        weight = model["size"] / total

        # for name, param in data.named_parameters():  #c
        #     merged[name] = merged.get(name, 0) + param * weight
        # del data

        for k in data:  
            merged[k] = merged.get(k, 0) + data[k] * weight
        del data

    torch.save({"state_dict": merged}, merged_output_path)
    logging.info("merged_output_saved")
    


def aggregate(models, aggregated_model):
    logging.debug(f"local models: {models}")
    logging.debug(f"output path: {aggregated_model.path}")

    models = [
        {"path": os.path.join(REPO_ROOT, m.path, "weights.ckpt"), "size": m.datasetSize}
        for m in models
        if os.path.isfile(os.path.join(REPO_ROOT, m.path, "weights.ckpt"))
    ]

    output_full_path = os.path.join(REPO_ROOT, aggregated_model.path, "merged.ckpt")

    # merge(models, output_full_path)
    # send_result(AGGREGATE_SUCCESS)

    try:  #c
        merge(models, output_full_path)
        send_result(AGGREGATE_SUCCESS)

    except Exception as err:
        logging.error("merge error: {}".format(err))


class AggregateServerServicer(service_pb2_grpc.AggregateServerAppServicer):
    def Aggregate(self, request, context):
        logging.debug("Aggregate")

        # Use another process do training
        Process(
            target=aggregate,
            args=(request.localModels, request.aggregatedModel),
            daemon=True,
        ).start()
        resp = service_pb2.Empty()
        logging.info(f"Sending response: {resp}")
        return resp

    def TrainFinish(self, _request, _context):
        logging.info("received TrainFinish message")
        global loop
        loop = False
        return service_pb2.Empty()


def serve():
    logging.basicConfig(level=logging.DEBUG)

    logging.info(f"Start server... {APPLICATION_URI}")

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    service_pb2_grpc.add_AggregateServerAppServicer_to_server(AggregateServerServicer(), server)
    server.add_insecure_port(APPLICATION_URI)
    server.start()
    while loop:
        time.sleep(10)
    server.stop()
    os._exit(os.EX_OK)


if __name__ == "__main__":
    serve()
