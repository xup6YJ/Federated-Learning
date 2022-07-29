import argparse
import glob
import numpy as np
import os
import time
import torch
import yaml


def fedavg(models, output_path, factors=None, **kargs):
    """
    Federated Average
    Args:
        models:       [list of weight dict] model weights to be aggregated
        output_path:  [string] aggrgate model output path
    """

    # Set average factors
    factors = [1.0 / len(models)] * len(models) if factors is None else factors

    # FedAvg on weights
    weights = [torch.load(model, map_location="cpu")["state_dict"] for model in models]
    weight_keys = list(weights[0].keys())
    merged_weight = {"state_dict": {}}
    for weight_key in weight_keys:
        merged_weight["state_dict"][weight_key] = sum(
            [weight[weight_key] * factor for weight, factor in zip(weights, factors)]
        )

    torch.save(merged_weight, output_path)

    return


def receive_signal(signal_dir):
    """
    Recieve signal
    Args:
        signal_dir: [string] signal path
    Returns:
        signal: signal file
    """

    try:
        signal = glob.glob(os.path.join(signal_dir, "signal_*"))[0]
    except IndexError:
        signal = None

    return signal


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--out", type=str, help="Yaml name")
    args, unparsed = parser.parse_known_args()

    # load client yaml
    with open(os.path.join(".", args.out), "r") as f:
        cfg = yaml.load(f, yaml.Loader)
    num_of_client = len(cfg["client_yaml"])

    client_yamls = []
    for client_yaml in cfg["client_yaml"]:
        with open(client_yaml, "r") as f:
            client_yamls.append(yaml.load(f, yaml.Loader))

    client_paths = [
        os.path.join(
            os.getcwd(),
            client_yaml["exp"]["trainer"]["default_root_dir"] + "_exp",
            client_yaml["exp"]["logger"]["name"],
            client_yaml["exp"]["logger"]["version"],
        )
        for client_yaml in client_yamls
    ]

    client_signal_paths = [receive_signal(client_path) for client_path in client_paths]

    # check client consistency
    if all(
        client_yamls[idx]["exp"]["trainer"]["max_epochs"]
        == client_yamls[0]["exp"]["trainer"]["max_epochs"]
        for idx in range(num_of_client)
    ):
        epoch = client_yamls[0]["exp"]["trainer"]["max_epochs"]
    else:
        raise ValueError("Total epoch number should be the same for all clients")

    if all(
        client_yamls[idx]["federated"]["server_path"] == client_yamls[0]["federated"]["server_path"]
        for idx in range(num_of_client)
    ):
        server_path = client_yamls[0]["federated"]["server_path"]
    else:
        raise ValueError("Server path should be the same for all clients")

    client_current_epoch = None
    for idx in range(num_of_client):
        if client_current_epoch is not None:
            if not client_signal_paths[idx]:
                if client_current_epoch > 0:
                    raise ValueError("Current epoch difference should be within 1 for all clients")
            else:
                client_current_epoch_ = int(client_signal_paths[idx].split("_")[-1])
                if abs(client_current_epoch - client_current_epoch_) > 1:
                    raise ValueError("Current epoch difference should be within 1 for all clients")
                client_current_epoch = max(client_current_epoch, client_current_epoch_)
        else:
            if not client_signal_paths[idx]:
                client_current_epoch = -1
            else:
                client_current_epoch = int(client_signal_paths[idx].split("_")[-1])

    # load server progress
    server_signal_path = receive_signal(server_path)
    if not server_signal_path:
        server_current_epoch = -1
    else:
        server_current_epoch = int(server_signal_path.split("_")[-1])

    if client_current_epoch - server_current_epoch not in [0, 1]:
        raise ValueError("Current epoch difference should be within 1 for client and server")

    print("total epoch: ", epoch)
    print("server_path: ", server_path)
    print("client_current_epoch: ", client_current_epoch)
    print("server_current_epoch: ", server_current_epoch)

    if not os.path.exists(server_path):
        os.makedirs(server_path)

    remove_server_ckpt = [client_yaml["federated"]["validate_ckpt"] for client_yaml in client_yamls]
    remove_server_ckpt = (
        True if remove_server_ckpt[0] == "server" and len(set(remove_server_ckpt)) == 1 else False
    )

    # start aggregation process
    while server_current_epoch < epoch:

        checkpoint_edge = [  # noqa F841
            os.path.join(
                client_path,
                "checkpoints",
                "last_epoch={}.ckpt".format(server_current_epoch + 1),
            )
            for client_path in client_paths
        ]

        client_signal_paths = [receive_signal(client_path) for client_path in client_paths]
        if not all(client_signal_paths):
            continue

        client_current_epochs = [
            int(client_signal_path.split("_")[-1]) for client_signal_path in client_signal_paths
        ]
        client_current_epochs = np.array(client_current_epochs)
        if not np.all(client_current_epochs == client_current_epochs[0]):
            continue

        if client_current_epochs[0] > server_current_epoch:

            print("Epoch {} Aggregating".format(server_current_epoch + 1))

            aggregate_method = cfg["aggregator"]["method"]  # noqa F401
            aggregate_model = "{}.ckpt".format(server_current_epoch + 1)  # noqa F401
            aggregate_kargs = cfg["aggregator"]["args"]  # noqa F401

            eval(
                f"{aggregate_method}(checkpoint_edge, os.path.join(server_path,aggregate_model), **aggregate_kargs)"
            )

            # Remove ckpt and signal for previous epoch
            if remove_server_ckpt:
                try:
                    os.remove(os.path.join(server_path, "{}.ckpt".format(server_current_epoch)))
                except FileNotFoundError:
                    pass
            try:
                os.remove(os.path.join(server_path, "signal_{}".format(server_current_epoch)))
            except FileNotFoundError:
                pass

            server_current_epoch += 1
            open(os.path.join(server_path, "signal_{}".format(server_current_epoch)), "w").close()

        for client_signal_path in client_signal_paths:
            os.remove(client_signal_path)

        time.sleep(1)


if __name__ == "__main__":
    main()
