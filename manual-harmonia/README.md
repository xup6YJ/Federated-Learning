# Manual Harmonia
## Prerequisite
OS: macOS Catalina

Environment:
```bash
brew install gnu-sed
pip3 install boto3==1.17.97 dvc==1.7.9 python-crontab==2.5.1 requests==2.25.1 lxml==4.6.2
```
Put pre-trains weight for covid in:
```bash
[Current Project]/storage/pretrained-weights/covid/covid19_0503mte11ep98+0502mte03ep21_fix.ckpt
```

Init a new project
1. Make sure all configs are ready
2. Create gitea users first
```
make create_gitea_users
```

## Use Harmonia

All required parameters are in `Makefile`

Application configs are in `edge_template/config/application/covid/application-config.yml`

Training data location:
```bash
# Kubernetes info
Namespace: group-medical
Volumn Name: manual-harmonia
subPath: edges/{EDGE_NAME}/application
mountPath: /manual-harmonia/{PROJECT_NAME}
```
```bash
# Path info
# from edge1 to edge4
Training data directory: edges/covid-edge[1,2,3,4]/application
```
```bash
# Example
# data_config.yaml parse_annotations
# dataset_folder: /manual-harmonia/covid/transformed_data
Mapping to
edges/covid-edge[1,2,3,4]/application/transformed_data
Other paths may like:
edges/covid-edge[1,2,3,4]/application/harmonia_labels
edges/covid-edge[1,2,3,4]/application/harmonia_labels/label.json
edges/covid-edge[1,2,3,4]/application/label_map.csv
```

Init training environment:
```bash
make init_harmonia
```

Start training:

**Make sure edges are `running` by `kubectl get pods -n group-medical | grep covid*`**
```bash
make start_training
```

Inspect running status:
```bash
kubectl logs [POD NAME] -n group-medical -c [CONTAINER NAME]  -f
#Example:
kubectl logs covid-edge1-d6fc99657-6z6l -n group-medical -c operator -f
```

Clean harmonia:
```bash
make clean_harmonia
```




## Gitea environment (Do not change)
```
admin: gitea/password
aggregator: aggregator/1qaz_WSX
edge1: edge1/1qaz_WSX
edge2: edge2/1qaz_WSX
edge3: edge3/1qaz_WSX
```