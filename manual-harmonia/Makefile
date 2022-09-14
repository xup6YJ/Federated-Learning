#Global setting
GITEA_ADMIN_USERNAME = gitea
GITEA_ADMIN_PASSWORD = password

AGGREGATOR_USERNAME = aggregator
AGGREGATOR_PASSWORD = 1qaz_WSX

LOGSERVER_USERNAME = logserver
LOGSERVER_PASSWORD = 1qaz_WSX

GITEA_API_SERVER_PROTOCOL = https
GITEA_SERVER_URL = gitea.corp.ailabs.tw
GITEA_SERVER_PORT = 443
GITEA_ADMIN_API_SERVER_URL = ${GITEA_API_SERVER_PROTOCOL}://${GITEA_ADMIN_USERNAME}:${GITEA_ADMIN_PASSWORD}@${GITEA_SERVER_URL}:${GITEA_SERVER_PORT}/api/v1
GITEA_ADMIN_URL = ${GITEA_API_SERVER_PROTOCOL}://${GITEA_ADMIN_USERNAME}:${GITEA_ADMIN_PASSWORD}@${GITEA_SERVER_URL}:${GITEA_SERVER_PORT}
GITEA_LOGSERVER_URL = ${GITEA_API_SERVER_PROTOCOL}://${LOGSERVER_USERNAME}@${GITEA_SERVER_URL}:${GITEA_SERVER_PORT}

DVC_ROOT_BUCKET = harmonia-bucket
DVC_BUCKET_URL = s3://${DVC_ROOT_BUCKET}
DVC_ENDPOINT_URL = https://minio.corp.ailabs.tw/
DVC_ACCESS_KEY_ID = AKIAIOSFODNN7EXAMPLE
DVC_SECRET_ACCESS_KEY = wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY

#Project setting
PROJECT_NAME = aspect
# Gitea
COVID_USERNAMES = ${PROJECT_NAME}_user1, ${PROJECT_NAME}_user2
COVID_PASSWORDS = 1qaz_WSX, 1qaz_WSX
COVID_EMAILS = ${PROJECT_NAME}_user1@${PROJECT_NAME}_test.com.tw, ${PROJECT_NAME}_user2@${PROJECT_NAME}_test.com.tw

TRAIN_PLAN_REPO = ${PROJECT_NAME}_lab3_trainplan
GLOBAL_REPO = ${PROJECT_NAME}_lab3_global
EDGE_REPOS = ${PROJECT_NAME}_edge_repo1, ${PROJECT_NAME}_edge_repo2
INIT_DVC_REPOS = ${GLOBAL_REPO}, ${EDGE_REPOS}

DELETE_REPOS = ${TRAIN_PLAN_REPO}, ${GLOBAL_REPO}, ${EDGE_REPOS}
DELETE_MINIO_BUCKETS = ${GLOBAL_REPO}, test1, test2

PRETRAINED_WEIGHT_PATH = ./storage/pretrained-weights/${PROJECT_NAME}/pretrain.ckpt

# Edge Pod
EDGE_NAMES = ${PROJECT_NAME}-edge1, ${PROJECT_NAME}-edge2

EDGE_OPERATOR_CONFIG_TEMPLATE = ./edge_template/config/operator/operator-config.yml
EDGE_APPLICATION_CONFIG_TEMPLATE = ./edge_template/config/application/${PROJECT_NAME}/application-config.yml
EDGE_DEPLOYMENT_TEMPLATE = ./edge_template/deployment.yml

# Application
REPO_ROOT_DIR = /repos
EDGE_ROOT_DIR = /manual-harmonia/${PROJECT_NAME}
# merge.ckpt is in global repo
EDGE_PRETRAINED_PATH = ${REPO_ROOT_DIR}/${GITEA_ADMIN_USERNAME}/${GLOBAL_REPO}/merged.ckpt

# train_config.yml
EDGE_TRANSFORMED_OUTPUT_PATH = ${EDGE_ROOT_DIR}/transformed_data
EDGE_DICOM_PATH = ${EDGE_ROOT_DIR}/harmonia_labels
EDGE_JSON_PATH = ${EDGE_ROOT_DIR}/harmonia_labels/label.json
EDGE_LABEL_MAP_PATH = ${EDGE_ROOT_DIR}/label_map.csv

# data_config.yaml parse_annotations.dataset_folder
EDGE_DATASET_FOLDER = ${EDGE_ROOT_DIR}/transformed_data

APPLICATION_IMAGE = xup6yj/aspect_fl

AGGREGATOR_LOGSERVER_CONFIG_TEMPLATE = ./aggregator_logserver_template/config.yml
AGGREGATOR_LOGSERVER_DEPLOYMENT_TEMPLATE = ./aggregator_logserver_template/deployment.yml
LOGSERVER_URL = ${PROJECT_NAME}-logserver

ROUND = 500

# ${GITEA_ADMIN_API_SERVER_URL} ${TRAIN_PLAN_REPO} ${LOGSERVER_REPO} ${GLOBAL_REPO} ${EDGE_REPOS} ${GITEA_ADMIN_USERNAME} ${AGGREGATOR_USERNAME} ${LOGSERVER_USERNAME} ${COVID_USERNAMES}
create_gitea_users:
	bash ./create_gitea_users.sh ${GITEA_ADMIN_API_SERVER_URL} "${COVID_USERNAMES}" "${COVID_PASSWORDS}" "${COVID_EMAILS}"

delete_gitea_users:
	bash ./delete_gitea_users.sh ${GITEA_ADMIN_API_SERVER_URL} "${COVID_USERNAMES}"

#Prepare harmonia env
init_harmonia: create_repos init_dvc push_pretrained_weight create_edges create_aggregator_logserver

create_repos:
	bash ./create_repos.sh ${GITEA_ADMIN_API_SERVER_URL} ${TRAIN_PLAN_REPO} ${GLOBAL_REPO} "${EDGE_REPOS}" ${GITEA_ADMIN_USERNAME} ${AGGREGATOR_USERNAME} ${LOGSERVER_USERNAME} "${COVID_USERNAMES}"

delete_repos:
	bash ./delete_repos.sh ${GITEA_ADMIN_API_SERVER_URL} ${GITEA_ADMIN_USERNAME} "${DELETE_REPOS}"

init_dvc:
	bash ./init_dvc.sh ${GITEA_ADMIN_URL} ${GITEA_ADMIN_USERNAME} ${DVC_BUCKET_URL} ${DVC_ENDPOINT_URL} ${DVC_ACCESS_KEY_ID} ${DVC_SECRET_ACCESS_KEY} "${INIT_DVC_REPOS}"

push_pretrained_weight:
	bash ./push_pretrained_weight.sh ${GITEA_ADMIN_URL} ${GITEA_ADMIN_USERNAME} ${GLOBAL_REPO} ${PRETRAINED_WEIGHT_PATH}

#Clean harmonia env
clean_harmonia: delete_repos  delete_edges delete_aggregator_logserver

clean_minio_storage:
	kubectl cp -n group-medical clean_minio_storage.sh `kubectl get pods -n group-medical -l app=minio -o template --template="{{(index .items 0).metadata.name }}"`:/
	kubectl exec -n group-medical `kubectl get pods -n group-medical -l app=minio -o template --template="{{(index .items 0).metadata.name }}"` -- bash /clean_minio_storage.sh ${DVC_ROOT_BUCKET} "${DELETE_MINIO_BUCKETS}"

create_aggregator_logserver:
	bash ./create_aggregator_logserver.sh ${PROJECT_NAME} ${AGGREGATOR_LOGSERVER_CONFIG_TEMPLATE} ${AGGREGATOR_LOGSERVER_DEPLOYMENT_TEMPLATE} ${AGGREGATOR_USERNAME} ${AGGREGATOR_PASSWORD} ${GITEA_ADMIN_USERNAME} ${GLOBAL_REPO} ${TRAIN_PLAN_REPO} "${EDGE_REPOS}" ${GITEA_LOGSERVER_URL}

delete_aggregator_logserver:
	bash ./delete_aggregator_logserver.sh ${PROJECT_NAME}

create_edges:
	bash ./create_edges.sh ${EDGE_OPERATOR_CONFIG_TEMPLATE} ${EDGE_APPLICATION_CONFIG_TEMPLATE} ${GITEA_ADMIN_USERNAME} ${GLOBAL_REPO} "${EDGE_REPOS}" ${TRAIN_PLAN_REPO} "${EDGE_NAMES}" "${COVID_USERNAMES}" "${COVID_PASSWORDS}" ${EDGE_PRETRAINED_PATH} ${EDGE_TRANSFORMED_OUTPUT_PATH} ${EDGE_DICOM_PATH} ${EDGE_JSON_PATH} ${EDGE_LABEL_MAP_PATH} ${EDGE_DATASET_FOLDER} ${EDGE_DEPLOYMENT_TEMPLATE} ${APPLICATION_IMAGE} ${EDGE_ROOT_DIR}

delete_edges:
	bash ./delete_edges.sh "${EDGE_NAMES}"

#Fix OCI runtime without "tar"
minio_install_tar:
	kubectl exec -n group-medical `kubectl get pods -n group-medical -l app=minio -o template --template="{{(index .items 0).metadata.name }}"` -- microdnf install tar

create_gitea_webhooks:
	bash ./create_gitea_webhooks.sh ${GITEA_ADMIN_API_SERVER_URL} ${GITEA_ADMIN_USERNAME} ${LOGSERVER_URL} ${GLOBAL_REPO} "${EDGE_REPOS}"

push_train_plan:
	bash ./push_train_plan.sh ${GITEA_ADMIN_URL} ${GITEA_ADMIN_USERNAME} ${TRAIN_PLAN_REPO} ${PROJECT_NAME} ${ROUND} "${EDGE_REPOS}"

start_training: create_gitea_webhooks push_train_plan

try:
	# echo "|${DVC_ACCESS_KEY_ID}|" || true
	# bash ./test.sh ${arr}
	# bash ./test.sh a b
