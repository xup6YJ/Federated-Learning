#Config template path
EDGE_OPERATOR_CONFIG_TEMPLATE=$1
EDGE_APPLICATION_CONFIG_TEMPLATE=$2

#Operator configs
GITEA_ADMIN_USERNAME=$3
GLOBAL_REPO=$4
IFS=', ' read -a EDGE_REPOS <<< $5;
TRAIN_PLAN_REPO=$6

IFS=', ' read -a EDGE_NAMES <<< $7;
IFS=', ' read -a EDGE_GITEA_USERS <<< $8;
IFS=', ' read -a EDGE_GITEA_PASSWORD <<< $9;

#Application configs
PRETRAINED_PATH=${10}
TRANSFORMED_OUTPUT_PATH=${11}
DICOM_PATH=${12}
JSON_PATH=${13}
LABEL_MAP_PATH=${14}
DATASET_FOLDER=${15}

# Edge deployment
EDGE_DEPLOYMENT_TEMPLATE=${16}
APPLICATION_IMAGE=${17}
EDGE_ROOT_DIR=${18}

for index in ${!EDGE_NAMES[@]}
do
    echo "Create edge ${EDGE_NAMES[index]}"

    mkdir -p ${EDGE_NAMES[index]}
    # Operator config
    cp $EDGE_OPERATOR_CONFIG_TEMPLATE "./${EDGE_NAMES[index]}/operator-config.yml"
    # ADD  FOR MacOS
    sed -i  "s#{EDGE_NAME}#${EDGE_NAMES[index]}#g" "./${EDGE_NAMES[index]}/operator-config.yml"
    sed -i  "s#{USERNAME}#${EDGE_GITEA_USERS[index]}#g" "./${EDGE_NAMES[index]}/operator-config.yml"
    sed -i  "s#{USER_TOKEN}#${EDGE_GITEA_PASSWORD[index]}#g" "./${EDGE_NAMES[index]}/operator-config.yml"
    sed -i  "s#{GITEA_ADMIN_USERNAME}#$GITEA_ADMIN_USERNAME#g" "./${EDGE_NAMES[index]}/operator-config.yml"
    sed -i  "s#{GLOBAL_REPO}#$GLOBAL_REPO#g" "./${EDGE_NAMES[index]}/operator-config.yml"
    sed -i  "s#{EDGE_REPO}#${EDGE_REPOS[index]}#g" "./${EDGE_NAMES[index]}/operator-config.yml"
    sed -i  "s#{TRAIN_PLAN_REPO}#$TRAIN_PLAN_REPO#g" "./${EDGE_NAMES[index]}/operator-config.yml"

    # Application config
    cp $EDGE_APPLICATION_CONFIG_TEMPLATE "./${EDGE_NAMES[index]}/application-config.yml"

    sed -i  "s#{EDGE_NAME}#${EDGE_NAMES[index]}#g" "./${EDGE_NAMES[index]}/application-config.yml"
    sed -i  "s#{PRETRAINED_PATHS}#$PRETRAINED_PATH#g" "./${EDGE_NAMES[index]}/application-config.yml"
    sed -i  "s#{TRANSFORMED_OUTPUT_PATH}#$TRANSFORMED_OUTPUT_PATH#g" "./${EDGE_NAMES[index]}/application-config.yml"
    sed -i  "s#{DICOM_PATH}#$DICOM_PATH#g" "./${EDGE_NAMES[index]}/application-config.yml"
    sed -i  "s#{JSON_PATH}#$JSON_PATH#g" "./${EDGE_NAMES[index]}/application-config.yml"
    sed -i  "s#{LABEL_MAP_PATH}#$LABEL_MAP_PATH#g" "./${EDGE_NAMES[index]}/application-config.yml"
    sed -i  "s#{DATASET_FOLDER}#$DATASET_FOLDER#g" "./${EDGE_NAMES[index]}/application-config.yml"

    # Edge Deployment
    cp $EDGE_DEPLOYMENT_TEMPLATE "./${EDGE_NAMES[index]}/deployment.yml"

    sed -i  "s#{EDGE_NAME}#${EDGE_NAMES[index]}#g" "./${EDGE_NAMES[index]}/deployment.yml"
    sed -i  "s#{APPLICATION_IMAGE}#$APPLICATION_IMAGE#g" "./${EDGE_NAMES[index]}/deployment.yml"
    sed -i  "s#{EDGE_ROOT_DIR}#$EDGE_ROOT_DIR#g" "./${EDGE_NAMES[index]}/deployment.yml"

    kubectl apply -n group-medical -f "./${EDGE_NAMES[index]}/operator-config.yml"
    kubectl apply -n group-medical -f "./${EDGE_NAMES[index]}/application-config.yml"
    kubectl apply -n group-medical -f "./${EDGE_NAMES[index]}/deployment.yml"

done
