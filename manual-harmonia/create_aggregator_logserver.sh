
PROJECT_NAME=$1
AGGREGATOR_LOGSERVER_CONFIG_TEMPLATE=$2
AGGREGATOR_LOGSERVER_DEPLOYMENT_TEMPLATE=$3
AGGREGATOR_USERNAME=$4
AGGREGATOR_PASSWORD=$5

GITEA_ADMIN_USERNAME=$6
GLOBAL_REPO=$7
TRAIN_PLAN_REPO=$8
IFS=', ' read -a EDGE_REPOS <<< $9;
GITEA_LOGSERVER_URL=${10}

mkdir -p "$PROJECT_NAME-aggregator-logserver"
cp $AGGREGATOR_LOGSERVER_DEPLOYMENT_TEMPLATE "./$PROJECT_NAME-aggregator-logserver/deployment.yml"
cp $AGGREGATOR_LOGSERVER_CONFIG_TEMPLATE "./$PROJECT_NAME-aggregator-logserver/config.yml"

#Deployment
sed -i  "s#{PROJECT_NAME}#$PROJECT_NAME#g" "./$PROJECT_NAME-aggregator-logserver/deployment.yml"
#Config
sed -i  "s#{PROJECT_NAME}#$PROJECT_NAME#g" "./$PROJECT_NAME-aggregator-logserver/config.yml"
sed -i  "s#{AGGREGATOR_USERNAME}#$AGGREGATOR_USERNAME#g" "./$PROJECT_NAME-aggregator-logserver/config.yml"
sed -i  "s#{AGGREGATOR_PASSWORD}#$AGGREGATOR_PASSWORD#g" "./$PROJECT_NAME-aggregator-logserver/config.yml"
sed -i  "s#{GITEA_ADMIN_USERNAME}#$GITEA_ADMIN_USERNAME#g" "./$PROJECT_NAME-aggregator-logserver/config.yml"
sed -i  "s#{GLOBAL_REPO}#$GLOBAL_REPO#g" "./$PROJECT_NAME-aggregator-logserver/config.yml"
sed -i  "s#{TRAIN_PLAN_REPO}#$TRAIN_PLAN_REPO#g" "./$PROJECT_NAME-aggregator-logserver/config.yml"

for (( index=${#EDGE_REPOS[@]}-1 ; index>=0 ; index-- )) ; do
    echo "Add repo in config: ${EDGE_REPOS[index]}"
    sed -i "/edgeModelRepoNames.*/a\      - $GITEA_ADMIN_USERNAME/${EDGE_REPOS[index]}" "./$PROJECT_NAME-aggregator-logserver/config.yml"
    sed -i "/modelRepos.*/a\      - gitHttpURL: $GITEA_LOGSERVER_URL/$GITEA_ADMIN_USERNAME/${EDGE_REPOS[index]}" "./$PROJECT_NAME-aggregator-logserver/config.yml"
done

sed -i "/modelRepos.*/a\      - gitHttpURL: $GITEA_LOGSERVER_URL/$GITEA_ADMIN_USERNAME/$GLOBAL_REPO" "./$PROJECT_NAME-aggregator-logserver/config.yml"

kubectl apply -n group-medical -f "./$PROJECT_NAME-aggregator-logserver/config.yml"
kubectl apply -n group-medical -f "./$PROJECT_NAME-aggregator-logserver/deployment.yml"
