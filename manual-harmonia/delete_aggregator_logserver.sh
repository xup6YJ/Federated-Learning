PROJECT_NAME=$1

# Delete aggregator, logserver
kubectl delete -n group-medical -f "./$PROJECT_NAME-aggregator-logserver/config.yml"
kubectl delete -n group-medical -f "./$PROJECT_NAME-aggregator-logserver/deployment.yml"
rm -rf "./$PROJECT_NAME-aggregator-logserver"
