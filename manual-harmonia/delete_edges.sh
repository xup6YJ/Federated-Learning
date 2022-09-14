IFS=', ' read -a EDGE_NAMES <<< $1;

# Delete edge pods
for index in ${!EDGE_NAMES[@]}
do
    echo "Delete edge ${EDGE_NAMES[index]}"
    kubectl delete -n group-medical -f "./${EDGE_NAMES[index]}/operator-config.yml"
    kubectl delete -n group-medical -f "./${EDGE_NAMES[index]}/application-config.yml"
    kubectl delete -n group-medical -f "./${EDGE_NAMES[index]}/deployment.yml"
    rm -rf "./${EDGE_NAMES[index]}"
done