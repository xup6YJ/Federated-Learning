ROOT_BUCKET=$1
IFS=', ' read -a REPOS <<< $2;

# Delete repo folders
for REPO in "${REPOS[@]}"
do
    echo "Delete repo folder in minio: /data/$ROOT_BUCKET/$REPO/"
    rm -rf "/data/$ROOT_BUCKET/$REPO"
done