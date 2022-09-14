GITEA_ADMIN_API_URL=$1
GITEA_ADMIN_USERNAME=$2
DVC_BUCKET_URL=$3
DVC_ENDPOINT_URL=$4
DVC_ACCESS_KEY_ID=$5
DVC_SECRET_ACCESS_KEY=$6
IFS=', ' read -a REPOS <<< $7;

for REPO in "${REPOS[@]}"
do
    echo "Init repo: $REPO"
    git clone "$GITEA_ADMIN_API_URL/$GITEA_ADMIN_USERNAME/$REPO.git"
    pushd "$REPO"
    sleep 2

    dvc init -q
    dvc remote add -fd myremote "$DVC_BUCKET_URL/$REPO"
    dvc remote modify myremote endpointurl $DVC_ENDPOINT_URL
    dvc remote modify myremote access_key_id $DVC_ACCESS_KEY_ID
    dvc remote modify myremote secret_access_key $DVC_SECRET_ACCESS_KEY
    sleep 2

    git add --all
    git commit -m setup --allow-empty
    git push origin master
    sleep 2

    popd
    rm -rf "$REPO"
    sleep 2
done