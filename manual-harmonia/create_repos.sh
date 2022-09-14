# GITEA_ADMIN_API_URL, TRAIN_PLAN_REPO, LOGSERVER_REPO, GLOBAL_REPO, EDGE_REPOS[]
# ADMIN_USERNAME, AGGREGATOR_USERNAME, LOGSERVER_USERNAME, EDGE_USERNAMES[]
GITEA_ADMIN_API_URL=$1
TRAIN_PLAN_REPO=$2
GLOBAL_REPO=$3
IFS=', ' read -a EDGE_REPOS <<< $4;

ADMIN_USERNAME=$5
AGGREGATOR_USERNAME=$6
LOGSERVER_USERNAME=$7
IFS=', ' read -a EDGE_USERNAMES <<< $8;

echo "GITEA_ADMIN_API_URL: $GITEA_ADMIN_API_URL"
echo "PUT: $GITEA_ADMIN_API_URL/repos/$ADMIN_USERNAME/$GLOBAL_REPO/collaborators/$AGGREGATOR_USERNAME"
# Create repositories

# Create train plan repo
echo "Create repo: $TRAIN_PLAN_REPO"
curl -X POST \
    -H "accept: application/json" \
    -H "Content-Type: application/json" \
    -d '{"name": "'$TRAIN_PLAN_REPO'", "auto_init": false, "private": true}' \
    $GITEA_ADMIN_API_URL/user/repos

# Create global repo
echo "Create repo: $GLOBAL_REPO"
curl -X POST \
    -H "accept: application/json" \
    -H "Content-Type: application/json" \
    -d '{"name": "'$GLOBAL_REPO'", "auto_init": false, "private": true}' \
    $GITEA_ADMIN_API_URL/user/repos

# Create edge repos
for EDGE_REPO in "${EDGE_REPOS[@]}"
do
    echo "Create repo: $EDGE_REPO"
    curl -X POST \
    -H "accept: application/json" \
    -H "Content-Type: application/json" \
    -d '{"name": "'$EDGE_REPO'", "auto_init": false, "private": true}' \
    $GITEA_ADMIN_API_URL/user/repos
done

# Add permissions
#Global repo - aggregator: write, logserver: read
curl -X PUT \
    -H "accept: application/json" \
    -H "Content-Type: application/json" \
    -d '{"permission": "write"}' \
    $GITEA_ADMIN_API_URL/repos/$ADMIN_USERNAME/$GLOBAL_REPO/collaborators/$AGGREGATOR_USERNAME

curl -X PUT \
    -H "accept: application/json" \
    -H "Content-Type: application/json" \
    -d '{"permission": "read"}' \
    $GITEA_ADMIN_API_URL/repos/$ADMIN_USERNAME/$GLOBAL_REPO/collaborators/$LOGSERVER_USERNAME

#Train plan repo - aggregator: read
curl -X PUT \
    -H "accept: application/json" \
    -H "Content-Type: application/json" \
    -d '{"permission": "read"}' \
    $GITEA_ADMIN_API_URL/repos/$ADMIN_USERNAME/$TRAIN_PLAN_REPO/collaborators/$AGGREGATOR_USERNAME

# Edges
for index in ${!EDGE_REPOS[@]}
do
    echo "Add permissions for User: ${EDGE_USERNAMES[index]} Repo: ${EDGE_REPOS[index]}"

    curl -X PUT \
      -H "accept: application/json" \
      -H "Content-Type: application/json" \
      -d '{"permission": "write"}' \
      $GITEA_ADMIN_API_URL/repos/$ADMIN_USERNAME/${EDGE_REPOS[index]}/collaborators/${EDGE_USERNAMES[index]}

    curl -X PUT \
      -H "accept: application/json" \
      -H "Content-Type: application/json" \
      -d '{"permission": "read"}' \
      $GITEA_ADMIN_API_URL/repos/$ADMIN_USERNAME/${EDGE_REPOS[index]}/collaborators/$AGGREGATOR_USERNAME

    curl -X PUT \
      -H "accept: application/json" \
      -H "Content-Type: application/json" \
      -d '{"permission": "read"}' \
      $GITEA_ADMIN_API_URL/repos/$ADMIN_USERNAME/${EDGE_REPOS[index]}/collaborators/$LOGSERVER_USERNAME

    curl -X PUT \
      -H "accept: application/json" \
      -H "Content-Type: application/json" \
      -d '{"permission": "read"}' \
      $GITEA_ADMIN_API_URL/repos/$ADMIN_USERNAME/$TRAIN_PLAN_REPO/collaborators/${EDGE_USERNAMES[index]}

    curl -X PUT \
      -H "accept: application/json" \
      -H "Content-Type: application/json" \
      -d '{"permission": "read"}' \
      $GITEA_ADMIN_API_URL/repos/$ADMIN_USERNAME/$GLOBAL_REPO/collaborators/${EDGE_USERNAMES[index]}

done
