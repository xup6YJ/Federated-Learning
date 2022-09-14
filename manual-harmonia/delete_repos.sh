# GITEA_ADMIN_API_URL, ADMIN_USERNAME, REPOS[]
GITEA_ADMIN_API_URL=$1
ADMIN_USERNAME=$2
IFS=', ' read -a REPOS <<< $3;

echo "Input: $3"

# Delete edge repos
for REPO in "${REPOS[@]}"
do
    echo "Delete repo: $REPO"
    curl -X DELETE \
      "$GITEA_ADMIN_API_URL/repos/$ADMIN_USERNAME/$REPO"
done