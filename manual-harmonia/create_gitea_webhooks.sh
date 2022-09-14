# Create webhooks
GITEA_ADMIN_API_SERVER_URL=$1
ADMIN_USERNAME=$2
LOGSERVER_URL=$3
GLOBAL_REPO=$4
IFS=', ' read -a EDGE_REPOS <<< $5;

# Global Model
curl -X POST \
    -H "accept: application/json" \
    -H "Content-Type: application/json" \
    -d '{"active": true, "config": {"content_type": "json", "url": "'http://$LOGSERVER_URL:9080'"}, "events": ["push"], "type": "gitea"}' \
    $GITEA_ADMIN_API_SERVER_URL/repos/$ADMIN_USERNAME/$GLOBAL_REPO/hooks

# Edge Models
for EDGE_REPO in "${EDGE_REPOS[@]}"
do
    echo "Set webhook for: $EDGE_REPO"
    curl -X POST \
      -H "accept: application/json" \
      -H "Content-Type: application/json" \
      -d '{"active": true, "config": {"content_type": "json", "url": "'http://$LOGSERVER_URL:9080'"}, "events": ["push"], "type": "gitea"}' \
      $GITEA_ADMIN_API_SERVER_URL/repos/$ADMIN_USERNAME/$EDGE_REPO/hooks
done



