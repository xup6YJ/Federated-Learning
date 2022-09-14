GITEA_ADMIN_API_URL=$1
IFS=', ' read -a USERNAMES <<< $2;

# Delete user
for index in ${!USERNAMES[@]}
do
    echo "Delete user: ${USERNAMES[index]}"
    curl -X DELETE \
      -H "Accept: application/json" \
      -H "Content-Type: application/json" \
      $GITEA_ADMIN_API_URL/admin/users/${USERNAMES[index]}
done
