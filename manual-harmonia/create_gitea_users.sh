GITEA_ADMIN_API_URL=$1

IFS=', ' read -a USERNAMES <<< $2;
IFS=', ' read -a PASSWORDS <<< $3;
IFS=', ' read -a EMAILS <<< $4;

echo "GITEA_ADMIN_API_URL: $GITEA_ADMIN_API_URL"

# Create user
for index in ${!USERNAMES[@]}
do
    echo "Create user: ${USERNAMES[index]}, password: ${PASSWORDS[index]}, email: ${EMAILS[index]}"
    # echo '{"username": "'${USERNAMES[index]}'", "password": "'${PASSWORDS[index]}'", "email": "'${EMAILS[index]}'", "must_change_password": "false"}'
    curl -X POST \
      -H "Accept: application/json" \
      -H "Content-Type: application/json" \
      -d '{"username": "'${USERNAMES[index]}'", "password": "'${PASSWORDS[index]}'", "email": "'${EMAILS[index]}'", "must_change_password": false}' \
      $GITEA_ADMIN_API_URL/admin/users
done
