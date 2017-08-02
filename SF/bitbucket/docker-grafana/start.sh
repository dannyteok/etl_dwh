#!/bin/sh
set -e

export PREFIX=${PREFIX:-"/"}
export BACKEND=${BACKEND:-"env"}

export NODE=${NODE:-""}
export AUTH_TYPE=${AUTH_TYPE:-""}
export AUTH_TOKEN=${AUTH_TOKEN:-""}

[ -n "$NODE" ] && export NODE="-node "$NODE
[ -n "$AUTH_TYPE" ] && export AUTH_TYPE="-auth-type "$AUTH_TYPE
[ -n "$AUTH_TOKEN" ] && export AUTH_TOKEN="-auth-token "$AUTH_TOKEN

mkdir /grafana/bin/conf/

/usr/local/bin/confd -prefix=$PREFIX -onetime -backend $BACKEND $NODE $AUTH_TYPE $AUTH_TOKEN

echo "Setting Environment Variables"
chmod +x /env.sh
. /env.sh

echo 'Starting Grafana'
exec ./grafana-server "$@" &
AddDataSource() {
  curl --user $GF_SECURITY_ADMIN_USER:$GF_SECURITY_ADMIN_PASSWORD \
    'http://localhost:3000/api/datasources' \
    -X POST \
    -H 'Content-Type: application/json;charset=UTF-8' \
    --data-binary \
    '{"name":"Prometheus","type":"prometheus","url":"http://prometheus.weave.local:9090","access":"proxy","isDefault":true}'
}
until AddDataSource; do
  echo 'Configuring Grafana...'
  sleep 1
done

AddDashboards() {
  curl --user $GF_SECURITY_ADMIN_USER:$GF_SECURITY_ADMIN_PASSWORD \
    'http://localhost:3000/api/dashboards/db' \
    -X POST \
    -H 'Content-Type: application/json;charset=UTF-8' \
    -d @"/grafana/dashboards/Activity.json"
  curl --user $GF_SECURITY_ADMIN_USER:$GF_SECURITY_ADMIN_PASSWORD \
    'http://localhost:3000/api/dashboards/db' \
    -X POST \
    -H 'Content-Type: application/json;charset=UTF-8' \
    -d @"/grafana/dashboards/Alerting.json"
  curl --user $GF_SECURITY_ADMIN_USER:$GF_SECURITY_ADMIN_PASSWORD \
    'http://localhost:3000/api/dashboards/db' \
    -X POST \
    -H 'Content-Type: application/json;charset=UTF-8' \
    -d @"/grafana/dashboards/Containers.json"
  curl --user $GF_SECURITY_ADMIN_USER:$GF_SECURITY_ADMIN_PASSWORD \
    'http://localhost:3000/api/dashboards/db' \
    -X POST \
    -H 'Content-Type: application/json;charset=UTF-8' \
    -d @"/grafana/dashboards/Images.json"
  curl --user $GF_SECURITY_ADMIN_USER:$GF_SECURITY_ADMIN_PASSWORD \
    'http://localhost:3000/api/dashboards/db' \
    -X POST \
    -H 'Content-Type: application/json;charset=UTF-8' \
    -d @"/grafana/dashboards/Nodes.json"
  curl --user $GF_SECURITY_ADMIN_USER:$GF_SECURITY_ADMIN_PASSWORD \
    'http://localhost:3000/api/dashboards/db' \
    -X POST \
    -H 'Content-Type: application/json;charset=UTF-8' \
    -d @"/grafana/dashboards/System.json"
  curl --user $GF_SECURITY_ADMIN_USER:$GF_SECURITY_ADMIN_PASSWORD \
    'http://localhost:3000/api/dashboards/db' \
    -X POST \
    -H 'Content-Type: application/json;charset=UTF-8' \
    -d @"/grafana/dashboards/Systems.json"
}
until AddDashboards; do
  echo 'Grafana Dashboards...'
  sleep 1
done

echo 'Done!'
wait
