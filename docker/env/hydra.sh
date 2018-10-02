#!/usr/bin/env bash
set -e
CMD=$@
DOCKER_ENV_DIR=$(dirname $(readlink -f $0 ))
SCT_DIR=$(dirname $(dirname ${DOCKER_ENV_DIR}))
VERSION=$(cat ${DOCKER_ENV_DIR}/version)
PY_PREREQS_FILE=requirements-python.txt
CENTOS_PREREQS_FILE=install-prereqs.sh
WORK_DIR=/sct

# if running on Build server
if [[ ${USER} == "jenkins" ]]; then
    echo "Running on Build Server..."
else
    TTY_STDIN="-it"
    TERM_SET_SIZE="export COLUMNS=`tput cols`; export LINES=`tput lines`;"
fi

if ! docker --version; then
    echo "Docker not installed!!! Please run 'install-hydra.sh'!"
    exit 1
fi

if docker images |grep scylladb/hydra.*${VERSION}; then
    echo "Image up-to-date"
else
    echo "Image with version $VERSION not found. Building..."
    sleep 7
    cd ${DOCKER_ENV_DIR}
    cp -f ${SCT_DIR}/${PY_PREREQS_FILE} .
    cp -f ${SCT_DIR}/${CENTOS_PREREQS_FILE} .
    docker build -t scylladb/hydra:v${VERSION} .
    rm -f ${PY_PREREQS_FILE} ${CENTOS_PREREQS_FILE}
    cd -
fi
# Check for SSH keys
${SCT_DIR}/get-qa-ssh-keys.sh

docker run --rm ${TTY_STDIN} --privileged \
    -h SCT-CONTAINER \
    -v /var/run:/run \
    -v ${SCT_DIR}:${WORK_DIR} \
    -v /sys/fs/cgroup:/sys/fs/cgroup:ro \
    -v /tmp:/tmp \
    -v /var/tmp:/var/tmp \
    -v ~:/root \
    -w ${WORK_DIR} \
    -e JOB_NAME=${JOB_NAME} \
    -e BUILD_URL=${BUILD_URL} \
    --net=host \
    scylladb/hydra:v${VERSION} \
    /bin/bash -c "/usr/libexec/postfix/master -w; ${TERM_SET_SIZE} eval ${CMD}"