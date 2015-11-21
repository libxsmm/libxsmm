#!/bin/bash

HERE=$(cd $(dirname $0); pwd -P)
GIT_DIR=${HERE}/../.git
LOCKFILE=${GIT_DIR}/.commit
CP=$(which cp)
RM=$(which rm)

if [[ -e ${GIT_DIR}/hooks ]] ; then
  ${CP} ${HERE}/pre-commit ${HERE}/post-commit ${GIT_DIR}/hooks
  ${RM} -f ${LOCKFILE}
fi
