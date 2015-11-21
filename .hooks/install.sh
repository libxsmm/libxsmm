#!/bin/bash

HERE=$(cd $(dirname $0); pwd -P)
GIT_DIR=${HERE}/../.git
CP=$(which cp)

if [[ -e ${GIT_DIR}/hooks ]] ; then
  ${CP} ${HERE}/pre-commit ${HERE}/post-commit ${GIT_DIR}/hooks
fi
