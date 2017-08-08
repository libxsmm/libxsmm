#!/bin/sh

HERE=$(cd $(dirname $0); pwd -P)
GIT_DIR=${HERE}/../.git
LOCKFILE=${GIT_DIR}/.commit

GIT=$(which git)
CP=$(which cp)
RM=$(which rm)

if [ -e ${GIT_DIR}/hooks ] && \
   [ "" != "${GIT}" ] && [ "" != "${CP}" ] && [ "" != "${RM}" ]; \
then
  # make sure the path to .gitconfig is a relative path
  ${GIT} config --local include.path ../.gitconfig 2> /dev/null
  ${CP} ${HERE}/post-commit ${GIT_DIR}/hooks
  ${CP} ${HERE}/pre-commit ${GIT_DIR}/hooks
  ${CP} ${HERE}/prepare-commit-msg ${GIT_DIR}/hooks
  ${RM} -f ${LOCKFILE}-version
  ${RM} -f ${LOCKFILE}-readme
fi
