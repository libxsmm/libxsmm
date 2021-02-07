#!/usr/bin/env sh
###############################################################################
# Copyright (c) Intel Corporation - All rights reserved.                      #
# This file is part of the LIBXSMM library.                                   #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# Further information: https://github.com/hfp/libxsmm/                        #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################
# Hans Pabst (Intel Corp.)
###############################################################################

HERE=$(cd "$(dirname "$0")" && pwd -P)
GIT_DIR=${HERE}/../.git
LOCKFILE=${GIT_DIR}/.commit
HOOKS="post-commit pre-commit prepare-commit-msg"

GIT=$(command -v git)
CHMOD=$(command -v chmod)
CP=$(command -v cp)
RM=$(command -v rm)

if [ -e "${GIT_DIR}/hooks" ] && [ "${GIT}" ] && \
   [ "${CHMOD}" ] && [ "${CP}" ] && [ "${RM}" ];
then
  # make sure the path to .gitconfig is a relative path
  ${GIT} config --local include.path ../.gitconfig 2>/dev/null
  for HOOK in ${HOOKS}; do
    ${CP} "${HERE}/${HOOK}" "${GIT_DIR}/hooks"
    ${CHMOD} +x "${GIT_DIR}/hooks/${HOOK}"
  done
  ${RM} -f "${LOCKFILE}-version"
fi
