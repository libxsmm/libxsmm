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

HERE=$(cd "$(dirname "$0")"; pwd -P)
GIT_DIR=${HERE}/../.git
LOCKFILE=${GIT_DIR}/.commit

GIT=$(command -v git)
CP=$(command -v cp)
RM=$(command -v rm)

if [ -e "${GIT_DIR}/hooks" ] && \
   [ "" != "${GIT}" ] && [ "" != "${CP}" ] && [ "" != "${RM}" ]; \
then
  # make sure the path to .gitconfig is a relative path
  ${GIT} config --local include.path ../.gitconfig 2>/dev/null
  ${CP} "${HERE}/post-commit" "${GIT_DIR}/hooks"
  ${CP} "${HERE}/pre-commit" "${GIT_DIR}/hooks"
  ${CP} "${HERE}/prepare-commit-msg" "${GIT_DIR}/hooks"
  ${RM} -f "${LOCKFILE}-version"
fi
