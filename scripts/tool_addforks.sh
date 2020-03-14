#!/usr/bin/env bash
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

CURL=$(command -v curl)
GREP=$(command -v grep)
CUT=$(command -v cut)
GIT=$(command -v git)

if [ "" != "${CURL}" ] && [ "" != "${GIT}" ] && \
   [ "" != "${GREP}" ] && [ "" != "${CUT}" ];
then
  for FORK in $(${CURL} -s https://api.github.com/repos/hfp/libxsmm/forks \
  | ${GREP} "\"html_url\"" | ${GREP} "libxsmm" | ${CUT} -d/ -f4);
  do
    echo "Adding fork ${FORK}..."
    ${GIT} remote add ${FORK} https://github.com/${FORK}/libxsmm.git
    ${GIT} fetch ${FORK}
  done
else
  echo "Error: missing prerequisites!"
  exit 1
fi

