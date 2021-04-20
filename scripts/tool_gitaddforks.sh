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

USR=hfp
PRJ=libxsmm
URL="https://api.github.com/repos/${USR}/${PRJ}/forks"

if [ "${CURL}" ] && [ "${GIT}" ] && \
   [ "${GREP}" ] && [ "${CUT}" ];
then
  N=0
  for FORK in $(${CURL} -s ${URL} \
  | ${GREP} "\"html_url\"" | ${GREP} "${PRJ}" | ${CUT} -d/ -f4);
  do
    echo -n "Fork ${FORK} "
    if $(${GIT} remote add ${FORK} https://github.com/${FORK}/${PRJ}.git 2>/dev/null); then
      echo -n "added and "
    fi
    if $(${GIT} fetch ${FORK} 2>/dev/null); then
      echo "updated."
    else
      ${GIT} remote remove ${FORK} 2>/dev/null
      echo "removed."
    fi
    N=$((N+1))
  done
  if [ "0" != "${N}" ]; then
    echo "Processed number of forks: ${N}"
  else
    ${CURL} ${URL}
  fi
else
  >&2 echo "Error: missing prerequisites!"
  exit 1
fi

