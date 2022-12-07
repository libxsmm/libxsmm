#!/usr/bin/env bash
###############################################################################
# Copyright (c) Intel Corporation - All rights reserved.                      #
# This file is part of the LIBXSMM library.                                   #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# Further information: https://github.com/libxsmm/libxsmm/                    #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################
# Hans Pabst (Intel Corp.)
###############################################################################

CURL=$(command -v curl)
GIT=$(command -v git)
SED=$(command -v sed)

PRJ_DEFAULT="libxsmm/libxsmm"

if [ "${CURL}" ] && [ "${GIT}" ] && [ "${SED}" ]; then
  N=0
  PRJ="$(${GIT} remote get-url origin 2>/dev/null | ${SED} "s/..*\/\(..*\)\/\(..*\)\.git/\1\/\2/")"
  URL="https://api.github.com/repos/${PRJ:-${PRJ_DEFAULT}}/forks"
  FORKS="$(${CURL} -s "${URL}" | ${SED} -n "s/[[:space:]]*\"html_url\":[[:space:]]*\"..*\/\/..*\/\(..*\)\/\(..*\)\".*/\1\/\2/p")"
  if [ "${FORKS}" ]; then
    for FORK in ${FORKS}; do
      USER=$(echo "${FORK}" | ${SED} "s/\/..*//")
      AND=""
      echo -n "Fork ${USER}"
      if ${GIT} remote add "${USER}" "https://github.com/${FORK}.git" 2>/dev/null; then
        echo -n " added"
        AND=" and"
      fi
      if ${GIT} fetch "${USER}" 2>/dev/null; then
        echo -n "${AND} updated"
        N=$((N+1))
      fi
      echo "."
    done
    for USER in $(${GIT} remote 2>/dev/null); do
      if [ ! "$(echo "${FORKS}" | ${SED} -n "/${USER}/p")" ]; then
        if ${GIT} fetch "${USER}" 2>/dev/null; then
          echo "Fork ${USER} updated."
          N=$((N+1))
        elif [ "$(${GIT} remote -v 2>/dev/null | ${SED} -n "/${USER}[[:space:]]..*\/${USER}\/..*[[:space:]](fetch)/p")" ]; then
          ${GIT} remote remove "${USER}"
          echo "Fork ${USER} removed."
          N=$((N+1))
        fi
      fi
    done
    if [ "0" != "${N}" ]; then
      echo "Processed number of forks: ${N}"
    fi
  else
    >&2 echo "ERROR: \"${CURL} -s ${URL}\" failed!"
    exit 1
  fi
else
  >&2 echo "ERROR: missing prerequisites!"
  exit 1
fi
