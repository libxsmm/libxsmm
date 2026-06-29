#!/usr/bin/env sh
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
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd) || exit 1
RELEASE_FILE=${ROOT}/VERSION
SORT=$(command -v sort)
HEAD=$(command -v head)
GIT=$(command -v git)
CAT=$(command -v cat)
GREP=$(command -v grep)

SHIFT=0
if [ "$1" ]; then
  SHIFT=$1
fi

if [ ! -r "${RELEASE_FILE}" ] || [ "" = "${CAT}" ] || [ "" = "${GREP}" ]; then
  >&2 echo "ERROR: cannot read LIBXSMM release version!"
  exit 1
fi
RELEASE=$(${CAT} "${RELEASE_FILE}")
if ! echo "${RELEASE}" | ${GREP} -Eq '^[0-9]+\.[0-9]+\.[0-9]+$'; then
  >&2 echo "ERROR: invalid LIBXSMM release version: ${RELEASE}"
  exit 1
fi

NAME=
REVC=0
if [ "${GIT}" ] && ${GIT} -C "${ROOT}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  if [ "${SORT}" ] && [ "${HEAD}" ]; then
    MAIN=$(${GIT} -C "${ROOT}" tag | ${SORT} -nr -t. -k1,1 -k2,2 -k3,3 | ${HEAD} -n1)
  fi
  NAME=$(${GIT} -C "${ROOT}" rev-parse --abbrev-ref HEAD 2>/dev/null)
  if [ "${MAIN}" ]; then
    REVC=$(${GIT} -C "${ROOT}" rev-list --count --no-merges "${MAIN}"..HEAD 2>/dev/null)
  else
    REVC=$(${GIT} -C "${ROOT}" rev-list --count --no-merges HEAD 2>/dev/null)
  fi
fi

if [ "${NAME}" ]; then
  echo "${NAME}-${RELEASE}-$((REVC+SHIFT))"
else
  echo "${RELEASE}-$((REVC+SHIFT))"
fi
