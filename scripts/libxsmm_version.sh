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
SORT=$(command -v sort)
TAIL=$(command -v tail)
GIT=$(command -v git)

SHIFT=0
if [ "$1" ]; then
  SHIFT=$1
fi

NAME=$(${GIT} rev-parse --abbrev-ref HEAD 2>/dev/null)
if [ "${SORT}" ] && [ "${TAIL}" ]; then
  MAIN=$(${GIT} tag | ${SORT} -n -t. -k1,1 -k2,2 -k3,3 | ${TAIL} -n1)
else
  MAIN=$(${GIT} describe --tags --match "[0-9]*" --abbrev=0 2>/dev/null)
fi

if [ "${MAIN}" ]; then
  VERSION="${NAME}-${MAIN}"
  REVC=$(${GIT} rev-list --count --no-merges "${MAIN}"..HEAD 2>/dev/null)
else
  VERSION=${NAME}
  REVC=$(${GIT} rev-list --count --no-merges HEAD 2>/dev/null)
fi

echo "${VERSION}-$((REVC+SHIFT))"
