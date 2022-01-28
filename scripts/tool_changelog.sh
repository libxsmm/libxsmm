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

GREP=$(command -v grep)
UNIQ=$(command -v uniq)
GIT=$(command -v git)

if [ "${GREP}"   ] && [ "${UNIQ}"  ] && [ "${GIT}" ];
then
  LASTRELEASE=$(${GIT} describe --match "[0-9]*" --abbrev=0)
  ${GIT} shortlog "${LASTRELEASE}..HEAD" \
  | ${GREP} -v "Merge branch " \
  | ${GREP} -v "Merge pull request " \
  | ${UNIQ}
else
  >&2 echo "Error: missing prerequisites!"
  exit 1
fi

