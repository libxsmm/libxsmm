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
GIT=$(command -v git)

SHIFT=0
if [ "$1" ]; then
  SHIFT=$1
fi

NAME=$(${GIT} rev-parse --abbrev-ref HEAD 2>/dev/null)
MAIN=$(${GIT} describe --tags --match "[0-9]*" --abbrev=0 2>/dev/null)

if [ "${MAIN}" ]; then
  VERSION="${NAME}-${MAIN}"
  REVC=$(${GIT} rev-list --count "${MAIN}"..HEAD 2>/dev/null)
else
  VERSION=${NAME}
  REVC=$(${GIT} rev-list --count HEAD 2>/dev/null)
fi

echo "${VERSION}-$((REVC+SHIFT))"
