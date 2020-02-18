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
FLOCK=$(command -v flock)

if [ -d "$1" ]; then
  ABSDIR=$(cd "$1"; pwd -P)
elif [ -f "$1" ]; then
  ABSDIR=$(cd "$(dirname "$1")"; pwd -P)
else
  ABSDIR=$(cd "$(dirname "$0")"; pwd -P)
fi

shift
cd "${ABSDIR}"
if [ "" != "${FLOCK}" ]; then
  ${FLOCK} "${ABSDIR}" -c "$@"
else
  eval "$@"
fi

